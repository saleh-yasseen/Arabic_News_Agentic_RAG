import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import time
import hashlib
import requests
import trafilatura
from datetime import datetime, timezone

from qdrant_client import QdrantClient,models
from qdrant_client.models import (
    VectorParams, Distance, SparseVectorParams,
    SparseIndexParams, PointStruct, SparseVector
)
from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

CATEGORIES ={
    "politics":"politics",
    "economy":"ebusiness",
    "Technology": "tech",
    "Health": "health",
    "Culture": "culture",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ar,en-US;q=0.7,en;q=0.3",
    "Referer": "https://www.google.com/",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

LIVE_COLLECTION = "arabic_news_live"

client = QdrantClient(host="localhost", port=6333)
print(client.get_collection("arabic_news").points_count)
dense_model= SentenceTransformer("aubmindlab/bert-base-arabertv02")
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""],
)

SOURCES = {
    "Al Jazeera": {
        "domain": "https://www.aljazeera.net",
        "categories": {
            "Politics": "politics",
            "Economy": "ebusiness",
            "Sports": "sport",
            "Technology": "tech",
            "Health": "health",
            "Culture": "culture",
        },
    },
    "Al Arabiya": {
        "domain": "https://www.alarabiya.net",
        "categories": {
            "Politics": "politics",
            "Sports": "sport",
            "Economy": "aswaq",
            "Technology": "technology",
            "Health": "medicine-and-health",   # TODO: confirm slug — check alarabiya.net nav for صحة
            "Culture": "culture-and-art",  # TODO: confirm slug — check alarabiya.net nav for ثقافة وفن
        },
    },
    "BBC Arabic": {
        "domain": "https://www.bbc.com/arabic",
        "categories": {
            "General": None,  # BBC's taxonomy isn't slug-based — homepage pull only for now
        },
    },
}

CATEGORY_KEYWORDS = {
    "Politics": ["سياسة", "حكومة", "رئيس", "وزير", "انتخابات", "برلمان", "دبلوماسي", "أزمة"],
    "Economy": ["اقتصاد", "أسواق", "بورصة", "نفط", "عملة", "تجارة", "شركة", "استثمار", "بنك"],
    "Sports": ["رياضة", "كرة القدم", "مباراة", "بطولة", "كأس العالم", "لاعب", "منتخب", "الدوري"],
    "Technology": ["تكنولوجيا", "ذكاء اصطناعي", "تطبيق", "هاتف", "إنستغرام", "منصة رقمية", "برمجيات"],
    "Health": ["صحة", "مرض", "علاج", "طبي", "فيروس", "دواء", "مستشفى"],
    "Culture": ["ثقافة", "فن", "سينما", "أغنية", "فنان", "كتاب", "مهرجان", "رواية"],
}

def clean_text(text):
    text = re.sub(r'[اأإآء]', 'ا', text)
    text = re.sub(r'[\u064B-\u065F]', '', text)
    text = re.sub(r'\u0640', '', text)
    text = re.sub(r'[^\u0600-\u06FF0-9\s\.\،\؟\!]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def ensure_live_collection():
    if not client.collection_exists(LIVE_COLLECTION):
        client.create_collection(
            collection_name = LIVE_COLLECTION,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            sparse_vectors_config={
                "sparse":SparseVectorParams(index=SparseIndexParams(on_disk=False))
            }
        )
        print(f"Created collection '{LIVE_COLLECTION}' with dense and sparse vector configurations.")

def make_id(url, chunk_idx):
    return hashlib.md5(f"{url}_{chunk_idx}".encode()).hexdigest()

def get_article_links(domain, slug, limit=15):
    url = f"{domain}/{slug}" if slug else f"{domain}/"
    try:
        resp = SESSION.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"fetch failed for {url}",{e})
        return[]
    
    prefix = f"{domain}/{slug}/" if slug else f"{domain}/"
    pattern = re.escape(prefix) + r'[a-zA-Z0-9\-_/]+'
    candidates = set(re.findall(pattern,resp.text))
    links = [link for link in candidates if len(link) > len(prefix) + 10]
    return sorted(links)[:limit]

def extract_article(url):
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return None
    return trafilatura.extract(
        downloaded,
        include_comments= False,
        include_tables=False,
        favor_precision=True
    )

def classify_category(text):
    scores ={cat: 0 for cat in CATEGORY_KEYWORDS}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            scores[cat] += text.count(kw)
    best = max(scores, key =scores.get)
    return best if scores[best] > 0 else "General"

def get_bbc_feed_articles(limit=25):
    url = "https://bbc.github.io/world-service-rss/arabic.html"
    try:
        resp = SESSION.get(url, headers=HEADERS, timeout=30) # Fixed line
        resp.raise_for_status()
    except Exception as e:
        print(f"fetch failed for BBC feed: {e}")
        return []
        
    pattern = r'\[([^\]]+)\]\((https://www\.bbc\.(?:com|co\.uk)/arabic/[^\)\s]+)\)'
    matches = re.findall(pattern, resp.text)
    seen_urls = set()
    articles = []
    for title, article_url in matches:
        clean_url = article_url.split('?')[0]
        if clean_url in seen_urls:
            continue
        seen_urls.add(clean_url)
        articles.append((title, clean_url))
    return articles[:limit]  

def ingest_category(source_name, category_name, domain, slug):
    if slug is None:
        print(f"    [{source_name} / {category_name}] no slug set — skipping")
        return 0

    links = get_article_links(domain, slug)
    print(f"    [{source_name} / {category_name}] {len(links)} links found")
    count = 0

    for url in links:
        raw_text = extract_article(url)
        if not raw_text or len(raw_text) < 200:
            continue

        cleaned = clean_text(raw_text)
        chunks = splitter.split_text(cleaned)
        points = []

        for i, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if len(chunk) < 20:
                continue
            dense_vec = dense_model.encode(chunk, normalize_embeddings=True).tolist()
            sparse_vec = list(sparse_model.embed([chunk]))[0]

            points.append(PointStruct(
                id=make_id(url, i),
                vector={
                    "": dense_vec,
                    "sparse": SparseVector(
                        indices=sparse_vec.indices.tolist(),
                        values=sparse_vec.values.tolist()
                    )
                },
                payload={
                    "text": chunk,
                    "category": category_name,
                    "source": source_name,
                    "url": url,
                    "ingested_at": datetime.now(timezone.utc).isoformat()
                }
            ))

        if points:
            client.upsert(collection_name=LIVE_COLLECTION, points=points)
            count += len(points)

        time.sleep(1)

    return count

def ingest_bbc():
    articles = get_bbc_feed_articles()
    print(f"    [BBC Arabic] {len(articles)} articles found")
    count = 0

    for title, url in articles:
        raw_text = extract_article(url)
        if not raw_text or len(raw_text) < 200:
            continue

        category = classify_category(title + " " + raw_text[:300])
        cleaned = clean_text(raw_text)
        chunks = splitter.split_text(cleaned)
        points = []

        for i, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if len(chunk) < 20:
                continue
            dense_vec = dense_model.encode(chunk, normalize_embeddings=True).tolist()
            sparse_vec = list(sparse_model.embed([chunk]))[0]

            points.append(PointStruct(
                id=make_id(url, i),
                vector={
                    "": dense_vec,
                    "sparse": SparseVector(
                        indices=sparse_vec.indices.tolist(),
                        values=sparse_vec.values.tolist()
                    )
                },
                payload={
                    "text": chunk,
                    "category": category,
                    "source": "BBC Arabic",
                    "url": url,
                    "ingested_at": datetime.now(timezone.utc).isoformat()
                }
            ))

        if points:
            client.upsert(collection_name=LIVE_COLLECTION, points=points)
            count += len(points)

        time.sleep(1)

    return count

def ingest_all():
    ensure_live_collection()
    total = 0
    for source_name, config in SOURCES.items():
        print(f"\n{source_name}")
        if source_name == "BBC Arabic":
            n = ingest_bbc()
            print(f"  {n} chunks indexed")
            total += n
        else:
            for category_name, slug in config["categories"].items():
                n = ingest_category(source_name, category_name, config["domain"], slug)
                print(f"  {category_name}: {n} chunks indexed")
                total += n
                time.sleep(2)
    print(f"\nDone. {total} chunks total across {len(SOURCES)} sources.")


if __name__ == "__main__":
    ingest_all()