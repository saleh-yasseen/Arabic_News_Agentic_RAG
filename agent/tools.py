
import os
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Loading Qdrant client and models...")

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

from qdrant_client import models, qdrant_client
from qdrant_client.models import SparseVector
from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer
import cohere

co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))

print("Starting Qdrant client and model initialization...")

print("_____")

collection_name = "arabic_news"

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# QDRANT_PATH = os.path.join(BASE_DIR, "data", "qdrant_db")
# client = qdrant_client.QdrantClient(path=QDRANT_PATH)

client = qdrant_client.QdrantClient(host="localhost", port=6333)

print("client loaded")
model = SentenceTransformer("aubmindlab/bert-base-arabertv02")
print("dense model loaded")
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
print("sparse model loaded")

print("_____")

print("Qdrant client and models initialized successfully.")

def _rerank(query, points, top_k=5, category_filter=None):
    if not points:
        return points, {}
    if co is None:
        return points[:top_k], {}
    documents = [p.payload["text"] for p in points]
    try:
        response = co.rerank(
            model = "rerank-multilingual-v3.0",
            query=query,
            documents=documents,
            top_n=min(top_k, len(documents))
        )
        reranked_points = [points[r.index] for r in response.results]
        scores= {str(points[r.index].id): r.relevance_score for r in response.results}
        return reranked_points, scores
    except Exception as e:
        print(f"Error occurred while reranking: {e}")
        return points[:top_k], {}

def _retrieve(query: str, top_k: int =5, category_filter: str = None, mode="hybrid"):
    query_filter = None

    if category_filter:
        query_filter = models.Filter(
            must=[models.FieldCondition(
                key="category",
                match=models.MatchValue(value=category_filter)
            )]
        )    
    if mode == "dense":
        return _retrieve_dense_only(query, top_k, category_filter)
    if mode == "sparse":
        return _retrieve_sparse_only(query, top_k, category_filter)
    
    dense_vec = model.encode(query, normalize_embeddings=True).tolist()
    sparse_vec = list(sparse_model.embed([query]))[0]

    results = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(query=dense_vec, using="", limit=top_k*2, filter=query_filter),
            models.Prefetch(
                query=SparseVector(
                    indices=sparse_vec.indices.tolist(),
                    values=sparse_vec.values.tolist()
                ),
                using="sparse",
                limit=top_k * 2,
                filter=query_filter
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=query_filter,
        limit=top_k
    )
    return results.points

def _retrieve_dense_only(query, top_k=5, category_filter=None):
    dense_vec = model.encode(query, normalize_embeddings=True).tolist()
    query_filter = None
    if category_filter:
        query_filter = models.Filter(
            must=[models.FieldCondition(
                key="category",
                match=models.MatchValue(value=category_filter)
            )]
        )
    results = client.query_points(
        collection_name=collection_name,
        query=dense_vec,
        using="",
        query_filter=query_filter,
        limit=top_k
    )
    return results.points
def _retrieve_sparse_only(query, top_k=5, category_filter=None):
    sparse_vec = list(sparse_model.embed([query]))[0]
    query_filter = None
    if category_filter:
        query_filter = models.Filter(
            must= models.field_condition(
                key="category",
                match=models.MatchValue(value=category_filter)
            )
        )
    results = client.query_points(
        collection_name=collection_name,
        query = SparseVector(
            indices=sparse_vec.indices.tolist(),
            values=sparse_vec.values.tolist()
        ),
        using="sparse",
        query_filter=query_filter,
        limit=top_k
    )
    return results.points

def search_news(query: str, mode:str = "hybrid", top_k: int = 5, category_filter: str = None, use_reranker:bool = True, min_score: float =None) ->dict:
    pool_size = max(15, top_k *3)
    fused_points =_retrieve(query, top_k=pool_size,category_filter=category_filter, mode=mode)
    dense_points = _retrieve_dense_only(query, top_k=5,category_filter=category_filter)
    sparse_points = _retrieve_sparse_only(query, top_k=5,category_filter=category_filter)
    reranked_points, rerank_scores = _rerank(query, fused_points, top_k=5,category_filter=category_filter)

    if use_reranker:
        result_points, result_scores = _rerank(query, fused_points, top_k=top_k)
    else:
        result_points, result_scores = fused_points[:top_k], {}

    if min_score is not None:
        result_points = [
            p for p in result_points
            if (result_scores.get(str(p.id), p.score) >= min_score)
        ]

    def serialize(points, score_override=None):
        return [
                {
                    "id": str(p.id),
                    "category": p.payload["category"],
                    "text": p.payload["text"],
                    "score": (score_override.get(str(p.id), p.score) if score_override else p.score)
                }
                for p in points
            ]
    return{
        "tool":"search_news",
        "query": query,
        "results": serialize(reranked_points, rerank_scores),
        "comparison": {
            "dense": serialize(dense_points),
            "sparse": serialize(sparse_points),
            "fused": serialize(fused_points[:5]),
            "reranked": serialize(reranked_points, rerank_scores) if use_reranker else []
        }
    }
def summarize_topic(query: str) -> dict:
    points = _retrieve(query, top_k=5)
    combined_content = " ".join([point.payload["text"] for point in points  ])
    return {
        "tool": "summarize_topic",
        "query": query,
        "context": combined_content,
        "source_count": len(points)
    }
def compare_timeline(query: str, category: str = None ) -> dict:
    points = _retrieve(query, top_k=10,category_filter= category)
    return {
        "tool": "compare_timeline",
        "query" : query,
        "results": [
            {"text":point.payload["text"], "category": point.payload["category"]}
            for point in points
        ]
    }
def answer_direct(query:str) ->dict:
    return {
        "tool": "answer_direct",
        "query": query,
        "context": ""

    }

