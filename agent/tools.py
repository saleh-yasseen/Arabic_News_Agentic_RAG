import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from qdrant_client import models, QdrantClient
from qdrant_client.models import SparseVector
from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer

collection_name = "arabic_news"

client = QdrantClient(path="../data/qdrant_db")
model = SentenceTransformer("aubmindlab/bert-base-arabertv02")
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

def _retrieve(query: str, top_k: int =5, category_filter: str = None):
    dense_vec = model.encode(query, normalize_embeddings=True).tolist()
    sparse_vec = list(sparse_model.embed([query]))[0]
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
        prefetch=[
            models.Prefetch(query=dense_vec, using="", limit=20, filter=query_filter),
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

def search_news(query: str)->dict:
    points =_retrieve(query, top_k=5)
    return {
        "tool":"search_news",
        "query": query,
        "results": [
            {
                "id": str(p.id),
                "category": p.payload["category"],
                "text": p.payload["text"],
                "score": p.score
            }
            for p in points
        ]
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

if __name__ == "__main__":
    result = search_news("ما هي آخر الأخبار السياسية؟")
    client.close()
    print(result)