
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Loading Qdrant client and models...")

from qdrant_client import models, qdrant_client
from qdrant_client.models import SparseVector
from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer

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

def search_news(query: str)->dict:
    fused_points =_retrieve(query, top_k=5)
    dense_points = _retrieve_dense_only(query, top_k=5)
    sparse_points = _retrieve_sparse_only(query, top_k=5)
    def serialize(points):
        return [
                {
                    "id": str(p.id),
                    "category": p.payload["category"],
                    "text": p.payload["text"],
                    "score": p.score
                }
                for p in points
            ]
    return{
        "tool":"search_news",
        "query": query,
        "results": serialize(fused_points),
        "comparison": {
            "dense": serialize(dense_points),
            "sparse": serialize(sparse_points),
            "fused": serialize(fused_points)
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

