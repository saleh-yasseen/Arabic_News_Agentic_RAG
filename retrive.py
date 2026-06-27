from qdrant_client import models, QdrantClient
from qdrant_client.models import SparseVector
from fastembed import SparseTextEmbedding
from langchain_huggingface import HuggingFaceEmbeddings
try:
    embeddings = HuggingFaceEmbeddings(
        model_name="aubmindlab/bert-base-arabertv02",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print("Dense model loaded")
except Exception as e:
    print(f"Dense model error: {e}")
    raise

try:
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    print("Sparse model loaded")
except Exception as e:
    print(f"Sparse model error: {e}")
    raise

client = QdrantClient(path="./data/qdrant_db")

collection_name = "arabic_news"

def retrieve(query, top_k=5):
    dense_vec = embeddings.embed_query(query)

    sparse_vec = list(sparse_model.embed([query]))[0]

    results = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=dense_vec,
                using="",
                limit=20
            ),
            models.Prefetch(
                query=SparseVector(
                    indices=sparse_vec.indices.tolist(),
                    values=sparse_vec.values.tolist()
                ),
                using="sparse",
                limit=20
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k
    )
    return results.points

queries = [
    "ما هي آخر الأخبار السياسية؟",      # politics
    "نتائج كأس العالم لكرة القدم",        # sports
    "أخبار الاقتصاد والأسواق المالية"     # finance
]

for q in queries:
    print(f"\nQuery: {q}")
    results = retrieve(q, top_k=3)
    for r in results:
        print(f"  [{r.payload['category']}] {r.payload['text'][:100]}...")