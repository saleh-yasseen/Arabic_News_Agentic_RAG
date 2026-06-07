import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from qdrant_client import models, qdrant_client
from qdrant_client.models import SparseVector
from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer

collection_name = "arabic_news"

client= qdrant_client.QdrantClient(path="./data/qdrant_db")
model = SentenceTransformer("aubmindlab/bert-base-arabertv02")
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

def retrieve(query: str, top_k: int =5, category_filter: str = None):
    dense_vec = model.encode(query, normalize_embeddings=True).tolist()
    sparse_vec = list(sparse_model.embed([query]))[0]
    query_filter = None
    