import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import time
import hashlib
import requests
import trafilatura
from datetime import datetime, timezone

from qdrant_client import QdrantClient,models
from qdrant_client.models import (vector_params, Distance, PointStruct,SparseIndexParams, sparse_vector_params, SparseVector)
from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

CATEGORIES ={
    "politics":"politics",
    "economy":"ebusiness",
    "Technology": "tech",
    "Health": "health",
    "Culture": "culture",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ArabicNewsRAG/1.0; portfolio project, non-commercial)"}
LIVE_COLLECTION = "arabic_news_live"

client = QdrantClient(host="localhost", port=6333)
dense_model= SentenceTransformer("aubmindlab/bert-base-arabertv02")
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""],
)

