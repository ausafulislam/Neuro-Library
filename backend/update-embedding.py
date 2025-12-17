import os
import hashlib
import requests
import uuid
from bs4 import BeautifulSoup
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from dotenv import load_dotenv

load_dotenv()

SITEMAP_URL = os.getenv("SITEMAP_URL")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 300

app = FastAPI(title="Neuro Library RAG")

# Load embedding model
print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)
VECTOR_SIZE = embedder.get_sentence_embedding_dimension()
print(f"Embedding model loaded, vector size {VECTOR_SIZE}")

# Connect to Qdrant
print("Connecting to Qdrant...")
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Create collection if it doesn't exist
if COLLECTION_NAME not in [c.name for c in qdrant.get_collections().collections]:
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"Collection '{COLLECTION_NAME}' created.")
else:
    print(f"Collection '{COLLECTION_NAME}' exists. Using incremental update.")

# ===============================
# Helper functions
# ===============================


def fetch_sitemap_urls():
    print(f"Fetching sitemap: {SITEMAP_URL}")
    xml = requests.get(SITEMAP_URL, timeout=30).text
    soup = BeautifulSoup(xml, "xml")
    urls = [loc.text for loc in soup.find_all("loc")]
    print(f"Found {len(urls)} URLs")
    return urls


def fetch_page_text(url):
    html = requests.get(url, timeout=30).text
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("main") or soup.body
    return main.get_text(" ", strip=True) if main else ""


def chunk_text(text, chunk_size=CHUNK_SIZE):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i : i + chunk_size])


def generate_chunk_id(url, chunk_index):
    """Stable ID for chunk: URL + index hash"""
    return hashlib.md5(f"{url}-{chunk_index}".encode()).hexdigest()


# ===============================
# Ingest website into Qdrant
# ===============================


def ingest_site():
    print("Starting incremental ingestion...")
    urls = fetch_sitemap_urls()
    total_chunks = 0
    points = []

    for idx, url in enumerate(urls, start=1):
        print(f"Processing {idx}/{len(urls)} -> {url}")
        text = fetch_page_text(url)
        if not text:
            print("No content found, skipping")
            continue

        for i, chunk in enumerate(chunk_text(text)):
            chunk_id = generate_chunk_id(url, i)
            vector = embedder.encode(chunk).tolist()
            points.append(
                PointStruct(
                    id=chunk_id,
                    vector=vector,
                    payload={"text": chunk, "source": url},
                )
            )
            total_chunks += 1

        print(f"Chunks prepared so far: {total_chunks}")

    if points:
        print("Uploading vectors to Qdrant...")
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

    print("Ingestion completed.")
    print(f"Total pages processed: {len(urls)}")
    print(f"Total chunks uploaded: {total_chunks}")


# ===============================
# Run ingestion on startup
# ===============================


@app.on_event("startup")
def startup():
    ingest_site()


# ===============================
# Query API
# ===============================


class Query(BaseModel):
    question: str


@app.get("/")
def health():
    return {"status": "RAG server running"}


@app.post("/ask")
def ask(query: Query):
    print(f"User question: {query.question}")
    vector = embedder.encode(query.question).tolist()
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME, query=vector, limit=5
    ).points
    context = "\n".join(r.payload["text"] for r in results)
    sources = list(set(r.payload["source"] for r in results))
    print(f"Returned {len(results)} chunks")
    return {"context": context, "sources": sources}
