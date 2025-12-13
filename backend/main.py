import requests
import os
from dotenv import load_dotenv
import xml.etree.ElementTree as ET
import trafilatura
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import cohere


# Enviroment variables setup
load_dotenv()
sitemap_url = os.getenv("SITEMAP_URL")
collection_name = os.getenv("COLLECTION_NAME")

# qdrant cloud
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

# cohere embedding
cohere_api_key = os.getenv("COHERE_API_KEY")
cohere_embed_model = os.getenv("COHERE_EMBED_MODLE")

cohere_client = cohere.Client("key-here")

# Connect to Qdrant Cloud
qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)


# extracting urls from sitemap
def get_all_urls(sitemap_url):
    xml = requests.get(sitemap_url).text
    root = ET.fromstring(xml)

    urls = []
    for child in root:
        loc_tag = child.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")
        if loc_tag is not None:
            urls.append(loc_tag.text)

    print("\nFOUND URLS:")
    for u in urls:
        print(" -", u)

    return urls


# download page from sitemap urls and extracct the extract text


def extract_text_from_url(url):
    html = requests.get(url).text
    text = trafilatura.extract(html)

    if not text:
        print("[WARNING] No text extracted from:", url)

    return text


# Chunk the text


def chunk_text(text, max_chars=1200):
    chunks = []
    while len(text) > max_chars:
        split_pos = text[:max_chars].rfind(". ")
        if split_pos == -1:
            split_pos = max_chars
        chunks.append(text[:split_pos])
        text = text[split_pos:]
    chunks.append(text)
    return chunks


# Create embedding
def embed(text):
    response = cohere_client.embed(
        model=cohere_embed_model,
        input_type="search_query",  # Use search_query for queries
        texts=[text],
    )
    return response.embeddings[0]  # Return the first embedding


# storing embedded data in Qdrant Cloud
def create_collection():
    print("\nCreating Qdrant collection...")
    qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=1024, distance=Distance.COSINE  # Cohere embed-english-v3.0 dimension
        ),
    )


def save_chunk_to_qdrant(chunk, chunk_id, url):
    vector = embed(chunk)

    qdrant.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=chunk_id,
                vector=vector,
                payload={"url": url, "text": chunk, "chunk_id": chunk_id},
            )
        ],
    )


# MAIN INGESTION PIPELINE
def ingest_book():
    urls = get_all_urls(sitemap_url)

    create_collection()

    global_id = 1

    for url in urls:
        print("\nProcessing:", url)
        text = extract_text_from_url(url)

        if not text:
            continue

        chunks = chunk_text(text)

        for ch in chunks:
            save_chunk_to_qdrant(ch, global_id, url)
            print(f"Saved chunk {global_id}")
            global_id += 1

    print("\n✔️ Ingestion completed!")
    print("Total chunks stored:", global_id - 1)


if __name__ == "__main__":
    ingest_book()
