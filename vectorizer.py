import os
import uuid
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
INDEX_NAME = os.getenv("PINECONE_INDEX", "bytex-vectors")

# Initialize Pinecone (serverless)
pc = Pinecone(api_key=PINECONE_API_KEY)


# Delete existing index if it exists (needed when changing dimension!)
# if INDEX_NAME in pc.list_indexes().names():
#     pc.delete_index(INDEX_NAME)


# Create index if it doesn't exist
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,  # Must match your model's output
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
    )

# Load index
index = pc.Index(INDEX_NAME)

# Load HuggingFaceEmbeddings model
model = HuggingFaceEmbeddings()


def flatten_requirements(data: dict) -> list[str]:
    """
    Flattens nested system requirements into plain strings for vectorization.
    """
    flattened = []

    def recurse(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                recurse(v, prefix + k + ": ")
        elif isinstance(obj, list):
            for item in obj:
                recurse(item, prefix)
        elif obj:
            flattened.append(f"{prefix}{str(obj)}")

    recurse(data)
    return flattened


def vectorize_and_store(doc_id: str, software: str, body: dict) -> dict:
    """
    Vectorizes system requirements and stores them in Pinecone.
    """
    texts = flatten_requirements(body)
    if not texts:
        raise ValueError("Empty or invalid content — nothing to vectorize.")

    # Generate embeddings
    # ✅ NEW (correct usage with HuggingFaceEmbeddings)
    embeddings = model.embed_documents(texts)

    print("Vectorizing data!!")

    # Prepare metadata for Pinecone
    vectors = [
        {
            "id": str(uuid.uuid4()),
            "values": vector,
            "metadata": {
                "software": software.lower(),
                "source_id": doc_id,
                "text": text
            }
        }
        for vector, text in zip(embeddings, texts)
    ]

    # Upsert to Pinecone
    index.upsert(vectors=vectors)

    print("Upserted data in pinecone!!")

    return {
        "vectors_stored": len(vectors),
        "software": software,
        "source_id": doc_id
    }
