import os
import uuid
import traceback
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
INDEX_NAME = os.getenv("PINECONE_INDEX", "bytex-vectors")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# Pinecone setup
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if not exists
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
    )

index = pc.Index(INDEX_NAME)

# Correct configuration
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
embedding_model = genai.EmbeddingModel(model_name="models/embedding-001")

def get_gemini_embeddings(texts: list[str]) -> list[list[float]]:
    embeddings = []
    for text in texts:
        try:
            result = embedding_model.embed(content=text)
            embeddings.append(result["embedding"])
        except Exception as e:
            print(f"❌ Failed to embed: {text[:30]}... — {str(e)}")
            embeddings.append([0.0] * 768)  # fallback, will be filtered later
    return embeddings

# Flatten nested requirements
def flatten_requirements(data: dict) -> list[str]:
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

# Main function
def vectorize_and_store(doc_id: str, software: str, body: dict) -> dict:
    try:
        print(f"\n--- Vectorizing: {software} (ID: {doc_id}) ---")

        print("Step 1: Flattening requirements")
        texts = flatten_requirements(body)
        print(f"Flattened {len(texts)} texts")

        if not texts:
            raise ValueError("Empty or invalid content — nothing to vectorize.")

        print("Step 2: Getting Gemini embeddings")
        embeddings = get_gemini_embeddings(texts)
        print(f"Generated {len(embeddings)} embeddings")

        print("Step 3: Preparing vectors")
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

        print("Step 4: Upserting to Pinecone...")
        index.upsert(vectors=vectors)
        print("✅ Upserted to Pinecone successfully!")

        return {
            "vectors_stored": len(vectors),
            "software": software,
            "source_id": doc_id
        }

    except Exception as e:
        print("❌ Error during vectorization:")
        traceback.print_exc()
        raise e











# version 1: working but getting memory outof error on render due to embedding model size.

# import os
# import traceback
# import uuid
# from dotenv import load_dotenv
# from pinecone import Pinecone, ServerlessSpec
# # from sentence_transformers import SentenceTransformer
# # from langchain_huggingface import HuggingFaceEmbeddings

# # Load environment variables
# load_dotenv()

# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
# PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
# INDEX_NAME = os.getenv("PINECONE_INDEX", "bytex-vectors")

# # Initialize Pinecone (serverless)
# pc = Pinecone(api_key=PINECONE_API_KEY)


# # Delete existing index if it exists (needed when changing dimension!)
# # if INDEX_NAME in pc.list_indexes().names():
# #     pc.delete_index(INDEX_NAME)


# # Create index if it doesn't exist
# if INDEX_NAME not in pc.list_indexes().names():
#     pc.create_index(
#         name=INDEX_NAME,
#         dimension=768,  # Must match your model's output
#         metric="cosine",
#         spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
#     )

# # Load index
# index = pc.Index(INDEX_NAME)




# def flatten_requirements(data: dict) -> list[str]:
#     """
#     Flattens nested system requirements into plain strings for vectorization.
#     """
#     flattened = []

#     def recurse(obj, prefix=""):
#         if isinstance(obj, dict):
#             for k, v in obj.items():
#                 recurse(v, prefix + k + ": ")
#         elif isinstance(obj, list):
#             for item in obj:
#                 recurse(item, prefix)
#         elif obj:
#             flattened.append(f"{prefix}{str(obj)}")

#     recurse(data)
#     return flattened

# def vectorize_and_store(doc_id: str, software: str, body: dict) -> dict:
#     """
#     Vectorizes system requirements and stores them in Pinecone.
#     """
#     try:
#         print(f"\n--- Vectorizing: {software} (ID: {doc_id}) ---")

#         from langchain_huggingface import HuggingFaceEmbeddings  # moved inside
#         model = HuggingFaceEmbeddings()  # moved inside

#         print("Step 1: Flattening requirements")
#         texts = flatten_requirements(body)
#         print("Flattened texts:", texts)

#         if not texts:
#             raise ValueError("Empty or invalid content — nothing to vectorize.")

#         print("Step 2: Generating embeddings")
#         embeddings = model.embed_documents(texts)
#         print(f"Generated {len(embeddings)} embeddings.")

#         print("Step 3: Preparing vectors for Pinecone")
#         vectors = [
#             {
#                 "id": str(uuid.uuid4()),
#                 "values": vector,
#                 "metadata": {
#                     "software": software.lower(),
#                     "source_id": doc_id,
#                     "text": text
#                 }
#             }
#             for vector, text in zip(embeddings, texts)
#         ]

#         print("Step 4: Upserting to Pinecone...")
#         index.upsert(vectors=vectors)
#         print("✅ Successfully upserted to Pinecone!")

#         return {
#             "vectors_stored": len(vectors),
#             "software": software,
#             "source_id": doc_id
#         }

#     except Exception as e:
#         print("❌ Error during vectorization process:")
#         traceback.print_exc()
#         raise e

