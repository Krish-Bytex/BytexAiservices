import os
from dotenv import load_dotenv
from pinecone import Pinecone
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Configure Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "bytex-vectors")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

def get_gemini_response(prompt: str) -> str:
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)
    return response.text

def fetch_vectors_by_doc_id(doc_id: str) -> list[str]:
    # Query Pinecone for vectors with the given document ID in metadata
    results = index.query(
        vector=[0.0]*768,  # dummy vector; only using filter
        top_k=100,
        include_metadata=True,
        filter={"source_id": {"$eq": doc_id}}
    )
    return [match["metadata"]["text"] for match in results["matches"]]

def generate_reviewer_suggestions(doc_id: str) -> dict:
    texts = fetch_vectors_by_doc_id(doc_id)
    
    if not texts:
        return {"error": "No vectors found for this document ID."}

    combined_text = "\n".join(texts)
    prompt = f"""You are an AI reviewer. Analyze the following system requirement text for clarity, correctness, and formatting.
If there are suggestions to improve the information or make it more consistent, concise, or technically correct, list them clearly.

--- TEXT TO REVIEW ---
{combined_text}
--- END OF TEXT ---

Provide bullet-point suggestions for improvement only if necessary."""

    ai_response = get_gemini_response(prompt)

    return {
        "document_id": doc_id,
        "suggestions": ai_response
    }
