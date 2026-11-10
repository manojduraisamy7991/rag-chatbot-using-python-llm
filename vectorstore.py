
from pinecone import Pinecone
import os
from dotenv import load_dotenv
from typing import List

# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()

# Read Pinecone API key from .env
PINECONE_API_KEY = os.getenv("VECTOR_PINE_API_key")

if not PINECONE_API_KEY:
    raise ValueError("VECTOR_PINE_API_key is missing from .env")


# --------------------------------------------------
# 2. Initialize Pinecone client
# --------------------------------------------------

pinecone_client = Pinecone(
    api_key=PINECONE_API_KEY
)


# --------------------------------------------------
# 3. Connect to the existing Pinecone index
# --------------------------------------------------

INDEX_NAME = "belk-python-rag-voyage"

# Connect to the index created using create_index.py
index = pinecone_client.Index(INDEX_NAME)


# --------------------------------------------------
# 4. Store document chunks and embeddings
# --------------------------------------------------

def store_in_pinecone(
    chunks: List[str],
    embeddings: List[List[float]],
    namespace: str = "hr-policy"
):
    """
    Store document chunks and their corresponding
    Voyage AI embeddings in Pinecone.

    Args:
        chunks: Original text chunks extracted from PDF.
        embeddings: 1024-dimensional Voyage AI vectors.
        namespace: Namespace used to organize documents.
    """

    # Validate that chunks and embeddings have equal lengths
    if len(chunks) != len(embeddings):
        raise ValueError(
            "Number of chunks and embeddings must be equal."
        )

    # Return if there is no data to upload
    if not chunks:
        print("No chunks available for storage.")
        return

    # Prepare document records for Pinecone
    documents = []

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):

        # Validate embedding dimensions
        if len(embedding) != 1024:
            raise ValueError(
                f"Chunk {i} has {len(embedding)} dimensions. Expected 1024."
            )

        # Create a document containing text fields and its dense vector
        document = {
            "_id": f"chunk_{i}",
            "title": "HR Policy",
            "chunk_text": chunk,
            "chunk_embedding": embedding
        }

        # Add the document to the upload list
        documents.append(document)

    # Define the maximum number of documents per batch
    batch_size = 100

    # Upload documents in smaller batches
    for i in range(0, len(documents), batch_size):

        # Extract the current batch
        batch = documents[i:i + batch_size]

        # Upload the batch using Pinecone Documents API
        response = index.documents.upsert(
            documents=batch,
            namespace=namespace
        )

        # Display upload progress
        print(
            f"Uploaded batch: {i + 1} - "
            f"{i + len(batch)}"
        )

    # Display final upload status
    print(
        f"Successfully submitted {len(documents)} "
        f"documents to Pinecone!"
    )
