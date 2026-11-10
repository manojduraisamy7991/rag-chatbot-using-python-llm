# # 


# from openai import OpenAI
# from dotenv import load_dotenv
# import os
# from typing import List

# load_dotenv()

# api_key = os.getenv("OPENAI_API_KEY")

# if not api_key:
#     raise ValueError("OPENAI_API_KEY is missing from .env")

# client = OpenAI(api_key=api_key)

# EMBEDDING_MODEL = "text-embedding-3-small"


# def embed_chunks(chunks: List[str]) -> List[List[float]]:
#     """Generate embeddings for multiple text chunks."""

#     if not chunks:
#         return []

#     response = client.embeddings.create(
#         input=chunks,
#         model=EMBEDDING_MODEL
#     )

#     # Preserve the original chunk order
#     sorted_data = sorted(response.data, key=lambda item: item.index)

#     embeddings = [
#         item.embedding for item in sorted_data
#     ]

#     return embeddings



import voyageai
from dotenv import load_dotenv
import os
from typing import List

load_dotenv()

api_key = os.getenv("VOYAGE_API_KEY")

if not api_key:
    raise ValueError("VOYAGE_API_KEY is missing from .env")

client = voyageai.Client(api_key=api_key)

EMBEDDING_MODEL = "voyage-4-lite"


def embed_chunks(chunks: List[str]) -> List[List[float]]:
    """Generate embeddings for document chunks using Voyage AI."""

    if not chunks:
        return []

    response = client.embed(
        texts=chunks,
        model=EMBEDDING_MODEL,
        input_type="document"
    )

    return response.embeddings

