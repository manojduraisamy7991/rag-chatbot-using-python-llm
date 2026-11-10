
from pinecone import Pinecone, SchemaBuilder
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(
    api_key=os.getenv("VECTOR_PINE_API_key")
)

# Define the searchable fields and Voyage embedding field
schema = (
    SchemaBuilder()
    .add_string_field(
        "title",
        full_text_search={"language": "en"}
    )
    .add_string_field(
        "chunk_text",
        full_text_search={"language": "en"}
    )
    .add_dense_vector_field(
        "chunk_embedding",
        dimension=1024,
        metric="cosine"
    )
    .build()
)

# Create the index if it does not already exist
index_name = "belk-python-rag-voyage"

if not pc.indexes.exists(index_name):
    pc.indexes.create(
        name=index_name,
        schema=schema,
        deployment={
            "deployment_type": "managed",
            "cloud": "aws",
            "region": "us-east-1"
        }
    )

    print("Pinecone index created successfully!")
else:
    print("Pinecone index already exists!")
