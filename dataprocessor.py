from pdfreader import read_pdf
from chunker import chunk_pages
from embedder import embed_chunks
from vectorstore import store_in_pinecone
from typing import List

pdf_path = "./resorces/HRPolicy.pdf"

def run():
    # Read HR Policy PDF and extrach text
    pages = read_pdf(pdf_path)
    # Chunk the data into smaller pieces
    chunks = chunk_pages(pages, chunk_size=900 , chunk_overlap=150)
    # print(chunks)
    embedded_chunks = embed_chunks(chunks)
    # Store the chunk and their embeddings in Pinecone vector database
    store_in_pinecone(
        chunks,
        embedded_chunks,
        namespace="hr-policy"
    )
if __name__ == "__main__":
    run()  