"""
file: _chunker.py
brief: Job Market Data Ingestion (Chunking + Qdrant Upload)
"""
import os
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional, List

from loguru import logger
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models

# --- Configuration ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "job_market"
EMBEDDING_MODEL = "text-embedding-3-small"  # Efficient OpenAI model


@dataclass
class JobDocument:
    """Structured representation of a job posting/career document"""
    content: str
    metadata: dict[str, Any]
    filename: str
    local_path: str


class JobMarketChunker:
    """Splits job documents into semantic chunks."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "•", "- ", ". ", " ", ""]
        )

    def process_file(self, file_path: str, url: str = "") -> List[Document]:
        """Loads and chunks a single file."""
        path = Path(file_path)

        # 1. Load the raw text
        try:
            if path.suffix.lower() == '.pdf':
                loader = PyPDFLoader(str(path))
            else:
                loader = UnstructuredFileLoader(str(path))

            raw_docs = loader.load()
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return []

        final_chunks = []

        # 2. Chunk and enrich metadata
        for raw_doc in raw_docs:
            # Clean content slightly
            clean_content = raw_doc.page_content.replace("\x00", "")

            chunks = self.splitter.split_text(clean_content)

            for i, chunk_text in enumerate(chunks):
                meta = {
                    "source": str(path.name),
                    "chunk_index": i,
                    "url": url,
                    "path": str(path),
                    "type": "job_posting"
                }

                final_chunks.append(Document(
                    page_content=chunk_text,
                    metadata=meta
                ))

        logger.info(f"Processed {path.name}: {len(final_chunks)} chunks created.")
        return final_chunks


class QdrantIngestor:
    """Handles embedding and uploading chunks to Qdrant."""

    def __init__(self):
        self.client = QdrantClient(url=QDRANT_URL)
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self.vector_size = 1536  # Standard for text-embedding-3-small

    def create_collection_if_not_exists(self):
        """Ensures the Qdrant collection exists with correct config."""
        if not self.client.collection_exists(COLLECTION_NAME):
            logger.info(f"Creating Qdrant collection: {COLLECTION_NAME}")
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                )
            )
        else:
            logger.info(f"Collection {COLLECTION_NAME} already exists.")

    def ingest(self, documents: List[Document]):
        """Embeds documents and uploads them to Qdrant in batches."""
        if not documents:
            logger.warning("No documents to ingest.")
            return

        self.create_collection_if_not_exists()

        logger.info(f"Embedding {len(documents)} documents...")

        # 1. Generate Embeddings
        texts = [doc.page_content for doc in documents]
        vectors = self.embeddings.embed_documents(texts)

        # 2. Prepare Points for Qdrant
        points = []
        for i, (doc, vector) in enumerate(zip(documents, vectors)):
            payload = doc.metadata
            payload["page_content"] = doc.page_content  # Store text in payload for retrieval

            points.append(models.PointStruct(
                id=str(uuid.uuid4()),  # Generate unique ID
                vector=vector,
                payload=payload
            ))

        # 3. Upload in batches
        logger.info("Uploading to Qdrant...")
        self.client.upload_points(
            collection_name=COLLECTION_NAME,
            points=points,
            batch_size=100  # Adjust based on network/memory
        )

        logger.success(f"Successfully ingested {len(points)} chunks into '{COLLECTION_NAME}'.")


# --- Helper to run ingestion easily ---
def run_ingestion(folder_path: str):
    """
    Utility function to ingest all files in a folder.
    """
    chunker = JobMarketChunker()
    ingestor = QdrantIngestor()

    all_chunks = []

    # 1. Scan folder
    files = [f for f in os.listdir(folder_path) if f.endswith(('.pdf', '.txt', '.docx'))]
    logger.info(f"Found {len(files)} files in {folder_path}")

    # 2. Chunk files
    for f in files:
        full_path = os.path.join(folder_path, f)
        chunks = chunker.process_file(full_path)
        all_chunks.extend(chunks)

    # 3. Upload
    if all_chunks:
        ingestor.ingest(all_chunks)
    else:
        logger.warning("No valid content found to ingest.")