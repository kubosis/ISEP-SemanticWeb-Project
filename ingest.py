"""
file: ingest.py
brief: One-off script to load job market data into Qdrant
"""
import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from chunker import run_ingestion
from loguru import logger

load_dotenv()


def main():
    DATA_FOLDER = "data_jobs"

    if not os.path.exists(DATA_FOLDER):
        logger.error(f"Folder '{DATA_FOLDER}' not found. Please create it and add documents.")
        return

    logger.info(f"Starting ingestion from folder: {DATA_FOLDER}")
    try:
        run_ingestion(DATA_FOLDER)
        logger.success("Ingestion complete! Data is now in Qdrant.")
    except Exception as e:
        logger.exception(f"Ingestion failed: {e}")


if __name__ == "__main__":
    main()