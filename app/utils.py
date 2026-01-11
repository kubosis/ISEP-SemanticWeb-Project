import os
import sys
from typing import Final

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

try:
    OPENAI_API_KEY: Final = os.getenv("OPENAI_API_KEY")
    OPENAI_DEPLOYMENT: Final = os.getenv("OPENAI_DEPLOYMENT", "gpt-5.1")
    OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
    NEO4J_AUTH = (os.getenv("NEO4J_USER", "neo4j"),
            os.getenv("NEO4J_PASSWORD", "password"))
except:
    logger.error("Unable to load required environmental variables, aborting...")
    sys.exit(1)
