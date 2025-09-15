"""
Document ingestion pipeline for the learning assistant.
Loads local resources (txt, md, pdf), chunks, and upserts to Pinecone.
"""

import logging
from pathlib import Path
from typing import Optional

from config import config, validate_environment
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager

# Set up logging
logger = logging.getLogger(__name__)


class DocumentIngestionPipeline:
    """Complete pipeline for document ingestion into vector store."""
    
    def __init__(self, source_directory: str, index_name: Optional[str] = None, namespace: Optional[str] = None):
        """Initialize ingestion pipeline."""
        self.source_directory = Path(source_directory)
        self.index_name = index_name or config.vector_store.index_name
        self.namespace = namespace
        self.document_processor = DocumentProcessor(str(self.source_directory))
        self.vector_store_manager = VectorStoreManager()
        # Bind connection with optional namespace
        try:
            self.vector_store_manager.connect_to_index(self.index_name, namespace=self.namespace)
        except Exception:
            # Fallback to default connect to preserve existing behavior
            self.vector_store_manager.connect_to_index(self.index_name)
    
    def validate_setup(self) -> None:
        """Validate environment and setup."""
        logger.info("Validating environment setup...")
        
        # Validate environment variables
        validate_environment()
        
        # Check source directory
        if not self.source_directory.exists():
            raise FileNotFoundError(f"Source directory not found: {self.source_directory}")
        
        # Check for supported files
        supported = []
        for pattern in ("*.txt", "*.md", "*.pdf"):
            supported.extend(self.source_directory.glob(pattern))
        if not supported:
            raise ValueError(f"No supported files (.txt, .md, .pdf) found in {self.source_directory}")
        logger.info(f"Found {len(supported)} files to process")
        logger.info("Environment validation completed")
    
    def run_ingestion(self) -> None:
        """Run the complete ingestion pipeline."""
        try:
            logger.info("Starting document ingestion pipeline...")
            
            # Validate setup
            self.validate_setup()
            
            # Process documents
            logger.info("Processing documents...")
            documents = self.document_processor.load_and_process()
            
            if not documents:
                raise ValueError("No documents were processed")
            
            # Ingest into vector store
            logger.info(f"Ingesting {len(documents)} documents into vector store...")
            self.vector_store_manager.batch_upsert(documents, self.index_name, namespace=self.namespace)
            
            logger.info("Document ingestion completed successfully!")
            logger.info(f"Total documents ingested: {len(documents)}")
            logger.info(f"Vector store index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Document ingestion failed: {e}")
            raise


def ingest_resources(source_dir: str = "scraped_text", index_name: Optional[str] = None, namespace: Optional[str] = None) -> None:
    """Convenience function to ingest generic resources from a folder."""
    try:
        pipeline = DocumentIngestionPipeline(source_dir, index_name, namespace=namespace)
        pipeline.run_ingestion()
    except Exception as e:
        logger.error(f"Resource ingestion failed: {e}")
        raise


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run ingestion
    ingest_resources()
