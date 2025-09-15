"""
Document loading and processing utilities for a learning assistant.
Handles reading PDFs and text files from a folder and splitting them into chunks.
"""

import os
import glob
import logging
from typing import List, Iterator
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, ReadTheDocsLoader
from langchain_core.documents import Document

from config import config

# Set up logging
logger = logging.getLogger(__name__)


class UTF8ReadTheDocsLoader(ReadTheDocsLoader):
    """Custom ReadTheDocs loader with UTF-8 encoding support."""
    
    def _get_file_content(self, file_path: str) -> str:
        """Read file content with UTF-8 encoding."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            logger.warning(f"Failed to read {file_path} with UTF-8, trying with errors='ignore'")
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    
    def lazy_load(self) -> Iterator[Document]:
        """Lazy load documents with chunking."""
        file_paths = glob.glob(os.path.join(self.file_path, "*.txt"))
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunking.chunk_size,
            chunk_overlap=config.chunking.chunk_overlap
        )
        
        for file_path in file_paths:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                
                if text.strip():
                    # Split the text into chunks
                    chunks = splitter.split_text(text)
                    for chunk in chunks:
                        yield Document(
                            page_content=chunk,
                            metadata={"source": file_path}
                        )
            except Exception as e:
                logger.error(f"Failed to process file {file_path}: {e}")
                continue


class DocumentProcessor:
    """Handles document loading and processing operations for mixed sources."""
    
    def __init__(self, source_directory: str):
        """Initialize with source directory path."""
        self.source_directory = Path(source_directory)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunking.chunk_size,
            chunk_overlap=config.chunking.chunk_overlap
        )
    
    def load_documents(self) -> List[Document]:
        """Load documents (.txt, .md, .pdf) from the source directory."""
        if not self.source_directory.exists():
            raise FileNotFoundError(f"Source directory not found: {self.source_directory}")

        logger.info(f"Loading documents from {self.source_directory}")

        docs: List[Document] = []
        # Text-like files
        for pattern in ("*.txt", "*.md"):
            for fp in self.source_directory.glob(pattern):
                try:
                    text = fp.read_text(encoding="utf-8", errors="ignore")
                    if text.strip():
                        docs.append(Document(page_content=text, metadata={"source": str(fp)}))
                except Exception as e:
                    logger.error(f"Failed to read {fp}: {e}")
        # PDFs
        for fp in self.source_directory.glob("*.pdf"):
            try:
                loader = PyPDFLoader(str(fp))
                pdf_docs = loader.load()
                for d in pdf_docs:
                    # ensure source path retained
                    d.metadata = {**d.metadata, "source": str(fp)}
                docs.extend(pdf_docs)
            except Exception as e:
                logger.error(f"Failed to load PDF {fp}: {e}")

        logger.info(f"Loaded {len(docs)} raw documents")
        return docs
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        logger.info("Splitting documents into chunks")
        
        chunked_docs = self.text_splitter.split_documents(documents)
        
        logger.info(f"Split into {len(chunked_docs)} chunks")
        return chunked_docs
    
    def process_metadata(self, documents: List[Document]) -> List[Document]:
        """Process and clean document metadata."""
        logger.info("Processing document metadata")
        
        for doc in documents:
            # Keep path or URL as-is; generic learning sources
            pass
        
        return documents
    
    def load_and_process(self) -> List[Document]:
        """Complete document loading and processing pipeline."""
        try:
            # Load raw documents
            raw_docs = self.load_documents()
            
            # Split into chunks
            chunked_docs = self.split_documents(raw_docs)
            
            # Process metadata
            processed_docs = self.process_metadata(chunked_docs)
            
            logger.info(f"Document processing completed. Total documents: {len(processed_docs)}")
            return processed_docs
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            raise
