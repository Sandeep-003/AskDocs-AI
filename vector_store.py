"""
Vector store management for document embeddings.
Handles Pinecone vector database operations.
"""

import logging
import os
from typing import List, Optional

from langchain_core.documents import Document
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from config import config

# Set up logging
logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages vector store operations for document embeddings."""
    
    def __init__(self):
        """Initialize the vector store manager."""
        self.embeddings = self._create_embeddings()
        self.vector_store: Optional[PineconeVectorStore] = None
        self.index_name: Optional[str] = None
        self.namespace: Optional[str] = None
    
    def _create_embeddings(self) -> VertexAIEmbeddings:
        """Create embedding model instance."""
        try:
            embeddings = VertexAIEmbeddings(
                model_name=config.embedding.model_name,
                project=config.embedding.project_id
            )
            logger.info(f"Created embeddings with model: {config.embedding.model_name}")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}")
            raise
    
    def connect_to_index(self, index_name: Optional[str] = None, namespace: Optional[str] = None) -> PineconeVectorStore:
        """Connect to existing Pinecone index with optional namespace."""
        self.index_name = index_name or config.vector_store.index_name
        self.namespace = namespace
        try:
            self.vector_store = PineconeVectorStore(
                index_name=self.index_name,
                embedding=self.embeddings,
                namespace=self.namespace
            )
            ns_log = self.namespace if self.namespace else "<default>"
            logger.info(f"Connected to Pinecone index: {self.index_name} (namespace={ns_log})")
            return self.vector_store
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone index {self.index_name}: {e}")
            raise
    
    def batch_upsert(self, documents: List[Document], index_name: Optional[str] = None, namespace: Optional[str] = None) -> None:
        """Upload documents to vector store in batches."""
        index_name = index_name or self.index_name or config.vector_store.index_name
        namespace = namespace if namespace is not None else self.namespace
        batch_size = config.vector_store.batch_size
        
        if not documents:
            logger.warning("No documents to upsert")
            return
        
        logger.info(f"Starting batch upsert of {len(documents)} documents to index: {index_name}")
        
        try:
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(documents) + batch_size - 1) // batch_size
                
                logger.info(f"Upserting batch {batch_num}/{total_batches} ({len(batch)} docs)")
                
                # Create vector store from documents
                PineconeVectorStore.from_documents(
                    batch,
                    self.embeddings,
                    index_name=index_name,
                    namespace=namespace,
                )
            
            logger.info("Batch upsert completed successfully")
            
        except Exception as e:
            logger.error(f"Batch upsert failed: {e}")
            raise
    
    def create_from_documents(self, documents: List[Document], index_name: Optional[str] = None, namespace: Optional[str] = None) -> PineconeVectorStore:
        """Create a new vector store from documents."""
        index_name = index_name or self.index_name or config.vector_store.index_name
        namespace = namespace if namespace is not None else self.namespace
        
        try:
            logger.info(f"Creating vector store from {len(documents)} documents")
            
            self.vector_store = PineconeVectorStore.from_documents(
                documents,
                self.embeddings,
                index_name=index_name,
                namespace=namespace,
            )
            
            logger.info(f"Vector store created successfully with index: {index_name}")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            raise
    
    def get_retriever(self, search_kwargs: Optional[dict] = None, namespace: Optional[str] = None):
        """Get a retriever from the vector store."""
        if namespace is not None and (namespace != self.namespace or self.vector_store is None):
            # Create a transient store bound to this namespace
            temp_store = PineconeVectorStore(
                index_name=self.index_name or config.vector_store.index_name,
                embedding=self.embeddings,
                namespace=namespace
            )
            return temp_store.as_retriever(search_kwargs=search_kwargs or {"k": 5})
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call connect_to_index() first.")
        search_kwargs = search_kwargs or {"k": 5}
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)
    
    def similarity_search(self, query: str, k: int = 5, namespace: Optional[str] = None) -> List[Document]:
        """Perform similarity search."""
        try:
            if namespace is not None and (namespace != self.namespace or self.vector_store is None):
                temp_store = PineconeVectorStore(
                    index_name=self.index_name or config.vector_store.index_name,
                    embedding=self.embeddings,
                    namespace=namespace
                )
                results = temp_store.similarity_search(query, k=k)
            else:
                if not self.vector_store:
                    raise ValueError("Vector store not initialized. Call connect_to_index() first.")
                results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Similarity search returned {len(results)} documents")
            return results
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise

    def list_namespaces(self) -> List[str]:
        """Return a list of namespaces present in the configured Pinecone index.

        Uses the Pinecone client to describe index stats and extract namespace names.
        Returns an empty list on failure.
        """
        try:
            from pinecone import Pinecone  # lazy import to avoid hard dependency at module import
        except Exception as ie:
            logger.warning(f"Pinecone SDK not available to list namespaces: {ie}")
            return []

        try:
            api_key = os.getenv("PINECONE_API_KEY")
            if not api_key:
                logger.warning("PINECONE_API_KEY not set; cannot list namespaces.")
                return []
            pc = Pinecone(api_key=api_key)
            index_name = self.index_name or config.vector_store.index_name
            host = os.getenv("PINECONE_HOST")
            index = pc.Index(index_name, host=host) if host else pc.Index(index_name)
            stats = index.describe_index_stats()
            ns_map = stats.get("namespaces") or {}
            names = sorted([n if n else "default" for n in ns_map.keys()])
            logger.info(f"Discovered namespaces: {names}")
            return names
        except Exception as e:
            logger.warning(f"Failed to list namespaces: {e}")
            return []
