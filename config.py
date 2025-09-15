"""
Configuration module for the Documentation Assistant.
Manages environment variables and application settings.
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    model_name: str = "text-embedding-005"
    project_id: Optional[str] = None
    
    def __post_init__(self):
        if self.project_id is None:
            self.project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")


@dataclass
class LLMConfig:
    """Configuration for Language Learning Models."""
    model_name: str = "gemini-2.0-flash"
    temperature: float = 0.2
    max_output_tokens: int = 4096
    top_p: float = 0.8
    top_k: int = 50


@dataclass
class VectorStoreConfig:
    """Configuration for vector store."""
    index_name: str = "langchain-docs-index"
    batch_size: int = 100
    
    def __post_init__(self):
        # Override with environment variable if available
        env_index = os.environ.get("INDEX_NAME")
        if env_index:
            self.index_name = env_index


@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    chunk_size: int = 600
    chunk_overlap: int = 50


@dataclass
class ScrapingConfig:
    """Configuration for web scraping."""
    timeout: int = 10
    max_retries: int = 3
    delay_between_requests: float = 1.0


@dataclass
class AppConfig:
    """Main application configuration."""
    embedding: EmbeddingConfig
    llm: LLMConfig
    vector_store: VectorStoreConfig
    chunking: ChunkingConfig
    scraping: ScrapingConfig
    
    @classmethod
    def from_defaults(cls):
        """Create configuration with default values."""
        return cls(
            embedding=EmbeddingConfig(),
            llm=LLMConfig(),
            vector_store=VectorStoreConfig(),
            chunking=ChunkingConfig(),
            scraping=ScrapingConfig()
        )


def validate_environment():
    """Validate required environment variables."""
    required_vars = [
        "GOOGLE_CLOUD_PROJECT",
        "PINECONE_API_KEY",
        "INDEX_NAME"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    # Validate Google Application Credentials
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        print("Warning: GOOGLE_APPLICATION_CREDENTIALS not set. Using default credentials.")


# Global configuration instance
config = AppConfig.from_defaults()
