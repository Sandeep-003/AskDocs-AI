"""
Streamlit web application for the Documentation Assistant.
Provides both simple and chat-based interfaces.
"""

import logging
from typing import Set, List, Dict, Any
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
import streamlit as st

from rag_system import RAGService
from tools import generate_code_example
from config import config
from web_scraper import WebScraper
from ingestion import DocumentIngestionPipeline

# Set up logging
logger = logging.getLogger(__name__)


class DocumentationApp:
    """Streamlit application for documentation Q&A."""
    
    def __init__(self):
        """Initialize the application."""
        self.setup_page_config()
        self.initialize_session_state()
        self.rag_service = self._get_rag_service()
    
    def setup_page_config(self) -> None:
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Documentation Assistant",
            page_icon="📚",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def initialize_session_state(self) -> None:
        """Initialize session state variables."""
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "namespace" not in st.session_state:
            st.session_state.namespace = "default"
    
    @st.cache_resource
    def _get_rag_service(_self) -> RAGService:
        """Get RAG service instance (cached)."""
        try:
            return RAGService(namespace=st.session_state.get("namespace"))
        except Exception as e:
            st.error(f"Failed to initialize RAG service: {e}")
            st.stop()
    
    def create_source_string(self, source_urls: Set[str]) -> str:
        """Create a formatted string of source URLs."""
        if not source_urls:
            return "**Sources:** No sources found."
        
        source_string = "**Sources:**\\n"
        for i, url in enumerate(sorted(source_urls), 1):
            source_string += f"{i}. {url}\\n"
        
        return source_string
    
    def process_query(self, query: str, use_history: bool = False) -> Dict[str, Any]:
        """Process a user query."""
        try:
            with st.spinner("Processing your query..."):
                if use_history and st.session_state.chat_history:
                    response = self.rag_service.ask(query, st.session_state.chat_history)
                else:
                    response = self.rag_service.ask(query)
                
                return response
        except Exception as e:
            st.error(f"Query processing failed: {e}")
            logger.error(f"Query processing error: {e}")
            return {}
    
    def render_simple_interface(self) -> None:
        """Render simple Q&A interface."""
        st.header("📚 Documentation Assistant - Simple Mode")
        st.markdown("Ask questions about LangChain documentation")
        
        # Input section
        with st.container():
            query = st.text_input(
                "Enter your question:",
                placeholder="How to use LangChain with Pinecone?",
                key="simple_query"
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                submit_button = st.button("Ask Question", type="primary")
        
        # Process query
        if submit_button and query:
            response = self.process_query(query, use_history=False)
            
            if response:
                # Display response
                st.subheader("Answer")
                st.write(response.get("answer", "No answer generated"))
                
                # Display sources
                sources = self.rag_service.get_sources(response)
                if sources:
                    with st.expander("📖 Sources", expanded=False):
                        for i, source in enumerate(sources, 1):
                            st.write(f"{i}. `{source}`")
        
        elif submit_button and not query:
            st.warning("Please enter a question before submitting.")
    
    def render_chat_interface(self) -> None:
        """Render chat-based interface with memory."""
        st.header("💬 Documentation Assistant - Chat Mode")
        st.markdown("Interactive chat with conversation memory")
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                # Show sources for assistant messages
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("📖 Sources", expanded=False):
                        for i, source in enumerate(message["sources"], 1):
                            st.write(f"{i}. `{source}`")
                if message["role"] == "assistant" and message.get("meta"):
                    with st.expander("ℹ️ Retrieval Meta", expanded=False):
                        meta = message["meta"]
                        st.write(f"Query type: {meta.get('query_type')}")
                        st.write(f"k used: {meta.get('k')}")
                        st.write(meta.get('context_summary',''))
        
        # Chat input
        if prompt := st.chat_input("Ask a question about the documentation or use /example <topic> ..."):
            # Slash command: /example
            if prompt.startswith('/example'):
                topic = prompt.replace('/example', '').strip() or 'rag'
                code = generate_code_example(topic)
                st.session_state.messages.append({"role": "assistant", "content": f"Code example for '{topic}':\n```python\n{code}\n```"})
                with st.chat_message("assistant"):
                    st.code(code, language='python')
            else:
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.write(prompt)
                # Generate response
                response = self.process_query(prompt, use_history=True)
                if response:
                    answer = response.get("answer", "No answer generated")
                    sources = self.rag_service.get_sources(response)
                    meta = response.get("meta", {})
                    # Add assistant message (single time)
                    assistant_msg = {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "meta": meta
                    }
                    st.session_state.messages.append(assistant_msg)
                    with st.chat_message("assistant"):
                        # Show fallback badge if used
                        if meta.get("fallback") == "model":
                            st.caption("Answered from model understanding (no/weak context)")
                        st.write(answer)
                        if sources:
                            with st.expander("📖 Sources", expanded=False):
                                for i, source in enumerate(sources, 1):
                                    st.write(f"{i}. `{source}`")
                        with st.expander("ℹ️ Retrieval Meta", expanded=False):
                            st.write(f"Query type: {meta.get('query_type')}")
                            st.write(f"k used: {meta.get('k')}")
                            st.write(meta.get('context_summary',''))
                    # Update chat history for RAG
                    st.session_state.chat_history.append({"role": "user", "content": prompt})
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})

    def render_sidebar(self) -> str:
        """Render sidebar with app controls."""
        with st.sidebar:
            st.title("🔧 Settings")
            mode = st.radio(
                "Select Mode:",
                ["Simple Q&A", "Chat with Memory"],
                help="Simple mode for single questions, Chat mode for conversations"
            )
            # Toggle for model fallback
            fallback_default = st.session_state.get("allow_model_fallback", True)
            allow_fb = st.checkbox("Allow model fallback when no context", value=fallback_default)
            st.session_state["allow_model_fallback"] = allow_fb
            try:
                # update underlying service instance if available
                if hasattr(self.rag_service, "rag_system"):
                    self.rag_service.rag_system.allow_model_fallback = bool(allow_fb)
            except Exception:
                pass
            st.subheader("📋 Configuration")
            st.info(f"""
            **LLM Model:** {config.llm.model_name}  
            **Embedding Model:** {config.embedding.model_name}  
            **Vector Index:** {config.vector_store.index_name}  
            **Chunk Size:** {config.chunking.chunk_size}
            """)
            # Namespace controls
            st.caption("Pinecone Namespace")
            # Cached list in session; allow manual refresh
            current_ns = st.session_state.get("namespace", "default")
            available_ns = st.session_state.get("ns_list")
            if available_ns is None:
                try:
                    available_ns = self.rag_service.vector_store_manager.list_namespaces()
                except Exception:
                    available_ns = []
                st.session_state["ns_list"] = available_ns
            # Refresh button
            col_r1, col_r2 = st.columns([3,1])
            with col_r2:
                if st.button("🔄 Refresh", help="Re-fetch namespaces from Pinecone"):
                    try:
                        available_ns = self.rag_service.vector_store_manager.list_namespaces()
                        st.session_state["ns_list"] = available_ns
                        st.success(f"Found {len(available_ns)} namespaces.")
                    except Exception as e:
                        st.warning(f"Could not refresh namespaces: {e}")
            if available_ns:
                ns = st.selectbox("Namespace", options=available_ns + ["<custom>"] , index=(available_ns.index(current_ns) if current_ns in available_ns else len(available_ns)), key="ns_select")
                if ns == "<custom>":
                    ns = st.text_input("Custom namespace", key="ns_input_custom", value=current_ns)
            else:
                ns = st.text_input("Namespace", key="ns_input", value=current_ns)
            col_ns1, col_ns2 = st.columns([1,1])
            with col_ns1:
                apply_ns = st.button("Apply Namespace")
            with col_ns2:
                reset_ns = st.button("Reset Namespace")
            if apply_ns:
                st.session_state.namespace = ns.strip() or "default"
                # Recreate service with new namespace
                st.cache_resource.clear()
                self.rag_service = self._get_rag_service()
                # Also refresh available namespaces list after switching
                try:
                    st.session_state["ns_list"] = self.rag_service.vector_store_manager.list_namespaces()
                except Exception:
                    st.session_state["ns_list"] = st.session_state.get("ns_list", [])
                st.success(f"Namespace set to '{st.session_state.namespace}'")
            if reset_ns:
                st.session_state.namespace = "default"
                st.cache_resource.clear()
                self.rag_service = self._get_rag_service()
                st.session_state["ns_list"] = None
                st.success("Namespace reset to 'default'")
            if mode == "Chat with Memory":
                if st.button("🗑️ Clear Chat History"):
                    st.session_state.chat_history = []
                    st.session_state.messages = []
                    st.success("Chat history cleared!")
                    st.rerun()
            if st.session_state.messages:
                user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
                st.metric("Total Questions", user_messages)

            st.markdown("\n---\n")
            st.subheader("🕷️ Crawl & Ingest")
            crawl_url = st.text_input("Seed URL", key="crawl_url", placeholder="https://example.com/docs/")
            col_c1, col_c2 = st.columns([1,1])
            with col_c1:
                max_pages = st.number_input("Max pages", min_value=1, max_value=200, value=10, step=1, key="crawl_max_pages")
            with col_c2:
                do_crawl = st.button("Crawl & Ingest", type="primary", key="crawl_btn")
            if do_crawl:
                if not crawl_url or not crawl_url.startswith(('http://', 'https://')):
                    st.error("Please enter a valid URL starting with http:// or https://")
                else:
                    with st.spinner("Scraping and ingesting... This may take a few minutes."):
                        try:
                            # Prepare output directory based on hostname and timestamp
                            host = urlparse(crawl_url).netloc.replace(':', '_')
                            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
                            out_dir = Path(__file__).parent / 'scraped_text' / f"user_{host}_{timestamp}"
                            scraper = WebScraper(str(out_dir))
                            scraper.scrape_website(crawl_url, max_pages=int(max_pages))
                            # Quick count of scraped txt files
                            txt_count = len(list(Path(out_dir).glob('*.txt')))
                            if txt_count == 0:
                                st.warning("No pages were saved. The site may block scraping or contained no parsable content.")
                            # Ingest into vector store
                            pipeline = DocumentIngestionPipeline(str(out_dir), namespace=st.session_state.get("namespace"))
                            pipeline.run_ingestion()
                            st.success(f"Crawled {txt_count} pages and ingested successfully into index '{config.vector_store.index_name}'.")
                            st.toast("Ingestion complete. New content is now searchable.")
                        except Exception as e:
                            st.error(f"Crawl/Ingest failed: {e}")
                            logger.error(f"Crawl/Ingest failed: {e}")
        return mode
    
    def run(self) -> None:
        """Run the Streamlit application."""
        try:
            # Render sidebar and get mode
            mode = self.render_sidebar()
            
            # Render appropriate interface
            if mode == "Simple Q&A":
                self.render_simple_interface()
            else:
                self.render_chat_interface()
                
        except Exception as e:
            st.error(f"Application error: {e}")
            logger.error(f"Application error: {e}")


def main():
    """Main function to run the app."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run app
    app = DocumentationApp()
    app.run()


if __name__ == "__main__":
    main()
