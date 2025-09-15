"""
RAG (Retrieval-Augmented Generation) system implementation.
Handles query processing and response generation.
"""

import logging
from typing import List, Dict, Any, Optional, Callable

from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from langchain_google_vertexai import VertexAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from config import config
from vector_store import VectorStoreManager
from query_classifier import QueryClassifier
from memory_store import MemoryStore
from tools import summarize_sources

# Set up logging
logger = logging.getLogger(__name__)


class RAGSystem:
    """Retrieval-Augmented Generation system for document Q&A."""

    def __init__(self, vector_store_manager: VectorStoreManager, memory_store: Optional[MemoryStore] = None):
        """Initialize RAG system with vector store manager."""
        self.vector_store_manager = vector_store_manager
        self.llm = self._create_llm()
        self._retrieval_chain = None
        self._history_aware_chain = None
        # Provide the same LLM to the classifier for better results
        self.classifier = QueryClassifier(llm=self.llm)
        self.memory_store = memory_store or MemoryStore()
        self._summary_fn = self._llm_summary_fn
        # Controls whether to answer from the model when retrieval finds no/weak context
        self.allow_model_fallback = True
    
    def _create_llm(self) -> VertexAI:
        """Create LLM instance."""
        try:
            llm = VertexAI(
                model_name=config.llm.model_name,
                temperature=config.llm.temperature,
                max_output_tokens=config.llm.max_output_tokens,
                top_p=config.llm.top_p,
                top_k=config.llm.top_k,
            )
            logger.info(f"Created LLM with model: {config.llm.model_name}")
            return llm
        except Exception as e:
            logger.error(f"Failed to create LLM: {e}")
            raise
    
    def _get_retrieval_chain(self, k: int = 5):
        """Get or create retrieval chain (recreates if k differs)."""
        recreate = False
        if self._retrieval_chain is None:
            recreate = True
        else:
            # If current retriever k differs, rebuild
            try:
                current_k = self._retrieval_chain.input_schema.schema().get('properties', {}).get('context', {})
            except Exception:
                current_k = None
            if current_k != k:
                recreate = True
        if recreate:
            try:
                retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
                stuff_documents_chain = create_stuff_documents_chain(
                    llm=self.llm,
                    prompt=retrieval_qa_chat_prompt
                )
                retriever = self.vector_store_manager.get_retriever({"k": k})
                self._retrieval_chain = create_retrieval_chain(
                    retriever=retriever,
                    combine_docs_chain=stuff_documents_chain
                )
                logger.info(f"Retrieval chain (k={k}) created successfully")
            except Exception as e:
                logger.error(f"Failed to create retrieval chain: {e}")
                raise
        return self._retrieval_chain

    def _should_fallback_to_model(self, answer: str, docs: List[Document]) -> bool:
        """Decide if we should answer from general model knowledge.

        Triggers when no docs are returned or the answer appears to refuse due to missing context.
        """
        if not docs:
            return True
        if not answer:
            return True
        text = answer.lower()
        refusal_markers = [
            "i don't know",
            "i do not know",
            "not in the provided context",
            "cannot find",
            "no relevant context",
            "insufficient information",
            "not enough context",
            "i don't have access",
        ]
        return any(m in text for m in refusal_markers)

    def _model_only_answer(self, question: str, history: Optional[List[BaseMessage]] = None) -> str:
        """Generate an answer using only the model's general knowledge (no retrieval).

        Produces a medium-length response with brief details and one small example when helpful.
        """
        try:
            history_hint = ""
            if history:
                last_user = next((m.content for m in reversed(history) if isinstance(m, HumanMessage)), "")
                if last_user:
                    history_hint = f"Last user message: {last_user}\n"
            prompt = (
                "You are a helpful technical assistant. Answer based on your general knowledge.\n"
                "Provide a medium-length answer (about 6–10 sentences).\n"
                "Be concrete and add a brief example or steps if useful.\n"
                "If there are important caveats, mention them concisely.\n\n"
                f"{history_hint}Question: {question}\n"
                "Answer:"
            )
            out = self.llm.invoke(prompt)
            return out.strip() if isinstance(out, str) else str(out).strip()
        except Exception as e:
            logger.warning(f"Model-only answer failed, returning minimal string: {e}")
            return "Here's what I understand: " + question

    def _expand_answer(self, answer: str, question: str, context_summary: str = "") -> str:
        """Expand a short answer to a medium-length response while preserving facts.

        Only expands when the original answer is quite short.
        """
        try:
            if not answer or len(answer) < 300:
                hint = f"Context summary: {context_summary}\n" if context_summary else ""
                prompt = (
                    "Rewrite the answer below to a medium-length response (6–10 sentences).\n"
                    "Preserve factual claims, add one brief example or steps if helpful, and keep it concise.\n\n"
                    f"Question: {question}\n"
                    f"{hint}Original answer: {answer}\n\n"
                    "Rewritten answer:"
                )
                out = self.llm.invoke(prompt)
                return out.strip() if isinstance(out, str) else str(out).strip()
            return answer
        except Exception as e:
            logger.warning(f"Answer expansion failed, using original: {e}")
            return answer
    
    def _get_history_aware_chain(self):
        """Get or create history-aware retrieval chain."""
        if self._history_aware_chain is None:
            try:
                # Get prompts
                retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
                rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
                
                # Create document chain
                stuff_documents_chain = create_stuff_documents_chain(
                    llm=self.llm,
                    prompt=retrieval_qa_chat_prompt
                )
                
                # Create history-aware retriever
                history_aware_retriever = create_history_aware_retriever(
                    llm=self.llm,
                    retriever=self.vector_store_manager.get_retriever(),
                    prompt=rephrase_prompt,
                )
                
                # Create retrieval chain with history
                self._history_aware_chain = create_retrieval_chain(
                    retriever=history_aware_retriever,
                    combine_docs_chain=stuff_documents_chain
                )
                
                logger.info("History-aware retrieval chain created successfully")
                
            except Exception as e:
                logger.error(f"Failed to create history-aware chain: {e}")
                raise
        
        return self._history_aware_chain
    
    def _fallback_light_summary(self, messages: List[BaseMessage], existing_summary: str) -> str:
        """Simple heuristic summary used only if LLM call fails."""
        last_user = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
        truncated = (last_user[:120] + '...') if len(last_user) > 120 else last_user
        if existing_summary:
            return existing_summary + f" | Recent focus: {truncated}"
        return f"User intent: {truncated}"

    def _llm_summary_fn(self, messages: List[BaseMessage], existing_summary: str) -> str:
        """Generate/merge a long-term conversation summary using the LLM.

        Keeps stable user goals & preferences; <= 80 words target.
        Falls back to lightweight heuristic if the LLM call fails.
        """
        try:
            # Build recent dialogue transcript (limit to last ~12 messages for brevity)
            recent = messages[-12:]
            lines = []
            for m in recent:
                role = 'User' if isinstance(m, HumanMessage) else 'Assistant'
                content = m.content.replace('\n', ' ').strip()
                if len(content) > 180:
                    content = content[:177] + '...'
                lines.append(f"{role}: {content}")
            transcript = "\n".join(lines)
            prompt = (
                "You are a system that maintains a persistent, concise summary of a user's ongoing goals, "
                "preferences, and technical focus for a documentation Q&A assistant.\n\n"
                f"EXISTING SUMMARY (may be empty):\n{existing_summary or 'NONE'}\n\n"
                f"RECENT DIALOGUE (most recent last):\n{transcript}\n\n"
                "Update and merge into a single refined summary (<= 80 words) focusing ONLY on stable intent, "
                "goals, problem areas, technology stack, and recurring themes. Do NOT include transient single questions. "
                "Return ONLY the summary text with no preface or labels."
            )
            new_summary = self.llm.invoke(prompt)
            if isinstance(new_summary, str):
                cleaned = new_summary.strip()
            else:
                cleaned = str(new_summary).strip()
            # Guard against empty
            if not cleaned:
                return self._fallback_light_summary(messages, existing_summary)
            return cleaned
        except Exception as e:
            logger.warning(f"LLM summary generation failed, using fallback: {e}")
            return self._fallback_light_summary(messages, existing_summary)

    def query(self, question: str) -> Dict[str, Any]:
        """Process a query without chat history (dynamic k)."""
        try:
            logger.info(f"Processing query: {question[:100]}...")
            classification = self.classifier.classify(question)
            chain = self._get_retrieval_chain(k=classification.suggested_k)
            # Optionally rephrase ambiguous queries
            rephrased = self.classifier.rephrase(question, classification)
            augmented_question = rephrased or question
            prefix = self.memory_store.build_context_prefix()
            if prefix:
                augmented_question = prefix + augmented_question
            response = chain.invoke({"input": augmented_question})
            docs = response["context"]
            # Provide quick summarized context snippet tool output
            context_summary = summarize_sources(docs)
            final_answer = response["answer"]
            if self.allow_model_fallback and self._should_fallback_to_model(final_answer, docs):
                logger.info("Falling back to model-only answer (no or weak context).")
                final_answer = self._model_only_answer(question)
                docs = []
            # Expand to medium-length when too short
            final_answer = self._expand_answer(final_answer, question, context_summary)
            result = {
                "query": response["input"],
                "answer": final_answer,
                "context": docs,
                "source_documents": docs,
                "meta": {
                    "query_type": classification.query_type,
                    "k": classification.suggested_k,
                    "context_summary": context_summary,
                    "fallback": "model" if not docs else "none",
                }
            }
            logger.info("Query processed successfully")
            return result
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            raise
    
    def _coerce_chat_history(self, history: List[Dict[str, Any]]) -> List[BaseMessage]:
        """Coerce various stored history formats into LangChain BaseMessage list.

        Supports two formats:
        1. New format (recommended): {"role": "user"|"assistant", "content": str}
        2. Legacy batch format: {"user": str, "assistant": str}
        """
        coerced: List[BaseMessage] = []
        for item in history:
            if not isinstance(item, dict):
                continue
            # Preferred format
            if "role" in item and "content" in item:
                role = item.get("role")
                content = item.get("content", "")
                if role == "user":
                    coerced.append(HumanMessage(content=content))
                elif role == "assistant":
                    coerced.append(AIMessage(content=content))
                else:
                    logger.warning(f"Unknown role in chat history item: {role}")
            # Legacy combined exchange format
            elif "user" in item and "assistant" in item:
                user_q = item.get("user", "")
                assistant_a = item.get("assistant", "")
                if user_q:
                    coerced.append(HumanMessage(content=user_q))
                if assistant_a:
                    coerced.append(AIMessage(content=assistant_a))
            else:
                logger.warning(f"Unrecognized chat history structure: {item}")
        return coerced

    def query_with_history(self, question: str, chat_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a query with chat history context.

        Ensures chat history matches LangChain expected format (list[BaseMessage]).
        """
        try:
            logger.info(f"Processing query with history: {question[:100]}...")
            
            chain = self._get_history_aware_chain()
            classification = self.classifier.classify(question)
            coerced_history = self._coerce_chat_history(chat_history)
            prefix = self.memory_store.build_context_prefix()
            # Optionally rephrase
            rephrased = self.classifier.rephrase(question, classification)
            augmented_question = rephrased or question
            if prefix:
                augmented_question = prefix + augmented_question
            chain = self._get_history_aware_chain()
            response = chain.invoke({
                "input": augmented_question,
                "chat_history": coerced_history
            })
            docs = response["context"]
            context_summary = summarize_sources(docs)
            final_answer = response["answer"]
            if self.allow_model_fallback and self._should_fallback_to_model(final_answer, docs):
                logger.info("Falling back to model-only answer (no or weak context) [history mode].")
                final_answer = self._model_only_answer(question, history=coerced_history)
                docs = []
            # Expand to medium-length when too short
            final_answer = self._expand_answer(final_answer, question, context_summary)
            result = {
                "query": response["input"],
                "answer": final_answer,
                "context": docs,
                "source_documents": docs,
                "meta": {
                    "query_type": classification.query_type,
                    "k": classification.suggested_k,
                    "context_summary": context_summary,
                    "fallback": "model" if not docs else "none",
                }
            }
            logger.info("Query with history processed successfully")
            # Update long-term summary periodically
            try:
                self.memory_store.maybe_update_summary(chat_history, self._summary_fn)
            except Exception as me:
                logger.warning(f"Memory summarization skipped: {me}")
            return result
            
        except Exception as e:
            logger.error(f"Query with history processing failed: {e}")
            raise


class RAGService:
    """High-level service for RAG operations."""
    
    def __init__(self, index_name: Optional[str] = None, namespace: Optional[str] = None):
        """Initialize RAG service."""
        self.vector_store_manager = VectorStoreManager()
        self.vector_store_manager.connect_to_index(index_name, namespace=namespace)
        self.rag_system = RAGSystem(self.vector_store_manager)
    
    def ask(self, question: str, chat_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Ask a question with optional chat history."""
        if chat_history:
            return self.rag_system.query_with_history(question, chat_history)
        else:
            return self.rag_system.query(question)
    
    def get_sources(self, response: Dict[str, Any]) -> List[str]:
        """Extract source URLs from response."""
        sources = set()
        context_docs = response.get("context", [])
        
        for doc in context_docs:
            if isinstance(doc, Document):
                source = doc.metadata.get("source", "")
                if source:
                    sources.add(source)
        
        return list(sources)
