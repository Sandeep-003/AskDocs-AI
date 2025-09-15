"""Conversation memory management.
Provides short-term (recent turns) and long-term (summarized) memory utilities.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

logger = logging.getLogger(__name__)


class MemoryStore:
    """Manages chat memory with summarization support."""

    def __init__(self, summary_path: str = "long_term_memory.json", max_turns_before_summary: int = 6):
        self.summary_path = Path(summary_path)
        self.max_turns_before_summary = max_turns_before_summary
        self._long_term_summary: str = self._load_summary()

    # ------------------------ Persistence ------------------------
    def _load_summary(self) -> str:
        if self.summary_path.exists():
            try:
                data = json.loads(self.summary_path.read_text(encoding="utf-8"))
                return data.get("summary", "")
            except Exception:
                logger.warning("Failed to load existing long term summary; starting fresh")
        return ""

    def _save_summary(self) -> None:
        try:
            self.summary_path.write_text(json.dumps({"summary": self._long_term_summary}, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to persist summary: {e}")

    # ------------------------ Public API ------------------------
    @property
    def long_term_summary(self) -> str:
        return self._long_term_summary

    def maybe_update_summary(self, messages: List[Dict[str, Any]], summarizer_fn) -> None:
        """Summarize conversation every N user turns.
        summarizer_fn: callable(list[BaseMessage]) -> str
        """
        user_turns = sum(1 for m in messages if m.get("role") == "user")
        if user_turns == 0 or user_turns % self.max_turns_before_summary != 0:
            return
        # Build message objects
        lc_messages: List[BaseMessage] = []
        for m in messages[-self.max_turns_before_summary*2:]:  # last few exchanges
            role, content = m.get("role"), m.get("content", "")
            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
        try:
            new_summary = summarizer_fn(lc_messages, self._long_term_summary)
            if new_summary:
                self._long_term_summary = new_summary
                self._save_summary()
                logger.info("Long-term memory summary updated")
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")

    def build_context_prefix(self) -> str:
        if not self._long_term_summary:
            return ""
        return f"User background & prior context summary (for grounding future answers):\n{self._long_term_summary}\n---\n"
