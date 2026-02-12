# api/chatbot.py
"""
Main chatbot API for legal research.
Provides a unified interface for processing queries with different modes.
"""

import os
from typing import Dict, Any, List, Optional, Literal
from dataclasses import dataclass, field
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Import tools and agents
from tools.agent import LegalResearchAgent


@dataclass
class ChatResponse:
    """Structured response from the chatbot."""
    answer: str
    raw_results: str = ""
    tools_used: List[str] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    query_time: float = 0.0
    mode: str = "single"  # "single" or "pipeline"
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.error is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "raw_results": self.raw_results,
            "tools_used": self.tools_used,
            "sources": self.sources,
            "query_time": self.query_time,
            "mode": self.mode,
            "error": self.error,
            "is_success": self.is_success,
            "metadata": self.metadata
        }


class LegalChatbot:
    """
    Main chatbot class that provides a unified interface for legal research.

    Supports three modes:
    - single: Uses GPT-4o for both retrieval and response
    - pipeline: Uses GPT-4o for retrieval + Gemini for detailed response
    - web_search: Uses Gemini 2.5 Pro with Google Search grounding for live web results
    """

    def __init__(
        self,
        mode: Literal["single", "pipeline", "web_search"] = "single",
        retrieval_model: str = "gpt-4o",
        response_model: str = "gemini-2.5-flash",
        verbose: bool = False
    ):
        """
        Initialize the chatbot.

        Args:
            mode: "single" for GPT-only, "pipeline" for GPT+Gemini, "web_search" for Gemini+Google Search
            retrieval_model: Model for retrieval/tool calling
            response_model: Model for response generation (pipeline mode only)
            verbose: Print debug information
        """
        self.mode = mode
        self.retrieval_model = retrieval_model
        self.response_model = response_model
        self.verbose = verbose

        # Initialize agents lazily
        self._retrieval_agent = None
        self._pipeline = None
        self._web_search_agent = None

        # Track if Gemini is available
        self._gemini_available = self._check_gemini_available()

    def _check_gemini_available(self) -> bool:
        """Check if Gemini API key is available."""
        return bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))

    def _get_retrieval_agent(self) -> LegalResearchAgent:
        """Lazy initialization of retrieval agent."""
        if self._retrieval_agent is None:
            self._retrieval_agent = LegalResearchAgent(
                model=self.retrieval_model,
                temperature=0,
                verbose=self.verbose
            )
        return self._retrieval_agent

    def _get_web_search_agent(self):
        """Lazy initialization of web search agent."""
        if self._web_search_agent is None:
            if not self._gemini_available:
                raise ValueError(
                    "Web search mode requires GOOGLE_API_KEY. "
                    "Please set it in your .env file."
                )
            from tools.web_search_agent import WebSearchAgent
            self._web_search_agent = WebSearchAgent(
                model="gemini-2.5-flash",
                temperature=0.3,
                verbose=self.verbose
            )
        return self._web_search_agent

    def _get_pipeline(self):
        """Lazy initialization of pipeline."""
        if self._pipeline is None:
            if not self._gemini_available:
                raise ValueError(
                    "Pipeline mode requires GOOGLE_API_KEY. "
                    "Please set it in your .env file or use single mode."
                )
            from tools.response_agent import LegalResearchPipeline
            self._pipeline = LegalResearchPipeline(
                retrieval_model=self.retrieval_model,
                response_model=self.response_model,
                verbose=self.verbose
            )
        return self._pipeline

    def chat(
        self,
        query: str,
        session_history: List[Dict[str, str]] = None,
        use_pipeline: bool = None,
        use_web_search: bool = False
    ) -> ChatResponse:
        """
        Process a user query and return a response.

        Args:
            query: User's question
            session_history: Previous conversation history
            use_pipeline: Override default mode (True=pipeline, False=single)
            use_web_search: If True, use web search mode (overrides other modes)

        Returns:
            ChatResponse with answer and metadata
        """
        import time
        start_time = time.time()

        # Web search mode takes priority
        if use_web_search or self.mode == "web_search":
            if not self._gemini_available:
                return ChatResponse(
                    answer="Web search requires GOOGLE_API_KEY. Please set it in your .env file.",
                    error="GOOGLE_API_KEY not set",
                    query_time=time.time() - start_time,
                    mode="web_search"
                )
            try:
                return self._chat_web_search(query, session_history or [], start_time)
            except Exception as e:
                return ChatResponse(
                    answer="Web search encountered an error. Please try again.",
                    error=str(e),
                    query_time=time.time() - start_time,
                    mode="web_search"
                )

        # Determine mode
        use_pipeline_mode = use_pipeline if use_pipeline is not None else (self.mode == "pipeline")

        # If pipeline requested but Gemini not available, fall back to single
        if use_pipeline_mode and not self._gemini_available:
            use_pipeline_mode = False
            if self.verbose:
                print("Gemini not available, falling back to single mode")

        try:
            if use_pipeline_mode:
                return self._chat_pipeline(query, start_time)
            else:
                return self._chat_single(query, session_history, start_time)

        except Exception as e:
            return ChatResponse(
                answer="I encountered an error while processing your query. Please try again.",
                error=str(e),
                query_time=time.time() - start_time,
                mode="pipeline" if use_pipeline_mode else "single"
            )

    def _chat_single(
        self,
        query: str,
        session_history: List[Dict[str, str]],
        start_time: float
    ) -> ChatResponse:
        """Process query using single agent mode."""
        import time

        agent = self._get_retrieval_agent()

        # Add session history context if available
        if session_history:
            # Inject history into agent
            agent.chat_history = session_history.copy()

        result = agent.query(query)

        # Extract sources from raw results
        sources = self._extract_sources(result.get("answer", ""))

        return ChatResponse(
            answer=result.get("answer", ""),
            raw_results=result.get("answer", ""),
            tools_used=result.get("tools_used", []),
            sources=sources,
            query_time=time.time() - start_time,
            mode="single"
        )

    def _chat_pipeline(self, query: str, start_time: float) -> ChatResponse:
        """Process query using pipeline mode."""
        import time

        pipeline = self._get_pipeline()
        result = pipeline.query(query)

        # Extract sources from raw results
        sources = self._extract_sources(result.get("raw_results", ""))

        return ChatResponse(
            answer=result.get("detailed_answer", ""),
            raw_results=result.get("raw_results", ""),
            tools_used=result.get("tools_used", []),
            sources=sources,
            query_time=time.time() - start_time,
            mode="pipeline"
        )

    def _chat_web_search(
        self,
        query: str,
        session_history: List[Dict[str, str]],
        start_time: float
    ) -> ChatResponse:
        """Process query using web search agent with PDF-filtering tools."""
        import time

        agent = self._get_web_search_agent()

        # Inject session history if available
        if session_history:
            agent.chat_history = session_history.copy()

        result = agent.query(query)

        # Build sources from pdf_links
        pdf_sources = []
        for link in result.get("pdf_links", []):
            pdf_sources.append({
                "title": link.split("/")[-1] or "Legal Document",
                "document_link": link
            })

        return ChatResponse(
            answer=result.get("answer", ""),
            raw_results=result.get("answer", ""),
            tools_used=result.get("tools_used", []),
            sources=pdf_sources,
            query_time=time.time() - start_time,
            mode="web_search",
            metadata={
                "search_type": "agent_pdf_filtered",
                "pdf_links": result.get("pdf_links", [])
            }
        )

    def _extract_sources(self, text: str) -> List[Dict[str, Any]]:
        """Extract case sources from response text."""
        import re

        sources = []

        # Pattern to match case information
        case_pattern = r"\*\*Case \d+: (.+?)\*\*"
        court_pattern = r"- Court: (.+)"
        year_pattern = r"- Year: (\d{4}|N/A)"
        citation_pattern = r"- Citation: (.+)"
        link_pattern = r"- Document Link: (https?://[^\s]+)"

        # Find all cases
        case_blocks = re.split(r"\*\*Case \d+:", text)

        for i, block in enumerate(case_blocks[1:], 1):  # Skip first empty split
            source = {"case_number": i}

            # Extract title
            title_match = re.match(r"(.+?)\*\*", block)
            if title_match:
                source["title"] = title_match.group(1).strip()

            # Extract other fields
            court_match = re.search(court_pattern, block)
            if court_match:
                source["court"] = court_match.group(1).strip()

            year_match = re.search(year_pattern, block)
            if year_match:
                source["year"] = year_match.group(1).strip()

            citation_match = re.search(citation_pattern, block)
            if citation_match:
                source["citation"] = citation_match.group(1).strip()

            link_match = re.search(link_pattern, block)
            if link_match:
                source["document_link"] = link_match.group(1).strip()

            if source.get("title"):
                sources.append(source)

        return sources

    def clear_history(self):
        """Clear conversation history."""
        if self._retrieval_agent:
            self._retrieval_agent.clear_history()
        if self._pipeline:
            self._pipeline.clear_history()
        if self._web_search_agent:
            self._web_search_agent.clear_history()

    def switch_mode(self, mode: Literal["single", "pipeline", "web_search"]):
        """Switch between single, pipeline, and web_search mode."""
        if mode in ("pipeline", "web_search") and not self._gemini_available:
            raise ValueError(f"{mode} mode requires GOOGLE_API_KEY")
        self.mode = mode

    @property
    def available_modes(self) -> List[str]:
        """Get list of available modes."""
        modes = ["single"]
        if self._gemini_available:
            modes.append("pipeline")
            modes.append("web_search")
        return modes

    def get_status(self) -> Dict[str, Any]:
        """Get chatbot status information."""
        return {
            "mode": self.mode,
            "retrieval_model": self.retrieval_model,
            "response_model": self.response_model,
            "gemini_available": self._gemini_available,
            "available_modes": self.available_modes
        }


# Convenience function for quick queries
def quick_chat(query: str, mode: str = "single") -> str:
    """
    Quick function to get a response without managing sessions.

    Args:
        query: User's question
        mode: "single" or "pipeline"

    Returns:
        Response text
    """
    chatbot = LegalChatbot(mode=mode)
    response = chatbot.chat(query)
    return response.answer
