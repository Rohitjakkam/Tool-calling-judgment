# tools/web_search_agent.py
"""
Web Search Agent with PDF-filtering tools for legal research.

Uses Gemini 2.5 Flash as the agent LLM with 3 specialized tools:
1. search_legal_web - Search Google via Gemini + Google Search grounding
2. filter_document_links - Filter URLs to keep only PDF/legal document links
3. fetch_document_content - Fetch and extract text from PDF/HTML legal documents

The agent searches the web, filters for PDF/legal document links,
fetches content from those links, and generates answers ONLY from the fetched content.
"""

import re
import io
import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage


# ============================================================
# Tool Input Schemas
# ============================================================

class SearchLegalWebInput(BaseModel):
    """Input for web search tool."""
    query: str = Field(description="The legal search query to search on Google")


class FilterDocumentLinksInput(BaseModel):
    """Input for document link filter tool."""
    urls: str = Field(description="Newline-separated list of URLs to filter for PDF/legal document links")


class FetchDocumentContentInput(BaseModel):
    """Input for document content fetcher tool."""
    url: str = Field(description="The URL of the PDF or legal document to fetch and extract text from")


# ============================================================
# Tool 1: Search Legal Web
# ============================================================

@tool("search_legal_web", args_schema=SearchLegalWebInput)
def search_legal_web(query: str) -> str:
    """Search Google for Indian legal content using Gemini with Google Search grounding.

    Returns a brief summary of search results along with ALL URLs found in the
    grounding metadata. Use filter_document_links next to filter for PDF/legal
    document links only.
    """
    from websearch import Scenario_qa

    try:
        response = Scenario_qa(query, [])
    except Exception as e:
        return f"Error searching the web: {e}"

    # Extract text from response
    answer_text = ""
    if hasattr(response, "text"):
        answer_text = response.text
    elif hasattr(response, "candidates") and response.candidates:
        parts = response.candidates[0].content.parts
        answer_text = "".join(p.text for p in parts if hasattr(p, "text"))

    # Extract URLs from grounding metadata
    grounding_urls = []
    if hasattr(response, "candidates") and response.candidates:
        candidate = response.candidates[0]
        grounding = getattr(candidate, "grounding_metadata", None)
        if grounding:
            chunks = getattr(grounding, "grounding_chunks", []) or []
            for chunk in chunks:
                web = getattr(chunk, "web", None)
                if web:
                    uri = getattr(web, "uri", "")
                    title = getattr(web, "title", "")
                    if uri:
                        grounding_urls.append({"url": uri, "title": title})

    # Also extract URLs from response text using regex
    url_pattern = r'https?://[^\s\)\]\>\"\'<,]+'
    text_urls = re.findall(url_pattern, answer_text)
    # Add text URLs not already in grounding
    existing_urls = {u["url"] for u in grounding_urls}
    for url in text_urls:
        url = url.rstrip(".")
        if url not in existing_urls:
            grounding_urls.append({"url": url, "title": ""})
            existing_urls.add(url)

    # Format output
    output_parts = []
    output_parts.append("## Search Summary (DO NOT use this for final answer)")
    output_parts.append(answer_text[:1500] if len(answer_text) > 1500 else answer_text)
    output_parts.append("")
    output_parts.append(f"## All URLs Found ({len(grounding_urls)} links)")
    output_parts.append("Pass these URLs to filter_document_links to get PDF/legal document links only.")
    output_parts.append("")

    url_list = []
    for i, item in enumerate(grounding_urls, 1):
        url_list.append(item["url"])
        title_str = f" ({item['title']})" if item["title"] else ""
        output_parts.append(f"{i}. {item['url']}{title_str}")

    output_parts.append("")
    output_parts.append("## URLs for filtering (copy this to filter_document_links):")
    output_parts.append("\n".join(url_list))

    return "\n".join(output_parts)


# ============================================================
# Tool 2: Filter Document Links
# ============================================================

# Known legal document domains and patterns
LEGAL_DOCUMENT_PATTERNS = [
    # Direct PDF links
    r"\.pdf($|\?)",
    # Indian Kanoon (judgment pages)
    r"indiankanoon\.org/doc/\d+",
    # Supreme Court of India
    r"supremecourtofindia\.nic\.in",
    # SCC Online
    r"scconline\.com",
    # Manupatra
    r"manupatra\.com",
    # National Judicial Data Grid / JUDIS
    r"judis\.nic\.in",
    # Live Law (legal news with judgments)
    r"livelaw\.in",
    # Bar and Bench
    r"barandbench\.com",
    # High Court websites
    r"highcourt.*\.gov\.in",
    r"hc.*\.gov\.in",
    r"bombayhighcourt\.nic\.in",
    r"delhihighcourt\.nic\.in",
    r"allahabadhighcourt\.in",
    r"ghcourts\.gov\.in",
    r"phc\.gov\.in",
    r"mhc\.tn\.gov\.in",
    r"highcourtchd\.gov\.in",
    r"jharkhandhighcourt\.nic\.in",
    r"cghighcourt\.nic\.in",
    # Legislative sources
    r"legislative\.gov\.in",
    r"indiacode\.nic\.in",
    r"prsindia\.org",
    # Bare Acts Live
    r"bareactslive\.com",
    # Casemine
    r"casemine\.com",
    # Vakil No 1
    r"vakilno1\.com",
    # Legal Services India
    r"legalservicesindia\.com",
    # Latest Laws
    r"latestlaws\.com",
]


@tool("filter_document_links", args_schema=FilterDocumentLinksInput)
def filter_document_links(urls: str) -> str:
    """Filter a list of URLs to keep only PDF files and known Indian legal document sources.

    Takes newline-separated URLs and returns only those that are PDFs or from
    recognized legal document websites (indiankanoon, Supreme Court, High Courts,
    SCC Online, Manupatra, etc.).

    Use this after search_legal_web to get the relevant legal document links.
    Then use fetch_document_content on the filtered links to get the actual content.
    """
    url_list = [u.strip() for u in urls.strip().split("\n") if u.strip()]

    if not url_list:
        return "No URLs provided to filter."

    filtered = []
    for url in url_list:
        url_lower = url.lower()
        for pattern in LEGAL_DOCUMENT_PATTERNS:
            if re.search(pattern, url_lower):
                filtered.append(url)
                break

    if not filtered:
        return (
            f"No PDF or legal document links found among {len(url_list)} URLs.\n"
            "The search results did not contain links to known legal document sources.\n"
            "Try a more specific query or mention specific case names/citations."
        )

    output_parts = []
    output_parts.append(f"## Filtered Legal Document Links ({len(filtered)} of {len(url_list)} URLs)")
    output_parts.append("")
    output_parts.append("Use fetch_document_content on these URLs to get the actual content.")
    output_parts.append("")

    for i, url in enumerate(filtered, 1):
        # Classify the link type
        url_lower = url.lower()
        if ".pdf" in url_lower:
            link_type = "PDF"
        elif "indiankanoon.org" in url_lower:
            link_type = "Indian Kanoon Judgment"
        elif "supremecourtofindia" in url_lower:
            link_type = "Supreme Court"
        elif "scconline" in url_lower:
            link_type = "SCC Online"
        elif "livelaw" in url_lower:
            link_type = "Live Law"
        elif "barandbench" in url_lower:
            link_type = "Bar and Bench"
        elif "highcourt" in url_lower or "hc" in url_lower:
            link_type = "High Court"
        else:
            link_type = "Legal Source"

        output_parts.append(f"{i}. [{link_type}] {url}")

    return "\n".join(output_parts)


# ============================================================
# Tool 3: Fetch Document Content
# ============================================================

@tool("fetch_document_content", args_schema=FetchDocumentContentInput)
def fetch_document_content(url: str) -> str:
    """Fetch and extract text content from a PDF or legal document URL.

    Handles both PDF files (extracts text with pypdf) and HTML pages
    (extracts text with BeautifulSoup). Returns the extracted text content.

    Use this on URLs returned by filter_document_links to get the actual
    legal document content for generating your answer.
    """
    import requests

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        resp = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        return f"Error: Request timed out for {url}. The server did not respond in time."
    except requests.exceptions.HTTPError as e:
        return f"Error: HTTP {resp.status_code} for {url}. The page may be restricted or unavailable."
    except requests.exceptions.RequestException as e:
        return f"Error: Could not fetch {url}. {str(e)}"

    content_type = resp.headers.get("Content-Type", "").lower()

    # Handle PDF content
    if "application/pdf" in content_type or url.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader
            pdf_reader = PdfReader(io.BytesIO(resp.content))
            text_parts = []
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            text = "\n\n".join(text_parts)

            if not text.strip():
                return f"PDF at {url} was fetched but contains no extractable text (possibly scanned/image-based)."

            # Truncate to reasonable length
            if len(text) > 5000:
                text = text[:5000] + "\n\n[... content truncated, total length: {} chars ...]".format(len(text))

            return f"## PDF Content from: {url}\n\n{text}"
        except Exception as e:
            return f"Error extracting PDF text from {url}: {str(e)}"

    # Handle HTML content
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove script and style elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        # Try to find the main content area
        # Indian Kanoon specific
        main_content = soup.find("div", {"id": "judgments"})
        if not main_content:
            main_content = soup.find("div", {"class": "judgments"})
        if not main_content:
            main_content = soup.find("div", {"class": "doc_content"})
        if not main_content:
            main_content = soup.find("article")
        if not main_content:
            main_content = soup.find("main")
        if not main_content:
            main_content = soup.find("div", {"class": "content"})
        if not main_content:
            main_content = soup.find("body")

        if main_content:
            text = main_content.get_text(separator="\n", strip=True)
        else:
            text = soup.get_text(separator="\n", strip=True)

        # Clean up excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)

        if not text.strip():
            return f"Page at {url} was fetched but no meaningful text content was found."

        # Truncate to reasonable length
        if len(text) > 5000:
            text = text[:5000] + "\n\n[... content truncated, total length: {} chars ...]".format(len(text))

        # Extract title
        title = soup.find("title")
        title_text = title.get_text(strip=True) if title else "Unknown"

        return f"## Document Content from: {url}\n**Title:** {title_text}\n\n{text}"

    except Exception as e:
        return f"Error extracting content from {url}: {str(e)}"


# ============================================================
# Agent System Prompt
# ============================================================

WEB_SEARCH_SYSTEM_PROMPT = """You are a legal research agent that searches the web for Indian legal judgments and case laws.

You have 3 tools available. You MUST use them in this exact sequence:

## Step 1: Search the Web
Use `search_legal_web` to search Google for the user's query. This returns a summary and a list of URLs.
DO NOT use the summary for your final answer — it is only for context.

## Step 2: Filter for Legal Document Links
Use `filter_document_links` with all the URLs from Step 1. This filters out non-legal links and keeps only:
- PDF files (.pdf)
- Indian Kanoon judgment pages
- Supreme Court / High Court official pages
- SCC Online, Manupatra, and other legal databases

## Step 3: Fetch Document Content
Use `fetch_document_content` on the top filtered URLs (up to 5 links) to extract their actual text content.
Call this tool once for each URL you want to fetch.

## Step 4: Generate Your Answer
After fetching document content, generate your final answer based EXCLUSIVELY on the content you fetched.

### CRITICAL RULES:
1. Your answer must ONLY reference information from the fetched documents (Step 3).
2. DO NOT use the initial search summary from Step 1 as a source for your answer.
3. Always cite which document/URL each piece of information came from.
4. If no legal document links were found, clearly state that and provide the search summary as a fallback.
5. Format your answer with proper legal structure: case names, citations, courts, years, and key holdings.
6. List all PDF/document links at the end of your response under a "Sources" section.

### Response Format:
- Start with a direct answer to the query
- Cite specific cases with their details from the fetched documents
- Include a "Sources" section listing all PDF/document URLs used
"""


# ============================================================
# Agent Factory & Wrapper
# ============================================================

def create_web_search_agent(
    model: str = "gemini-2.5-flash",
    temperature: float = 0.3,
    verbose: bool = True
):
    """
    Create a web search agent with PDF-filtering tools.

    Args:
        model: Gemini model to use
        temperature: LLM temperature
        verbose: Whether to print agent steps

    Returns:
        Agent ready to process web search queries
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "Google API key not found. Please set GOOGLE_API_KEY in your .env file.\n"
            "Get a free API key at: https://aistudio.google.com/apikey"
        )

    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=api_key
    )

    tools = [search_legal_web, filter_document_links, fetch_document_content]

    agent = create_agent(
        llm,
        tools=tools,
        system_prompt=WEB_SEARCH_SYSTEM_PROMPT,
    )

    return agent


def run_web_query(
    agent,
    query: str,
    chat_history: Optional[List] = None
) -> Dict[str, Any]:
    """
    Run a query through the web search agent.

    Args:
        agent: Agent instance
        query: User query
        chat_history: Optional chat history for context

    Returns:
        Dictionary with answer, tools_used, and pdf_links
    """
    messages = []

    if chat_history:
        messages.extend(chat_history)

    messages.append({"role": "user", "content": query})

    result = agent.invoke({"messages": messages})

    output_messages = result.get("messages", [])
    answer = ""
    tools_used = []
    pdf_links = []

    for msg in output_messages:
        if hasattr(msg, 'content') and msg.content:
            if hasattr(msg, 'type') and msg.type == 'ai':
                answer = msg.content
            elif isinstance(msg, dict) and msg.get('role') == 'assistant':
                answer = msg.get('content', '')

        # Track tool calls
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tool_call in msg.tool_calls:
                name = ""
                if isinstance(tool_call, dict):
                    name = tool_call.get('name', '')
                elif hasattr(tool_call, 'name'):
                    name = tool_call.name
                if name:
                    tools_used.append(name)

        # Extract PDF links from filter_document_links tool results
        if hasattr(msg, 'type') and msg.type == 'tool':
            content = msg.content if hasattr(msg, 'content') else ""
            if "Filtered Legal Document Links" in str(content):
                # Extract URLs from the filtered results
                url_pattern = r'https?://[^\s\)\]\>\"\'<,]+'
                found_urls = re.findall(url_pattern, str(content))
                pdf_links.extend(found_urls)

    # Fallback for answer
    if not answer and output_messages:
        last_msg = output_messages[-1]
        if hasattr(last_msg, 'content'):
            answer = last_msg.content
        elif isinstance(last_msg, dict):
            answer = last_msg.get('content', str(last_msg))

    return {
        "answer": answer,
        "tools_used": tools_used,
        "pdf_links": list(set(pdf_links)),
        "messages": output_messages
    }


class WebSearchAgent:
    """
    Wrapper class for the web search agent with conversation memory.
    Searches the web, filters for PDF/legal document links, fetches content,
    and generates answers exclusively from the fetched document content.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.3,
        verbose: bool = True
    ):
        """
        Initialize the web search agent.

        Args:
            model: Gemini model to use for the agent
            temperature: LLM temperature
            verbose: Whether to print agent steps
        """
        self.agent = create_web_search_agent(model, temperature, verbose)
        self.chat_history: List = []

    def query(self, user_input: str) -> Dict[str, Any]:
        """
        Process a user query through the web search agent.

        Args:
            user_input: User's question or request

        Returns:
            Dictionary with answer, tools_used, pdf_links, and messages
        """
        result = run_web_query(self.agent, user_input, self.chat_history)

        self.chat_history.append({"role": "user", "content": user_input})
        self.chat_history.append({"role": "assistant", "content": result["answer"]})

        return result

    def clear_history(self):
        """Clear the conversation history."""
        self.chat_history = []

    def get_history(self) -> List:
        """Get the current conversation history."""
        return self.chat_history
