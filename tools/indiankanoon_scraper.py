# tools/indiankanoon_scraper.py
"""
Indian Kanoon Browsing Agent — LLM-driven web scraper.

Uses a Gemini-powered agent with 2 browsing tools to autonomously search
indiankanoon.org for Supreme Court judgments. The agent decides which
search queries to try, which results to click, and how to extract
structured case data — replacing the previous rigid hard-coded pipeline.

Architecture:
- Agent LLM: Gemini 2.5 Flash (via LangChain create_agent)
- Tool 1: search_on_indiankanoon — search IK and return results
- Tool 2: fetch_ik_page — fetch a judgment page and return text

Acts as a fallback when the local Elasticsearch database doesn't have the case.
"""

import re
import os
import time
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain.agents import create_agent

from .base import setup_logger

logger = setup_logger(__name__)


# ============================================================
# Constants
# ============================================================

BASE_URL = "https://indiankanoon.org"
BROWSE_SC_URL = f"{BASE_URL}/browse/supremecourt/"
SEARCH_URL = f"{BASE_URL}/search/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Polite delay between requests (seconds)
REQUEST_DELAY = 1.0


# ============================================================
# Gemini Helper — LangChain ChatGoogleGenerativeAI
# ============================================================

def _get_gemini_llm():
    """Get a Gemini LLM instance for the browsing agent."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY required for Indian Kanoon scraper. "
            "Set it in your .env file."
        )
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=api_key,
    )


# ============================================================
# Fetching Helpers (used by agent tools)
# ============================================================

def _fetch_page(url: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Fetch a page from Indian Kanoon with polite delay.

    Returns:
        Tuple of (html_content, error_message).
        On success: (html, None)
        On failure: (None, error_description)
    """
    import requests

    logger.debug("Fetching URL: %s", url)
    time.sleep(REQUEST_DELAY)
    try:
        fetch_start = time.time()
        resp = requests.get(url, headers=HEADERS, timeout=15)
        fetch_elapsed = time.time() - fetch_start
        logger.debug("HTTP %d | %.2fs | %d bytes | %s", resp.status_code, fetch_elapsed, len(resp.text), url[:80])

        if resp.status_code == 403:
            logger.warning("HTTP 403 Forbidden from Indian Kanoon — blocked")
            return None, f"HTTP 403 Forbidden — Indian Kanoon blocked the request"
        if resp.status_code == 429:
            logger.warning("HTTP 429 Too Many Requests — rate limited by Indian Kanoon")
            return None, f"HTTP 429 Too Many Requests — rate limited"
        resp.raise_for_status()
        return resp.text, None
    except requests.exceptions.Timeout:
        logger.error("Request timed out (15s) for URL: %s", url[:100])
        return None, "Request timed out (15s)"
    except requests.exceptions.ConnectionError as e:
        logger.error("Connection error for URL: %s — %s", url[:80], str(e)[:80])
        return None, "Connection error — network issue or IK is down"
    except Exception as e:
        logger.error("HTTP error for URL: %s — %s", url[:80], str(e)[:100])
        return None, f"HTTP error: {str(e)[:100]}"


def _parse_search_results(html: str) -> List[Dict[str, str]]:
    """
    Parse Indian Kanoon search results page.
    Returns list of {title, date, url, snippet}.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    results = []

    # Indian Kanoon search results are in divs with class "result"
    # Each result has a title link and a snippet
    result_divs = soup.find_all("div", class_="result")

    for div in result_divs:
        entry = {}

        # Title link — first <a> tag with /doc/ URL
        title_link = div.find("a", href=re.compile(r"/doc/\d+"))
        if title_link:
            entry["title"] = title_link.get_text(strip=True)
            href = title_link.get("href", "")
            entry["url"] = f"{BASE_URL}{href}" if href.startswith("/") else href

            # Extract doc ID
            doc_match = re.search(r"/doc/(\d+)", href)
            entry["doc_id"] = doc_match.group(1) if doc_match else ""

        # Snippet — text content after the title
        snippet_div = div.find("div", class_="result_text")
        if snippet_div:
            entry["snippet"] = snippet_div.get_text(strip=True)[:300]
        else:
            # Fallback: get all text from the div
            entry["snippet"] = div.get_text(strip=True)[:300]

        if entry.get("title") and entry.get("url"):
            results.append(entry)

    # Fallback: if no result divs found, try a more generic approach
    if not results:
        logger.debug("No 'result' divs found, trying generic /doc/ link approach")
        # Look for any /doc/ links
        doc_links = soup.find_all("a", href=re.compile(r"/doc/\d+"))
        for link in doc_links:
            title_text = link.get_text(strip=True)
            if len(title_text) > 10:  # Filter out short non-title links
                href = link.get("href", "")
                results.append({
                    "title": title_text,
                    "url": f"{BASE_URL}{href}" if href.startswith("/") else href,
                    "doc_id": re.search(r"/doc/(\d+)", href).group(1) if re.search(r"/doc/(\d+)", href) else "",
                    "snippet": "",
                })

    logger.info("Parsed %d search results from IK page", len(results))
    for i, r in enumerate(results[:5]):
        logger.info("  IK Result %d: '%s'", i + 1, r.get("title", "?")[:80])

    return results


def _get_total_results(html: str) -> int:
    """Extract total result count from search page."""
    match = re.search(r'of\s+([\d,]+)', html)
    if match:
        return int(match.group(1).replace(",", ""))
    return 0


def _get_pagination_info(html: str) -> Dict[str, Any]:
    """Extract pagination info — total results and whether there's a next page."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")

    total = _get_total_results(html)

    # Check for "Next" link
    next_link = soup.find("a", string=re.compile(r"Next", re.IGNORECASE))
    has_next = next_link is not None

    return {"total": total, "has_next": has_next}


def _extract_judgment_text(html: str) -> Dict[str, str]:
    """Extract the judgment text and metadata from a /doc/ page."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    # Remove scripts, styles, nav
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # Indian Kanoon judgment content is in div#judgments
    judgment_div = soup.find("div", {"id": "judgments"})
    if not judgment_div:
        judgment_div = soup.find("div", class_="judgments")
    if not judgment_div:
        judgment_div = soup.find("div", class_="doc_content")

    if judgment_div:
        text = judgment_div.get_text(separator="\n", strip=True)
    else:
        # Fallback to body
        body = soup.find("body")
        text = body.get_text(separator="\n", strip=True) if body else ""

    # Clean up
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    # Extract title
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    # Extract the doc title heading
    doc_title = soup.find("h2", class_="doc_title")
    if not doc_title:
        doc_title = soup.find("h2")
    heading = doc_title.get_text(strip=True) if doc_title else title

    return {
        "title": heading,
        "page_title": title,
        "text": text,
        "text_length": len(text),
    }


# ============================================================
# URL Builder
# ============================================================

def _build_search_url(search_terms: str, year: Optional[int] = None, page: int = 0) -> str:
    """Build Indian Kanoon search URL with doctypes and year filter."""
    import urllib.parse

    form_input = search_terms
    form_input += " doctypes:supremecourt"
    if year:
        form_input += f" year:{year}"

    params = {"formInput": form_input}
    if page > 0:
        params["pagenum"] = str(page)

    return f"{SEARCH_URL}?{urllib.parse.urlencode(params)}"


# ============================================================
# Agent Tool Schemas
# ============================================================

class SearchOnIndianKanoonInput(BaseModel):
    """Input schema for searching Indian Kanoon."""
    query: str = Field(
        description=(
            "Search terms to use on Indian Kanoon. "
            "Use party names, keywords, or case references. "
            "Examples: 'Maneka Gandhi v Union of India', 'Jindal Equipment Leasing Commissioner Income Tax'"
        )
    )
    year: Optional[int] = Field(
        default=None,
        description="Optional year filter to narrow results (e.g., 2005). Omit to search all years."
    )


class FetchIKPageInput(BaseModel):
    """Input schema for fetching an Indian Kanoon page."""
    url: str = Field(
        description=(
            "Full Indian Kanoon URL to fetch. "
            "Must be a valid indiankanoon.org URL. "
            "Example: 'https://indiankanoon.org/doc/1766147/'"
        )
    )


# ============================================================
# Agent Tools
# ============================================================

@tool("search_on_indiankanoon", args_schema=SearchOnIndianKanoonInput)
def search_on_indiankanoon(query: str, year: Optional[int] = None) -> str:
    """Search indiankanoon.org for Supreme Court judgments.

    Returns a numbered list of search results with titles, snippets, and URLs.
    Use different search terms or add/remove the year filter to refine results.

    Tips:
    - Start without year filter, add it only if needed
    - Try simpler search terms if no results found
    - Try swapping party names or removing prefixes like "M/S", "Dr."

    Args:
        query: Search terms (party names, keywords)
        year: Optional year filter
    """
    logger.info("Tool search_on_indiankanoon: query='%s', year=%s", query[:60], year)

    url = _build_search_url(query, year=year)
    logger.debug("Search URL: %s", url[:120])

    html, error = _fetch_page(url)
    if error:
        logger.warning("Search failed: %s", error)
        return f"ERROR: {error}\nTry a different search query or remove the year filter."

    if not html:
        return "ERROR: Empty response from Indian Kanoon. Try again with different terms."

    # Parse results
    results = _parse_search_results(html)
    pagination = _get_pagination_info(html)

    # Format output for the agent
    year_label = f" (year: {year})" if year else ""
    output = [f"## Indian Kanoon Search Results"]
    output.append(f"Search: \"{query}\"{year_label}")
    output.append(f"Total results: {pagination['total']} | More pages: {'Yes' if pagination['has_next'] else 'No'}")
    output.append("")

    if not results:
        output.append("No results found. Try different search terms.")
        return "\n".join(output)

    for i, r in enumerate(results[:10], 1):
        output.append(f"{i}. **{r['title']}**")
        if r.get("snippet"):
            output.append(f"   {r['snippet'][:200]}")
        output.append(f"   URL: {r['url']}")
        output.append("")

    return "\n".join(output)


@tool("fetch_ik_page", args_schema=FetchIKPageInput)
def fetch_ik_page(url: str) -> str:
    """Fetch an Indian Kanoon judgment page and extract its text content.

    Returns the judgment title and full text (truncated to ~8000 chars).
    Use this after finding a promising result via search_on_indiankanoon.

    Args:
        url: Full Indian Kanoon URL (e.g., https://indiankanoon.org/doc/12345/)
    """
    logger.info("Tool fetch_ik_page: url='%s'", url[:80])

    html, error = _fetch_page(url)
    if error:
        logger.warning("Fetch failed: %s", error)
        return f"ERROR: {error}\nCould not fetch the judgment page."

    if not html:
        return "ERROR: Empty response from Indian Kanoon."

    data = _extract_judgment_text(html)
    logger.info("Judgment extracted: %d chars | title='%s'", data["text_length"], data.get("title", "?")[:60])

    # Truncate text for token management
    text_preview = data["text"][:8000]

    output = [
        f"## Judgment Content",
        f"**Title:** {data['title']}",
        f"**URL:** {url}",
        f"**Total Length:** {data['text_length']:,} characters",
        "",
        text_preview,
    ]

    if data["text_length"] > 8000:
        output.append(f"\n... [truncated, showing first 8000 of {data['text_length']:,} chars]")

    return "\n".join(output)


# ============================================================
# Agent System Prompt
# ============================================================

IK_AGENT_SYSTEM_PROMPT = """You are a legal research assistant that searches indiankanoon.org to find specific Indian Supreme Court judgments.

You have 2 tools:
1. `search_on_indiankanoon(query, year)` — Search Indian Kanoon. Returns numbered results with titles, snippets, and URLs.
2. `fetch_ik_page(url)` — Fetch a judgment page. Returns title + text content.

## Your Task
Given a case reference (party names, year, citation), find the exact judgment on Indian Kanoon, fetch it, and extract structured case data.

## Search Strategy
1. Start with the most natural search: "Petitioner v Respondent" (WITHOUT year filter first — it's usually more effective)
2. Examine the search results carefully. Look for party name matches (even partial).
3. If no match, try these reformulations one at a time:
   - Remove honorifics: "Dr.", "Smt.", "Shri." → just the name
   - Remove company prefixes: "M/S", "M/s", "Messrs" → just the company name
   - Swap "Government of India" ↔ "Union of India" (they are the SAME entity in Indian law)
   - "Commissioner of Income Tax" can appear as "CIT" or "C.I.T."
   - Remove "And Ors.", "& Anr." — try just the core party name
   - Try just the petitioner name alone (especially if respondent is a government body)
   - Add year filter if you haven't, or remove it if you have
4. When you spot a promising result, fetch it with fetch_ik_page
5. Give up after ~6-8 search attempts if nothing matches

## Name Matching Rules
- "SONA BALA BORA AND ORS." matches "Sona Bala Bora" (ignore suffixes)
- "M/S Jindal Equipment Leasing And Finance Company Ltd" matches "Jindal Equipment Leasing" (core name present)
- "Government of India" = "Union of India" = "UOI" = "Govt. of India"
- "State of X" includes its departments and officers
- Ignore case differences when comparing names

## Output Format
After fetching the judgment, extract and return this structured data:

**Case Name:** [Full case name with parties]
**Citation:** [Official citation if found, e.g., (2005) 4 SCC 501 or AIR 2005 SC xxx]
**Court:** [Court name]
**Year:** [Year of judgment]
**Judges:** [Names of judges on the bench]
**Key Holdings:** [2-3 sentence summary of the key legal holdings]
**Legal Principles:** [Key legal principles established]
**Document Link:** [Indian Kanoon URL]

If any field is not found in the text, write "Not available".

## Error Handling
- If you get HTTP 403 or 429 errors, try a different query formulation (the server may be rate-limiting)
- If search returns no results, reformulate — don't repeat the same query
- If after all attempts you cannot find the case, say so clearly and list what you tried

## Important Rules
- Be efficient: don't search for the same terms twice
- The first search result is often the best match for well-known cases
- Do NOT fabricate any information — only report what you find in the actual judgment text
- Always include the Indian Kanoon URL in your response
"""


# ============================================================
# Agent Factory & Runner
# ============================================================

def _create_ik_agent():
    """Create the Indian Kanoon browsing agent with Gemini 2.5 Flash."""
    llm = _get_gemini_llm()

    tools = [search_on_indiankanoon, fetch_ik_page]

    agent = create_agent(
        llm,
        tools=tools,
        system_prompt=IK_AGENT_SYSTEM_PROMPT,
    )

    logger.info("IK browsing agent created (Gemini 2.5 Flash, 2 tools)")
    return agent


def run_ik_agent(query: str) -> str:
    """
    Run the Indian Kanoon browsing agent to find a case.

    Creates the agent, invokes it with the user query, and returns
    the structured result or an error message.

    Args:
        query: Case reference (party names, citation, etc.)

    Returns:
        Formatted string with structured case data or "not found" message.
    """
    logger.info("=" * 60)
    logger.info("IK BROWSING AGENT started | query='%s'", query)
    logger.info("=" * 60)

    agent_start = time.time()

    try:
        agent = _create_ik_agent()

        result = agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            {"recursion_limit": 25},  # Safety: max 25 tool call rounds
        )

        # Extract the final AI message
        output_messages = result.get("messages", [])
        answer = ""
        tools_used = []

        def _content_to_str(content) -> str:
            """Normalize message content to a plain string.
            Gemini can return content as a list of blocks like
            [{"type": "text", "text": "..."}] instead of a plain string.
            """
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, str):
                        parts.append(block)
                    elif isinstance(block, dict) and "text" in block:
                        parts.append(block["text"])
                    elif hasattr(block, "text"):
                        parts.append(block.text)
                    else:
                        parts.append(str(block))
                return "\n".join(parts)
            return str(content) if content else ""

        for msg in output_messages:
            if hasattr(msg, 'content') and msg.content:
                if hasattr(msg, 'type') and msg.type == 'ai':
                    answer = _content_to_str(msg.content)
                elif isinstance(msg, dict) and msg.get('role') == 'assistant':
                    answer = _content_to_str(msg.get('content', ''))

            # Track tool calls
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    if isinstance(tool_call, dict):
                        name = tool_call.get('name', '')
                    elif hasattr(tool_call, 'name'):
                        name = tool_call.name
                    else:
                        name = ''
                    if name:
                        tools_used.append(name)

        # Fallback: if answer is empty, try last message
        if not answer and output_messages:
            last_msg = output_messages[-1]
            if hasattr(last_msg, 'content'):
                answer = _content_to_str(last_msg.content)
            elif isinstance(last_msg, dict):
                answer = _content_to_str(last_msg.get('content', str(last_msg)))

        agent_elapsed = time.time() - agent_start
        logger.info("=" * 60)
        logger.info("IK BROWSING AGENT completed in %.2fs | tools_used=%s | answer_length=%d",
                     agent_elapsed, tools_used, len(answer))
        logger.info("=" * 60)

        # Wrap in a report header for consistency with the rest of the system
        report = [
            "## Indian Kanoon Web Search",
            f"**Query:** {query}",
            f"**Agent tools used:** {tools_used}",
            f"**Time:** {agent_elapsed:.1f}s",
            "",
            answer,
        ]

        return "\n".join(report)

    except Exception as e:
        agent_elapsed = time.time() - agent_start
        logger.error("IK BROWSING AGENT EXCEPTION in %.2fs: %s", agent_elapsed, str(e)[:200], exc_info=True)
        return (
            f"## Indian Kanoon Web Search\n\n"
            f"**Query:** {query}\n"
            f"**Error:** Could not complete the search: {str(e)}\n\n"
            f"This may be due to:\n"
            f"- Network connectivity issues\n"
            f"- Indian Kanoon rate limiting\n"
            f"- Missing GOOGLE_API_KEY for Gemini\n"
        )


# ============================================================
# Public Interface (backward-compatible)
# ============================================================

def scrape_indiankanoon_search(query: str, size: int = 3) -> str:
    """
    Search Indian Kanoon for a case using the LLM-driven browsing agent.

    This function delegates to the IK browsing agent which autonomously:
    1. Formulates search queries
    2. Examines search results and picks the best match
    3. Fetches the full judgment text
    4. Extracts structured case data

    Args:
        query: Case reference (party names, citation, etc.)
        size: Number of results (kept for backward compatibility, agent decides internally)

    Returns:
        Formatted string with case data or "not found" message.
    """
    return run_ik_agent(query)


# ============================================================
# LangChain Tool Definition (for the outer legal agent)
# ============================================================

class ScrapeIndianKanoonInput(BaseModel):
    """Input for Indian Kanoon scraper tool."""
    query: str = Field(
        description=(
            "The case to search for on Indian Kanoon. "
            "Can be party names, citation, or full case reference. "
            "Examples: 'Sona Bala Bora v. Jyotirindra Bhatacharjee, (2005)', "
            "'Maneka Gandhi vs Union of India 1978', "
            "'Kesavananda Bharati case'"
        )
    )
    size: int = Field(default=3, description="Number of results to return if multiple matches")


@tool("scrape_indiankanoon", args_schema=ScrapeIndianKanoonInput)
def scrape_indiankanoon(query: str, size: int = 3) -> str:
    """Search Indian Kanoon (indiankanoon.org) for Supreme Court judgments.

    This is an INTERNET FALLBACK tool — use it when the local Elasticsearch
    database does not have the case. It uses an AI browsing agent to
    autonomously search indiankanoon.org.

    How it works:
    1. An AI agent formulates search queries based on your input
    2. It searches indiankanoon.org and examines the results
    3. It picks the best matching result and fetches the judgment
    4. It extracts structured case data (citation, court, judges, key holdings)

    The agent automatically handles:
    - Name variations (Dr., M/S, And Ors., etc.)
    - Government of India ↔ Union of India equivalence
    - Query reformulation when initial search fails

    Use when:
    - smart_case_search found no results in the database
    - User specifically asks to search on Indian Kanoon
    - You need the actual judgment text from a known case

    Args:
        query: Case name, citation, or search terms
        size: Number of results
    """
    try:
        logger.info("scrape_indiankanoon tool called with query='%s'", query[:80])
        return scrape_indiankanoon_search(query, size)
    except Exception as e:
        logger.error("scrape_indiankanoon EXCEPTION: %s", str(e)[:200], exc_info=True)
        return (
            f"## Indian Kanoon Scraper Error\n\n"
            f"Could not complete the search: {str(e)}\n\n"
            f"This may be due to:\n"
            f"- Network connectivity issues\n"
            f"- Indian Kanoon rate limiting\n"
            f"- Missing GOOGLE_API_KEY for Gemini analysis\n"
        )
