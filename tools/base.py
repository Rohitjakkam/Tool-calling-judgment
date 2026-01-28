# tools/base.py
"""
Base utilities and shared functions for legal search tools.
"""

from elasticsearch import Elasticsearch
from langchain_core.documents import Document
from typing import List, Dict, Any
import time
from pprint import pprint

# Elasticsearch configuration
ES_URL = "http://139.84.219.174:9200"
ES_INDEX = "judgements"
ES_TIMEOUT = 60


def get_es_client(max_retries: int = 3, sleep_time: int = 2) -> Elasticsearch:
    """
    Get Elasticsearch client with retry logic.

    Args:
        max_retries: Maximum number of connection attempts
        sleep_time: Seconds to wait between retries

    Returns:
        Elasticsearch client instance
    """
    for i in range(max_retries):
        try:
            es = Elasticsearch(ES_URL, request_timeout=ES_TIMEOUT)
            es.info()
            pprint("Connected to Elasticsearch!")
            return es
        except Exception as e:
            pprint(f"Could not connect to Elasticsearch (attempt {i+1}/{max_retries}): {e}")
            if i < max_retries - 1:
                time.sleep(sleep_time)
    raise ConnectionError("Failed to connect to Elasticsearch after multiple attempts.")


def execute_search(query: Dict[str, Any], index: str = ES_INDEX) -> Dict[str, Any]:
    """
    Execute an Elasticsearch search query.

    Args:
        query: Query body
        index: Index name

    Returns:
        Search response
    """
    es = get_es_client()
    return es.options(request_timeout=ES_TIMEOUT).search(index=index, body=query)


def format_results(response: Dict[str, Any], include_score: bool = False) -> List[Document]:
    """
    Format Elasticsearch response into LangChain Documents.

    Args:
        response: Elasticsearch response
        include_score: Whether to include relevance score in metadata

    Returns:
        List of LangChain Document objects
    """
    documents = []
    for hit in response["hits"]["hits"]:
        source = hit["_source"]

        metadata = {
            "source": source.get("source", ""),
            "court": source.get("court_name", ""),
        }

        if "petitioner_names" in source and source["petitioner_names"]:
            petitioner = source["petitioner_names"][0] if isinstance(source["petitioner_names"], list) else source["petitioner_names"]
            metadata["petitioner"] = petitioner

        if "respondent_names" in source and source["respondent_names"]:
            respondent = source["respondent_names"][0] if isinstance(source["respondent_names"], list) else source["respondent_names"]
            metadata["respondent"] = respondent

        if "year" in source:
            metadata["year"] = source["year"]

        if "judge_name" in source:
            metadata["judge"] = source["judge_name"]

        if "citation" in source:
            metadata["citation"] = source["citation"]

        if "petitioner" in metadata and "respondent" in metadata:
            metadata["title"] = f"{metadata['petitioner']} vs {metadata['respondent']}"

        if include_score:
            metadata["score"] = hit["_score"]

        doc = Document(
            page_content=source.get("page_content", ""),
            metadata=metadata
        )
        documents.append(doc)

    return documents


def format_results_to_string(response: Dict[str, Any], max_results: int = 5) -> str:
    """
    Format Elasticsearch response into a readable string for LLM.

    Args:
        response: Elasticsearch response
        max_results: Maximum results to include

    Returns:
        Formatted string with case summaries
    """
    hits = response["hits"]["hits"][:max_results]

    if not hits:
        return "No matching judgments found."

    results = []
    for i, hit in enumerate(hits, 1):
        source = hit["_source"]

        # Build case title
        petitioner = source.get("petitioner_names", ["Unknown"])[0] if source.get("petitioner_names") else "Unknown"
        respondent = source.get("respondent_names", ["Unknown"])[0] if source.get("respondent_names") else "Unknown"
        title = f"{petitioner} vs {respondent}"

        court = source.get("court_name", "Unknown Court")
        year = source.get("year", "N/A")
        citation = source.get("citation", "N/A")

        # Get first 500 chars of content
        content = source.get("page_content", "")[:500]

        result = f"""
**Case {i}: {title}**
- Court: {court}
- Year: {year}
- Citation: {citation}
- Summary: {content}...
"""
        results.append(result)

    total = response["hits"]["total"]["value"] if isinstance(response["hits"]["total"], dict) else response["hits"]["total"]
    header = f"Found {total} matching judgments. Showing top {len(hits)}:\n"

    return header + "\n".join(results)
