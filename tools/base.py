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


def generate_s3_link(source_file: Any, court: str, source_name: str) -> str:
    """
    Generate S3 link for a judgment document.

    Args:
        source_file: Source file path from metadata
        court: Court name
        source_name: Processed source name (title)

    Returns:
        S3 public URL for the document
    """
    import os

    bucket_name = "lawttorney"
    region_name = "ap-south-1"

    # Normalize court name
    s3_root_folder = (court or "").strip().lower()

    # Process source file path
    if isinstance(source_file, dict):
        file_path = source_file.get("source", "")
    else:
        file_path = str(source_file) if source_file else ""

    # Normalize path for Linux/Windows
    file_path = file_path.replace("\\", "/")
    file = os.path.basename(file_path)

    # Generate the correct S3 key based on court type
    if s3_root_folder == "supreme":
        s3_key = f"{s3_root_folder}/{source_name}.pdf"
    elif s3_root_folder == "tribunal":
        s3_key = f"{s3_root_folder}/{file}"
    else:
        s3_key = f"{s3_root_folder}/{file}"

    # Build the S3 public link
    link = f"https://{bucket_name}.s3.{region_name}.amazonaws.com/{s3_key}"

    return link


def format_results_to_string(response: Dict[str, Any], max_results: int = 5) -> str:
    """
    Format Elasticsearch response into a readable string for LLM.

    Args:
        response: Elasticsearch response
        max_results: Maximum results to include

    Returns:
        Formatted string with case summaries and document links
    """
    import os

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

        # Validate year - if it looks wrong, try to extract from citation or page_content
        if year and isinstance(year, (int, float)):
            if year < 1900 or year > 2030:
                import re
                year_pattern = r'\b(19\d{2}|20[0-2]\d)\b'

                # First try to extract year from citation
                year_match = re.search(year_pattern, str(citation))
                if year_match:
                    year = int(year_match.group(1))
                else:
                    # Fallback: try to extract year from page_content
                    content_for_year = source.get("page_content", "")[:1000]  # Check first 1000 chars
                    year_match = re.search(year_pattern, content_for_year)
                    if year_match:
                        year = int(year_match.group(1))
                    else:
                        year = "N/A"

        # Get source file info
        source_file = source.get("source", "")

        # Process source name
        if isinstance(source_file, dict):
            file_path = source_file.get("source", str(source_file))
        else:
            file_path = str(source_file) if source_file else ""

        file_path = file_path.replace("\\", "/")
        source_basename = os.path.basename(file_path)
        source_name = os.path.splitext(source_basename)[0]

        # Use source_name from metadata if available, otherwise use title
        source_name = source.get("source_name", source_name) or title

        # Generate S3 document link
        doc_link = generate_s3_link(source_file, court, source_name)

        # Get first 500 chars of content
        content = source.get("page_content", "")[:500]

        result = f"""
**Case {i}: {title}**
- Court: {court}
- Year: {year}
- Citation: {citation}
- Document Link: {doc_link}
- Summary: {content}...
"""
        results.append(result)

    total = response["hits"]["total"]["value"] if isinstance(response["hits"]["total"], dict) else response["hits"]["total"]
    header = f"Found {total} matching judgments. Showing top {len(hits)}:\n"

    return header + "\n".join(results)
