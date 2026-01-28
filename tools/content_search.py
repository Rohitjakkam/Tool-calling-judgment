# tools/content_search.py
"""
Tools for searching legal judgments by content, topics, and keywords.
Uses modern LangChain @tool decorator pattern.
"""

from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import List, Optional

from .base import execute_search, format_results_to_string


# ============================================================
# Schema Definitions
# ============================================================

class TopicSearchInput(BaseModel):
    """Input for topic-based search."""
    topic: str = Field(
        description="Legal topic (e.g., 'murder', 'bail', 'divorce', 'property dispute', 'cheque bounce', 'dowry')"
    )
    court: Optional[str] = Field(
        default=None,
        description="Court filter (e.g., 'supreme court', 'delhi high court')"
    )
    year_from: Optional[int] = Field(default=None, description="Start year filter")
    year_to: Optional[int] = Field(default=None, description="End year filter")
    size: int = Field(default=10, description="Number of results to return")


class KeywordSearchInput(BaseModel):
    """Input for keyword search."""
    keywords: str = Field(description="Keywords to search in judgment content")
    must_include: Optional[List[str]] = Field(
        default=None,
        description="Phrases that MUST appear in the judgment"
    )
    exclude: Optional[List[str]] = Field(
        default=None,
        description="Words or phrases to exclude from results"
    )
    court: Optional[str] = Field(default=None, description="Court filter")
    size: int = Field(default=10, description="Number of results to return")


class BooleanSearchInput(BaseModel):
    """Input for boolean search."""
    must_terms: List[str] = Field(
        description="Terms/phrases that MUST appear (AND logic)"
    )
    should_terms: Optional[List[str]] = Field(
        default=None,
        description="Terms that SHOULD appear for better ranking (OR logic)"
    )
    must_not_terms: Optional[List[str]] = Field(
        default=None,
        description="Terms that must NOT appear (exclusion)"
    )
    court: Optional[str] = Field(default=None, description="Court filter")
    year: Optional[int] = Field(default=None, description="Year filter")
    size: int = Field(default=10, description="Number of results to return")


class CaseTypeInput(BaseModel):
    """Input for case type search."""
    case_type: str = Field(
        description="Type of case (e.g., 'criminal appeal', 'civil appeal', 'writ petition', 'SLP', 'review petition')"
    )
    court: Optional[str] = Field(default=None, description="Court filter")
    year: Optional[int] = Field(default=None, description="Year filter")
    topic: Optional[str] = Field(default=None, description="Additional topic filter")
    size: int = Field(default=10, description="Number of results to return")


# ============================================================
# Tool Definitions
# ============================================================

@tool(args_schema=TopicSearchInput)
def search_by_legal_topic(
    topic: str,
    court: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    size: int = 10
) -> str:
    """Search judgments by legal topic or subject matter.

    Use when user asks for cases on a specific legal topic like:
    - 'Bail cases'
    - 'Murder judgments'
    - 'Property dispute cases'
    - 'Divorce cases'
    - 'Employment termination cases'
    - 'Cheque bounce matters'

    Args:
        topic: Legal topic to search
        court: Optional court filter
        year_from: Optional start year
        year_to: Optional end year
        size: Number of results
    """
    must_clauses = [
        {
            "match": {
                "keywords": {
                    "query": topic,
                    "boost": 15
                }
            }
        }
    ]

    should_clauses = [
        {
            "match": {
                "page_content": {
                    "query": topic,
                    "boost": 3
                }
            }
        }
    ]

    filters = []
    if court:
        filters.append({"match": {"court_name": court.lower()}})
    if year_from and year_to:
        filters.append({"range": {"year": {"gte": year_from, "lte": year_to}}})
    elif year_from:
        filters.append({"range": {"year": {"gte": year_from}}})
    elif year_to:
        filters.append({"range": {"year": {"lte": year_to}}})

    query = {
        "size": min(size, 20),
        "query": {
            "bool": {
                "must": must_clauses,
                "should": should_clauses,
                "filter": filters
            }
        }
    }

    response = execute_search(query)
    return format_results_to_string(response, max_results=size)


@tool(args_schema=KeywordSearchInput)
def search_by_keywords(
    keywords: str,
    must_include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    court: Optional[str] = None,
    size: int = 10
) -> str:
    """Full-text keyword search in judgment content.

    Use for general searches when specific tools don't apply.
    Supports must-include and exclusion terms.

    Args:
        keywords: Keywords to search
        must_include: Phrases that must appear
        exclude: Terms to exclude
        court: Optional court filter
        size: Number of results
    """
    must_clauses = [
        {
            "match": {
                "page_content": {
                    "query": keywords,
                    "operator": "or",
                    "minimum_should_match": "50%"
                }
            }
        }
    ]

    if must_include:
        for phrase in must_include:
            must_clauses.append({
                "match_phrase": {"page_content": phrase}
            })

    must_not_clauses = []
    if exclude:
        for term in exclude:
            must_not_clauses.append({
                "match": {"page_content": term}
            })

    filters = []
    if court:
        filters.append({"match": {"court_name": court.lower()}})

    query = {
        "size": min(size, 20),
        "query": {
            "bool": {
                "must": must_clauses,
                "must_not": must_not_clauses,
                "filter": filters
            }
        }
    }

    response = execute_search(query)
    return format_results_to_string(response, max_results=size)


@tool(args_schema=BooleanSearchInput)
def advanced_boolean_search(
    must_terms: List[str],
    should_terms: Optional[List[str]] = None,
    must_not_terms: Optional[List[str]] = None,
    court: Optional[str] = None,
    year: Optional[int] = None,
    size: int = 10
) -> str:
    """Advanced boolean search with AND/OR/NOT logic.

    Use for complex queries requiring multiple conditions like:
    - 'Must contain fraud AND misrepresentation'
    - 'Cases with bail but NOT anticipatory bail'
    - 'Judgments mentioning both negligence and compensation'

    Args:
        must_terms: Terms that must all appear (AND)
        should_terms: Terms where any should appear (OR)
        must_not_terms: Terms that must not appear (NOT)
        court: Optional court filter
        year: Optional year filter
        size: Number of results
    """
    must_clauses = [
        {"match_phrase": {"page_content": term}}
        for term in must_terms
    ]

    should_clauses = [
        {"match": {"page_content": term}}
        for term in (should_terms or [])
    ]

    must_not_clauses = [
        {"match_phrase": {"page_content": term}}
        for term in (must_not_terms or [])
    ]

    filters = []
    if court:
        filters.append({"match": {"court_name": court.lower()}})
    if year:
        filters.append({"term": {"year": year}})

    query = {
        "size": min(size, 20),
        "query": {
            "bool": {
                "must": must_clauses,
                "should": should_clauses,
                "must_not": must_not_clauses,
                "filter": filters
            }
        }
    }

    response = execute_search(query)
    return format_results_to_string(response, max_results=size)


@tool(args_schema=CaseTypeInput)
def search_by_case_type(
    case_type: str,
    court: Optional[str] = None,
    year: Optional[int] = None,
    topic: Optional[str] = None,
    size: int = 10
) -> str:
    """Search judgments by case type or nature.

    Use when user asks for specific case types like:
    - 'Criminal appeals'
    - 'Writ petitions'
    - 'Special Leave Petitions (SLP)'
    - 'Civil suits'
    - 'Review petitions'
    - 'Transfer petitions'

    Args:
        case_type: Type of case
        court: Optional court filter
        year: Optional year filter
        topic: Optional topic filter
        size: Number of results
    """
    must_clauses = [
        {
            "match": {
                "case_type": {
                    "query": case_type,
                    "boost": 10
                }
            }
        }
    ]

    if topic:
        must_clauses.append({
            "match": {
                "keywords": {
                    "query": topic,
                    "boost": 5
                }
            }
        })

    filters = []
    if court:
        filters.append({"match": {"court_name": court.lower()}})
    if year:
        filters.append({"term": {"year": year}})

    query = {
        "size": min(size, 20),
        "query": {
            "bool": {
                "must": must_clauses,
                "filter": filters
            }
        }
    }

    response = execute_search(query)
    return format_results_to_string(response, max_results=size)
