# tools/court_search.py
"""
Tools for searching legal judgments by court, judge, and date.
Uses modern LangChain @tool decorator pattern.
"""

from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import Optional

from .base import execute_search, format_results_to_string


# ============================================================
# Schema Definitions
# ============================================================

class CourtSearchInput(BaseModel):
    """Input for court-based search."""
    court_name: str = Field(
        description="Court name (e.g., 'supreme court', 'delhi high court', 'bombay high court', 'madras high court')"
    )
    keywords: Optional[str] = Field(
        default=None,
        description="Additional keywords to filter results (e.g., 'bail', 'murder')"
    )
    year: Optional[int] = Field(default=None, description="Year filter (4-digit year)")
    size: int = Field(default=10, description="Number of results to return")


class JudgeSearchInput(BaseModel):
    """Input for judge-based search."""
    judge_name: str = Field(
        description="Name of the judge (e.g., 'DY Chandrachud', 'RF Nariman', 'NV Ramana')"
    )
    court: Optional[str] = Field(default=None, description="Court name filter")
    topic: Optional[str] = Field(default=None, description="Topic or subject filter")
    size: int = Field(default=10, description="Number of results to return")


class DateRangeInput(BaseModel):
    """Input for date range search."""
    year_from: int = Field(description="Start year (e.g., 2018)")
    year_to: int = Field(description="End year (e.g., 2023)")
    court: Optional[str] = Field(default=None, description="Court name filter")
    topic: Optional[str] = Field(default=None, description="Topic or keyword filter")
    size: int = Field(default=10, description="Number of results to return")


class BenchSizeInput(BaseModel):
    """Input for bench size search."""
    bench_size: int = Field(
        description="Number of judges (2 for division bench, 3 for full bench, 5+ for constitution bench)"
    )
    court: Optional[str] = Field(default=None, description="Court name filter")
    topic: Optional[str] = Field(default=None, description="Topic filter")
    size: int = Field(default=10, description="Number of results to return")


# ============================================================
# Tool Definitions
# ============================================================

@tool(args_schema=CourtSearchInput)
def search_by_court(
    court_name: str,
    keywords: Optional[str] = None,
    year: Optional[int] = None,
    size: int = 10
) -> str:
    """Search judgments from a specific court.

    Use when user asks for judgments from a particular court like:
    - 'Supreme Court judgments'
    - 'Delhi High Court cases'
    - 'Bombay High Court orders'
    - 'cases from Karnataka High Court'

    Args:
        court_name: Name of the court
        keywords: Optional keywords to filter
        year: Optional year filter
        size: Number of results
    """
    must_clauses = [
        {"match": {"court_name": court_name.lower()}}
    ]

    if keywords:
        must_clauses.append({
            "match": {
                "page_content": {
                    "query": keywords,
                    "minimum_should_match": "60%"
                }
            }
        })

    filters = []
    if year:
        filters.append({"term": {"year": year}})

    query = {
        "size": min(size, 20),
        "query": {
            "bool": {
                "must": must_clauses,
                "filter": filters
            }
        },
        "sort": [{"year": "desc"}]
    }

    response = execute_search(query)
    return format_results_to_string(response, max_results=size)


@tool(args_schema=JudgeSearchInput)
def search_by_judge(
    judge_name: str,
    court: Optional[str] = None,
    topic: Optional[str] = None,
    size: int = 10
) -> str:
    """Search judgments authored by a specific judge.

    Use when user asks for judgments by a particular judge like:
    - 'Judgments by Justice Chandrachud'
    - 'Cases decided by Justice RF Nariman'
    - 'Orders by Justice NV Ramana'

    Args:
        judge_name: Name of the judge
        court: Optional court filter
        topic: Optional topic filter
        size: Number of results
    """
    must_clauses = [
        {
            "match": {
                "judge_name": {
                    "query": judge_name,
                    "fuzziness": "AUTO",
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
                    "boost": 3
                }
            }
        })

    filters = []
    if court:
        filters.append({"match": {"court_name": court.lower()}})

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


@tool(args_schema=DateRangeInput)
def search_by_date_range(
    year_from: int,
    year_to: int,
    court: Optional[str] = None,
    topic: Optional[str] = None,
    size: int = 10
) -> str:
    """Search judgments within a specific year range.

    Use when user asks for time-bound searches like:
    - 'Cases from 2018 to 2022'
    - 'Recent judgments from last 5 years'
    - 'Judgments between 2015-2020'
    - 'bail cases from 2020 to 2023'

    Args:
        year_from: Start year
        year_to: End year
        court: Optional court filter
        topic: Optional topic filter
        size: Number of results
    """
    filters = [
        {"range": {"year": {"gte": year_from, "lte": year_to}}}
    ]

    if court:
        filters.append({"match": {"court_name": court.lower()}})

    must_clauses = []
    if topic:
        must_clauses.append({
            "match": {
                "keywords": {
                    "query": topic,
                    "boost": 5
                }
            }
        })

    if must_clauses:
        query = {
            "size": min(size, 20),
            "query": {
                "bool": {
                    "must": must_clauses,
                    "filter": filters
                }
            },
            "sort": [{"year": "desc"}]
        }
    else:
        query = {
            "size": min(size, 20),
            "query": {
                "bool": {
                    "must": [{"match_all": {}}],
                    "filter": filters
                }
            },
            "sort": [{"year": "desc"}]
        }

    response = execute_search(query)
    return format_results_to_string(response, max_results=size)


@tool(args_schema=BenchSizeInput)
def search_by_bench_size(
    bench_size: int,
    court: Optional[str] = None,
    topic: Optional[str] = None,
    size: int = 10
) -> str:
    """Search judgments by bench composition size.

    Use when user asks for multi-judge decisions like:
    - 'Constitution bench judgments' (5+ judges)
    - 'Division bench orders' (2 judges)
    - 'Full bench decisions' (3+ judges)
    - '5 judge bench cases'

    Args:
        bench_size: Number of judges on the bench
        court: Optional court filter
        topic: Optional topic filter
        size: Number of results
    """
    filters = [
        {"term": {"bench_size": bench_size}}
    ]

    if court:
        filters.append({"match": {"court_name": court.lower()}})

    must_clauses = []
    if topic:
        must_clauses.append({"match": {"keywords": topic}})

    query = {
        "size": min(size, 20),
        "query": {
            "bool": {
                "must": must_clauses if must_clauses else [{"match_all": {}}],
                "filter": filters
            }
        }
    }

    response = execute_search(query)
    return format_results_to_string(response, max_results=size)
