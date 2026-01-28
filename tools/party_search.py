# tools/party_search.py
"""
Tools for searching legal judgments by party names.
Uses modern LangChain @tool decorator pattern.
"""

from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import Optional

from .base import execute_search, format_results_to_string


# ============================================================
# Schema Definitions
# ============================================================

class PartySearchInput(BaseModel):
    """Input for searching by both party names."""
    petitioner: str = Field(description="Name of the petitioner/appellant (party who filed the case)")
    respondent: str = Field(description="Name of the respondent (party against whom case is filed)")
    year: Optional[int] = Field(default=None, description="Year of judgment (4-digit year, optional)")
    size: int = Field(default=5, description="Number of results to return (1-20)")


class FuzzyNameInput(BaseModel):
    """Input for fuzzy name search."""
    name: str = Field(description="Party name to search (can be misspelled or partial)")
    role: str = Field(
        default="any",
        description="Role of party: 'petitioner', 'respondent', or 'any' (searches both)"
    )
    size: int = Field(default=5, description="Number of results to return")


class SinglePartyInput(BaseModel):
    """Input for single party search."""
    party_name: str = Field(description="Name of the party (searches both petitioner and respondent)")
    court: Optional[str] = Field(default=None, description="Court name filter (e.g., 'supreme court', 'delhi high court')")
    year: Optional[int] = Field(default=None, description="Year filter (4-digit year)")
    size: int = Field(default=10, description="Number of results to return")


# ============================================================
# Tool Definitions
# ============================================================

@tool(args_schema=PartySearchInput)
def search_by_party_names(
    petitioner: str,
    respondent: str,
    year: Optional[int] = None,
    size: int = 5
) -> str:
    """Search legal judgments by petitioner and respondent names.

    Use this tool when the user mentions a specific case with both party names,
    like 'Ram vs Shyam', 'State of Maharashtra vs John Doe', or
    'find the case between ABC Corporation and XYZ Ltd'.

    Args:
        petitioner: Name of the petitioner/appellant
        respondent: Name of the respondent
        year: Optional year of judgment
        size: Number of results to return
    """
    must_clauses = [
        {
            "match": {
                "petitioner_names": {
                    "query": petitioner,
                    "minimum_should_match": "80%",
                    "boost": 10
                }
            }
        },
        {
            "match": {
                "respondent_names": {
                    "query": respondent,
                    "minimum_should_match": "80%",
                    "boost": 10
                }
            }
        }
    ]

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
        }
    }

    response = execute_search(query)
    return format_results_to_string(response, max_results=size)


@tool(args_schema=FuzzyNameInput)
def fuzzy_name_search(
    name: str,
    role: str = "any",
    size: int = 5
) -> str:
    """Search judgments with fuzzy matching for potentially misspelled party names.

    Use this tool when the user might have misspelled a party name, or only knows
    a partial name. Handles typos like 'Rajeev' vs 'Rajiv', 'Sharma' vs 'Sharman'.

    Args:
        name: Party name (may be misspelled or partial)
        role: 'petitioner', 'respondent', or 'any'
        size: Number of results to return
    """
    if role == "petitioner":
        query = {
            "size": min(size, 20),
            "query": {
                "match": {
                    "petitioner_names": {
                        "query": name,
                        "fuzziness": "AUTO",
                        "prefix_length": 2
                    }
                }
            }
        }
    elif role == "respondent":
        query = {
            "size": min(size, 20),
            "query": {
                "match": {
                    "respondent_names": {
                        "query": name,
                        "fuzziness": "AUTO",
                        "prefix_length": 2
                    }
                }
            }
        }
    else:
        query = {
            "size": min(size, 20),
            "query": {
                "bool": {
                    "should": [
                        {
                            "match": {
                                "petitioner_names": {
                                    "query": name,
                                    "fuzziness": "AUTO",
                                    "prefix_length": 2,
                                    "boost": 2
                                }
                            }
                        },
                        {
                            "match": {
                                "respondent_names": {
                                    "query": name,
                                    "fuzziness": "AUTO",
                                    "prefix_length": 2,
                                    "boost": 2
                                }
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            }
        }

    response = execute_search(query)
    return format_results_to_string(response, max_results=size)


@tool(args_schema=SinglePartyInput)
def search_by_single_party(
    party_name: str,
    court: Optional[str] = None,
    year: Optional[int] = None,
    size: int = 10
) -> str:
    """Search all judgments involving a specific party name.

    Searches both petitioner and respondent fields. Use when user asks for
    'all cases involving XYZ', 'cases where ABC was a party', or
    'find judgments with Tata as a party'.

    Args:
        party_name: Name of the party
        court: Optional court filter
        year: Optional year filter
        size: Number of results to return
    """
    should_clauses = [
        {"match": {"petitioner_names": {"query": party_name, "boost": 5}}},
        {"match": {"respondent_names": {"query": party_name, "boost": 5}}}
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
                "should": should_clauses,
                "minimum_should_match": 1,
                "filter": filters
            }
        }
    }

    response = execute_search(query)
    return format_results_to_string(response, max_results=size)
