# tools/legal_reference_search.py
"""
Tools for searching legal judgments by acts, sections, and citations.
Uses modern LangChain @tool decorator pattern.
"""

from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import List, Optional

from .base import execute_search, format_results_to_string


# ============================================================
# Schema Definitions
# ============================================================

class ActSectionInput(BaseModel):
    """Input for act/section search."""
    act_name: str = Field(
        description="Name of the act (e.g., 'IPC', 'CrPC', 'NI Act', 'Income Tax Act', 'Companies Act', 'Constitution')"
    )
    section: str = Field(
        description="Section or Article number (e.g., '420', '138', '302', '21', '226')"
    )
    court: Optional[str] = Field(
        default=None,
        description="Court name filter (e.g., 'supreme court', 'delhi high court')"
    )
    year_from: Optional[int] = Field(default=None, description="Start year filter (e.g., 2020)")
    year_to: Optional[int] = Field(default=None, description="End year filter (e.g., 2024)")
    size: int = Field(default=10, description="Number of results to return")


class CitationInput(BaseModel):
    """Input for citation search."""
    citation: str = Field(
        description="Case citation (e.g., 'AIR 2020 SC 123', '(2019) 5 SCC 456', '2021 SCC OnLine Del 1234')"
    )
    size: int = Field(default=3, description="Number of results to return")


class MultipleSectionsInput(BaseModel):
    """Input for multi-section search."""
    sections: List[str] = Field(
        description="List of sections to search (e.g., ['section 302 ipc', 'section 307 ipc'])"
    )
    topic: Optional[str] = Field(
        default=None,
        description="Legal topic or subject to search along with sections (e.g., 'anticipatory bail', 'quashing', 'discharge')"
    )
    match_all: bool = Field(
        default=False,
        description="If True, judgment must contain ALL sections. If False, any section matches."
    )
    court: Optional[str] = Field(default=None, description="Court name filter")
    year_from: Optional[int] = Field(default=None, description="Start year filter (e.g., 2020)")
    year_to: Optional[int] = Field(default=None, description="End year filter (e.g., 2024)")
    size: int = Field(default=10, description="Number of results to return")


class LegalPrincipleInput(BaseModel):
    """Input for legal principle search."""
    principle: str = Field(
        description="Legal principle or doctrine (e.g., 'res judicata', 'natural justice', 'estoppel', 'basic structure')"
    )
    court: Optional[str] = Field(default=None, description="Court name filter")
    year_from: Optional[int] = Field(default=None, description="Start year filter (e.g., 2020)")
    year_to: Optional[int] = Field(default=None, description="End year filter (e.g., 2024)")
    size: int = Field(default=10, description="Number of results to return")


# ============================================================
# Tool Definitions
# ============================================================

@tool(args_schema=ActSectionInput)
def search_by_act_section(
    act_name: str,
    section: str,
    court: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    size: int = 10
) -> str:
    """Search judgments by specific legal act and section number.

    Use when user asks for cases under specific sections like:
    - 'Section 138 NI Act cases'
    - 'IPC 302 judgments'
    - 'Article 21 Constitution cases'
    - 'Section 482 CrPC orders'
    - 'judgments under Section 420 of IPC'
    - 'Section 420 IPC cases after 2020'

    Args:
        act_name: Name of the act (IPC, CrPC, NI Act, etc.)
        section: Section or article number
        court: Optional court filter
        year_from: Optional start year filter
        year_to: Optional end year filter
        size: Number of results
    """
    search_text = f"section {section} {act_name}"
    search_text_alt = f"{section} {act_name}"

    should_clauses = [
        {
            "match": {
                "acts_or_sections_invoked": {
                    "query": search_text,
                    "boost": 15,
                    "minimum_should_match": "75%"
                }
            }
        },
        {
            "match": {
                "acts_or_sections_invoked": {
                    "query": search_text_alt,
                    "boost": 12
                }
            }
        },
        {
            "match": {
                "page_content": {
                    "query": search_text,
                    "boost": 5
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
                "should": should_clauses,
                "minimum_should_match": 1,
                "filter": filters
            }
        },
        "sort": [{"year": "desc"}]
    }

    response = execute_search(query)
    return format_results_to_string(response, max_results=size)


@tool(args_schema=CitationInput)
def search_by_citation(citation: str, size: int = 3) -> str:
    """Search judgment by its legal citation.

    Use when user provides a specific case citation like:
    - 'AIR 2020 SC 123'
    - '(2019) 5 SCC 456'
    - '2021 SCC OnLine Del 1234'
    - 'MANU/SC/0123/2020'

    Args:
        citation: The case citation
        size: Number of results
    """
    query = {
        "size": min(size, 10),
        "query": {
            "bool": {
                "should": [
                    {
                        "match_phrase": {
                            "citation": {
                                "query": citation,
                                "boost": 25
                            }
                        }
                    },
                    {
                        "match": {
                            "citation": {
                                "query": citation,
                                "boost": 15
                            }
                        }
                    },
                    {
                        "match_phrase": {
                            "page_content": {
                                "query": citation,
                                "boost": 10
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


@tool(args_schema=MultipleSectionsInput)
def search_by_multiple_sections(
    sections: List[str],
    topic: Optional[str] = None,
    match_all: bool = False,
    court: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    size: int = 10
) -> str:
    """Search judgments citing multiple legal sections, optionally with a specific topic.

    Use when user asks for cases involving multiple sections like:
    - 'Cases under Section 302 and 307 IPC'
    - 'Judgments citing both Section 138 NI Act and Section 420 IPC'
    - 'Cases where sections 498A and 304B were invoked'
    - 'Section 467, 468, 471 IPC cases after 2020'
    - 'Anticipatory bail cases under Section 420, 467, 468, 471 IPC'
    - 'Quashing petitions under Section 482 CrPC with Section 420 IPC'

    Args:
        sections: List of sections to search
        topic: Optional topic/subject to search (e.g., 'anticipatory bail', 'quashing', 'discharge')
        match_all: If True, all sections must be present
        court: Optional court filter
        year_from: Optional start year filter
        year_to: Optional end year filter
        size: Number of results
    """
    section_clauses = [
        {
            "match": {
                "acts_or_sections_invoked": {
                    "query": section,
                    "boost": 10
                }
            }
        }
        for section in sections
    ]

    # Add topic search clause if provided
    topic_clause = None
    if topic:
        topic_clause = {
            "bool": {
                "should": [
                    {
                        "match_phrase": {
                            "page_content": {
                                "query": topic,
                                "boost": 15
                            }
                        }
                    },
                    {
                        "match": {
                            "keywords": {
                                "query": topic,
                                "boost": 8
                            }
                        }
                    }
                ],
                "minimum_should_match": 1
            }
        }

    filters = []
    if court:
        filters.append({"match": {"court_name": court.lower()}})
    if year_from and year_to:
        filters.append({"range": {"year": {"gte": year_from, "lte": year_to}}})
    elif year_from:
        filters.append({"range": {"year": {"gte": year_from}}})
    elif year_to:
        filters.append({"range": {"year": {"lte": year_to}}})

    if match_all:
        must_clauses = section_clauses.copy()
        if topic_clause:
            must_clauses.append(topic_clause)
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
        must_clauses = []
        if topic_clause:
            must_clauses.append(topic_clause)
        query = {
            "size": min(size, 20),
            "query": {
                "bool": {
                    "must": must_clauses if must_clauses else None,
                    "should": section_clauses,
                    "minimum_should_match": 1,
                    "filter": filters
                }
            },
            "sort": [{"year": "desc"}]
        }
        # Clean up None values
        if query["query"]["bool"]["must"] is None:
            del query["query"]["bool"]["must"]

    response = execute_search(query)
    return format_results_to_string(response, max_results=size)


@tool(args_schema=LegalPrincipleInput)
def search_by_legal_principle(
    principle: str,
    court: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    size: int = 10
) -> str:
    """Search judgments discussing specific legal principles or doctrines.

    Use when user asks for cases on legal concepts like:
    - 'res judicata'
    - 'principles of natural justice'
    - 'doctrine of estoppel'
    - 'separation of powers'
    - 'basic structure doctrine'
    - 'audi alteram partem'
    - 'recent cases on natural justice after 2020'

    Args:
        principle: Legal principle or doctrine name
        court: Optional court filter
        year_from: Optional start year filter
        year_to: Optional end year filter
        size: Number of results
    """
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
                "must": [
                    {
                        "match_phrase": {
                            "page_content": {
                                "query": principle,
                                "boost": 10
                            }
                        }
                    }
                ],
                "should": [
                    {
                        "match": {
                            "keywords": {
                                "query": principle,
                                "boost": 5
                            }
                        }
                    }
                ],
                "filter": filters
            }
        },
        "sort": [{"year": "desc"}]
    }

    response = execute_search(query)
    return format_results_to_string(response, max_results=size)
