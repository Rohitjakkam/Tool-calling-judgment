# tools/specialized_search.py
"""
Specialized search tools for common legal topics like bail, quashing, etc.
Uses modern LangChain @tool decorator pattern.
"""

from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import List, Optional

from .base import execute_search, format_results_to_string


# ============================================================
# Schema Definitions
# ============================================================

class BailSearchInput(BaseModel):
    """Input for bail-related case search."""
    bail_type: str = Field(
        description="Type of bail: 'anticipatory', 'regular', 'interim', 'default' or 'all'"
    )
    sections: Optional[List[str]] = Field(
        default=None,
        description="List of IPC/CrPC sections (e.g., ['420 IPC', '467 IPC', '468 IPC'])"
    )
    court: Optional[str] = Field(
        default=None,
        description="Court name filter (e.g., 'supreme court', 'bombay high court', 'delhi high court')"
    )
    year_from: Optional[int] = Field(default=None, description="Start year filter (e.g., 2020)")
    year_to: Optional[int] = Field(default=None, description="End year filter (e.g., 2024)")
    size: int = Field(default=10, description="Number of results to return")


class QuashingSearchInput(BaseModel):
    """Input for quashing petition search."""
    sections: Optional[List[str]] = Field(
        default=None,
        description="List of sections involved (e.g., ['482 CrPC', '420 IPC'])"
    )
    ground: Optional[str] = Field(
        default=None,
        description="Ground for quashing (e.g., 'compromise', 'no prima facie case', 'abuse of process')"
    )
    court: Optional[str] = Field(default=None, description="Court name filter")
    year_from: Optional[int] = Field(default=None, description="Start year filter")
    year_to: Optional[int] = Field(default=None, description="End year filter")
    size: int = Field(default=10, description="Number of results to return")


class WritPetitionSearchInput(BaseModel):
    """Input for writ petition search."""
    writ_type: Optional[str] = Field(
        default=None,
        description="Type of writ: 'habeas corpus', 'mandamus', 'certiorari', 'prohibition', 'quo warranto'"
    )
    subject: Optional[str] = Field(
        default=None,
        description="Subject matter (e.g., 'service matter', 'fundamental rights', 'illegal detention')"
    )
    article: Optional[str] = Field(
        default=None,
        description="Constitutional article (e.g., '226', '32', '21')"
    )
    court: Optional[str] = Field(default=None, description="Court name filter")
    year_from: Optional[int] = Field(default=None, description="Start year filter")
    year_to: Optional[int] = Field(default=None, description="End year filter")
    size: int = Field(default=10, description="Number of results to return")


class CriminalAppealSearchInput(BaseModel):
    """Input for criminal appeal search."""
    offence_type: Optional[str] = Field(
        default=None,
        description="Type of offence (e.g., 'murder', 'cheating', 'forgery', 'dowry death')"
    )
    appeal_ground: Optional[str] = Field(
        default=None,
        description="Ground of appeal (e.g., 'acquittal', 'conviction', 'sentence reduction')"
    )
    sections: Optional[List[str]] = Field(
        default=None,
        description="Relevant sections (e.g., ['302 IPC', '304B IPC'])"
    )
    court: Optional[str] = Field(default=None, description="Court name filter")
    year_from: Optional[int] = Field(default=None, description="Start year filter")
    year_to: Optional[int] = Field(default=None, description="End year filter")
    size: int = Field(default=10, description="Number of results to return")


# ============================================================
# Helper Functions
# ============================================================

def build_year_clauses(year_from: Optional[int], year_to: Optional[int]) -> List[dict]:
    """Build year search clauses for citation field."""
    year_clauses = []
    if year_from and year_to:
        for yr in range(year_from, min(year_to + 1, year_from + 10)):
            year_clauses.append({"wildcard": {"citation": f"*{yr}*"}})
    elif year_from:
        for yr in range(year_from, min(year_from + 10, 2030)):
            year_clauses.append({"wildcard": {"citation": f"*{yr}*"}})
    elif year_to:
        for yr in range(max(year_to - 9, 1990), year_to + 1):
            year_clauses.append({"wildcard": {"citation": f"*{yr}*"}})
    return year_clauses


# ============================================================
# Tool Definitions
# ============================================================

@tool(args_schema=BailSearchInput)
def search_bail_cases(
    bail_type: str = "all",
    sections: Optional[List[str]] = None,
    court: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    size: int = 10
) -> str:
    """Search for bail-related judgments (anticipatory bail, regular bail, interim bail).

    Use when user asks for bail cases like:
    - 'Anticipatory bail cases under Section 420 IPC'
    - 'Regular bail judgments for cheating cases'
    - 'Bail cases in Supreme Court after 2020'
    - 'Anticipatory bail under Section 467, 468, 471 IPC from Bombay High Court'

    Args:
        bail_type: Type of bail ('anticipatory', 'regular', 'interim', 'default', 'all')
        sections: Optional list of relevant sections
        court: Optional court filter
        year_from: Optional start year filter
        year_to: Optional end year filter
        size: Number of results
    """
    # Build bail type search terms
    bail_terms = {
        "anticipatory": ["anticipatory bail", "section 438", "pre-arrest bail"],
        "regular": ["regular bail", "section 439", "bail application"],
        "interim": ["interim bail", "temporary bail"],
        "default": ["default bail", "statutory bail", "section 167"],
        "all": ["bail", "anticipatory bail", "regular bail", "interim bail"]
    }

    bail_keywords = bail_terms.get(bail_type.lower(), bail_terms["all"])

    # Build must clauses for bail
    bail_should_clauses = []
    for term in bail_keywords:
        bail_should_clauses.append({
            "match_phrase": {
                "page_content": {
                    "query": term,
                    "boost": 15
                }
            }
        })
        bail_should_clauses.append({
            "match": {
                "keywords": {
                    "query": term,
                    "boost": 10
                }
            }
        })

    must_clauses = [
        {"bool": {"should": bail_should_clauses, "minimum_should_match": 1}}
    ]

    # Add sections if provided
    if sections:
        section_clauses = []
        for section in sections:
            section_clauses.append({
                "match": {
                    "acts_or_sections_invoked": {
                        "query": section,
                        "boost": 12
                    }
                }
            })
            section_clauses.append({
                "match": {
                    "page_content": {
                        "query": f"section {section}",
                        "boost": 5
                    }
                }
            })
        must_clauses.append({"bool": {"should": section_clauses, "minimum_should_match": 1}})

    # Add year filter
    year_clauses = build_year_clauses(year_from, year_to)
    if year_clauses:
        must_clauses.append({"bool": {"should": year_clauses, "minimum_should_match": 1}})

    # Build filters
    filters = []
    if court:
        filters.append({"match": {"court_name": court.lower()}})

    query = {
        "size": min(size, 20),
        "query": {
            "bool": {
                "must": must_clauses,
                "filter": filters if filters else None
            }
        },
        "sort": [{"_score": "desc"}]
    }

    # Clean up None values
    if query["query"]["bool"]["filter"] is None:
        del query["query"]["bool"]["filter"]

    response = execute_search(query)
    return format_results_to_string(response, max_results=size)


@tool(args_schema=QuashingSearchInput)
def search_quashing_cases(
    sections: Optional[List[str]] = None,
    ground: Optional[str] = None,
    court: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    size: int = 10
) -> str:
    """Search for quashing petition judgments under Section 482 CrPC.

    Use when user asks for quashing cases like:
    - 'Quashing of FIR under Section 482 CrPC'
    - 'Quashing cases where parties compromised'
    - 'Section 482 CrPC quashing of Section 420 IPC cases'

    Args:
        sections: Optional list of sections involved
        ground: Optional ground for quashing
        court: Optional court filter
        year_from: Optional start year filter
        year_to: Optional end year filter
        size: Number of results
    """
    must_clauses = [
        {
            "bool": {
                "should": [
                    {"match_phrase": {"page_content": {"query": "quashing", "boost": 15}}},
                    {"match_phrase": {"page_content": {"query": "section 482", "boost": 12}}},
                    {"match": {"acts_or_sections_invoked": {"query": "482 CrPC", "boost": 15}}},
                    {"match": {"keywords": {"query": "quashing", "boost": 10}}}
                ],
                "minimum_should_match": 1
            }
        }
    ]

    # Add sections if provided
    if sections:
        section_clauses = []
        for section in sections:
            section_clauses.append({
                "match": {"acts_or_sections_invoked": {"query": section, "boost": 10}}
            })
        must_clauses.append({"bool": {"should": section_clauses, "minimum_should_match": 1}})

    # Add ground if provided
    if ground:
        must_clauses.append({
            "match_phrase": {"page_content": {"query": ground, "boost": 8}}
        })

    # Add year filter
    year_clauses = build_year_clauses(year_from, year_to)
    if year_clauses:
        must_clauses.append({"bool": {"should": year_clauses, "minimum_should_match": 1}})

    filters = []
    if court:
        filters.append({"match": {"court_name": court.lower()}})

    query = {
        "size": min(size, 20),
        "query": {
            "bool": {
                "must": must_clauses,
                "filter": filters if filters else None
            }
        },
        "sort": [{"_score": "desc"}]
    }

    if query["query"]["bool"]["filter"] is None:
        del query["query"]["bool"]["filter"]

    response = execute_search(query)
    return format_results_to_string(response, max_results=size)


@tool(args_schema=WritPetitionSearchInput)
def search_writ_petitions(
    writ_type: Optional[str] = None,
    subject: Optional[str] = None,
    article: Optional[str] = None,
    court: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    size: int = 10
) -> str:
    """Search for writ petition judgments.

    Use when user asks for writ cases like:
    - 'Habeas corpus petitions'
    - 'Article 226 writ petitions in Delhi High Court'
    - 'Mandamus cases for service matters'
    - 'Article 32 petitions in Supreme Court'

    Args:
        writ_type: Type of writ (habeas corpus, mandamus, certiorari, prohibition, quo warranto)
        subject: Subject matter of the writ
        article: Constitutional article (226, 32)
        court: Optional court filter
        year_from: Optional start year filter
        year_to: Optional end year filter
        size: Number of results
    """
    must_clauses = [
        {
            "bool": {
                "should": [
                    {"match_phrase": {"page_content": {"query": "writ petition", "boost": 10}}},
                    {"match": {"keywords": {"query": "writ", "boost": 8}}}
                ],
                "minimum_should_match": 1
            }
        }
    ]

    if writ_type:
        must_clauses.append({
            "match_phrase": {"page_content": {"query": writ_type, "boost": 12}}
        })

    if subject:
        must_clauses.append({
            "match_phrase": {"page_content": {"query": subject, "boost": 8}}
        })

    if article:
        must_clauses.append({
            "bool": {
                "should": [
                    {"match": {"acts_or_sections_invoked": {"query": f"article {article}", "boost": 12}}},
                    {"match_phrase": {"page_content": {"query": f"article {article}", "boost": 10}}}
                ],
                "minimum_should_match": 1
            }
        })

    year_clauses = build_year_clauses(year_from, year_to)
    if year_clauses:
        must_clauses.append({"bool": {"should": year_clauses, "minimum_should_match": 1}})

    filters = []
    if court:
        filters.append({"match": {"court_name": court.lower()}})

    query = {
        "size": min(size, 20),
        "query": {
            "bool": {
                "must": must_clauses,
                "filter": filters if filters else None
            }
        },
        "sort": [{"_score": "desc"}]
    }

    if query["query"]["bool"]["filter"] is None:
        del query["query"]["bool"]["filter"]

    response = execute_search(query)
    return format_results_to_string(response, max_results=size)


@tool(args_schema=CriminalAppealSearchInput)
def search_criminal_appeals(
    offence_type: Optional[str] = None,
    appeal_ground: Optional[str] = None,
    sections: Optional[List[str]] = None,
    court: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    size: int = 10
) -> str:
    """Search for criminal appeal judgments.

    Use when user asks for criminal appeal cases like:
    - 'Murder conviction appeals in Supreme Court'
    - 'Appeals against acquittal under Section 302 IPC'
    - 'Sentence reduction appeals in cheating cases'

    Args:
        offence_type: Type of offence (murder, cheating, forgery, etc.)
        appeal_ground: Ground of appeal (acquittal, conviction, sentence reduction)
        sections: Relevant IPC sections
        court: Optional court filter
        year_from: Optional start year filter
        year_to: Optional end year filter
        size: Number of results
    """
    must_clauses = [
        {
            "bool": {
                "should": [
                    {"match_phrase": {"page_content": {"query": "criminal appeal", "boost": 12}}},
                    {"match": {"keywords": {"query": "criminal appeal", "boost": 10}}},
                    {"match_phrase": {"page_content": {"query": "appeal against conviction", "boost": 8}}},
                    {"match_phrase": {"page_content": {"query": "appeal against acquittal", "boost": 8}}}
                ],
                "minimum_should_match": 1
            }
        }
    ]

    if offence_type:
        must_clauses.append({
            "match_phrase": {"page_content": {"query": offence_type, "boost": 10}}
        })

    if appeal_ground:
        must_clauses.append({
            "match_phrase": {"page_content": {"query": appeal_ground, "boost": 8}}
        })

    if sections:
        section_clauses = []
        for section in sections:
            section_clauses.append({
                "match": {"acts_or_sections_invoked": {"query": section, "boost": 10}}
            })
        must_clauses.append({"bool": {"should": section_clauses, "minimum_should_match": 1}})

    year_clauses = build_year_clauses(year_from, year_to)
    if year_clauses:
        must_clauses.append({"bool": {"should": year_clauses, "minimum_should_match": 1}})

    filters = []
    if court:
        filters.append({"match": {"court_name": court.lower()}})

    query = {
        "size": min(size, 20),
        "query": {
            "bool": {
                "must": must_clauses,
                "filter": filters if filters else None
            }
        },
        "sort": [{"_score": "desc"}]
    }

    if query["query"]["bool"]["filter"] is None:
        del query["query"]["bool"]["filter"]

    response = execute_search(query)
    return format_results_to_string(response, max_results=size)
