# tools/advanced_search.py
"""
Advanced search tools including similarity search, hybrid search, and aggregations.
Uses modern LangChain @tool decorator pattern.
"""

from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .base import execute_search, format_results_to_string


# ============================================================
# Schema Definitions
# ============================================================

class SimilarCaseInput(BaseModel):
    """Input for similar case search."""
    case_text: str = Field(
        description="Text from a known case or description to find similar judgments"
    )
    court: Optional[str] = Field(default=None, description="Court filter")
    size: int = Field(default=5, description="Number of similar cases to return")


class HybridSearchInput(BaseModel):
    """Input for hybrid search."""
    query: str = Field(description="Natural language query about legal cases")
    size: int = Field(default=10, description="Number of results to return")


class AggregationInput(BaseModel):
    """Input for aggregation search."""
    topic: Optional[str] = Field(default=None, description="Topic to analyze")
    section: Optional[str] = Field(default=None, description="Section to analyze")
    group_by: str = Field(
        default="court",
        description="Field to group by: 'court', 'year', or 'judge'"
    )
    size: int = Field(default=10, description="Number of buckets to return")


class LandmarkCaseInput(BaseModel):
    """Input for landmark case search."""
    topic: str = Field(description="Legal topic or area (e.g., 'right to privacy', 'free speech')")
    court: Optional[str] = Field(
        default="supreme court",
        description="Court (defaults to Supreme Court)"
    )
    size: int = Field(default=5, description="Number of results")


class CaseStatusInput(BaseModel):
    """Input for case status search."""
    case_name: Optional[str] = Field(default=None, description="Case name to check status")
    citation: Optional[str] = Field(default=None, description="Citation to check status")
    status: str = Field(
        default="overruled",
        description="Status to search: 'overruled', 'distinguished', 'followed', 'approved'"
    )
    size: int = Field(default=10, description="Number of results")


class CaseMetadata(BaseModel):
    """Structured metadata extracted from query for hybrid search."""
    court_name: Optional[str] = Field(default=None, description="Court name")
    petitioner_names: Optional[List[str]] = Field(default_factory=list)
    respondent_names: Optional[List[str]] = Field(default_factory=list)
    year: Optional[int] = Field(default=None)
    topics: Optional[List[str]] = Field(default_factory=list)
    acts_or_sections: Optional[List[str]] = Field(default_factory=list)
    lexical_query: Optional[str] = Field(default=None)
    size: Optional[int] = Field(default=5)


# ============================================================
# Tool Definitions
# ============================================================

@tool(args_schema=SimilarCaseInput)
def search_similar_cases(
    case_text: str,
    court: Optional[str] = None,
    size: int = 5
) -> str:
    """Find cases similar to a given case text or description.

    Use when user asks for:
    - 'Similar cases to this judgment'
    - 'Related judgments'
    - 'Cases with similar facts'
    - 'Precedents like this case'
    - 'Find cases similar to [case description]'

    Args:
        case_text: Text from a case or description
        court: Optional court filter
        size: Number of similar cases
    """
    filters = []
    if court:
        filters.append({"match": {"court_name": court.lower()}})

    query = {
        "size": min(size, 20),
        "query": {
            "bool": {
                "must": [
                    {
                        "more_like_this": {
                            "fields": ["page_content", "keywords", "acts_or_sections_invoked"],
                            "like": case_text,
                            "min_term_freq": 1,
                            "min_doc_freq": 1,
                            "max_query_terms": 30,
                            "minimum_should_match": "30%"
                        }
                    }
                ],
                "filter": filters
            }
        }
    }

    response = execute_search(query)
    return format_results_to_string(response, max_results=size)


@tool(args_schema=HybridSearchInput)
def hybrid_search(query: str, size: int = 10) -> str:
    """Hybrid search combining LLM query parsing with Elasticsearch BM25.

    Use as a fallback for complex natural language queries that don't fit other tools.
    Automatically extracts entities, topics, and sections from the query.

    Examples:
    - 'Find recent Supreme Court cases about digital privacy and Aadhaar'
    - 'Judgments where bail was denied in murder cases'
    - 'Cases discussing the interpretation of Section 498A'

    Args:
        query: Natural language query
        size: Number of results
    """
    # Parse query with LLM
    metadata = _parse_query_with_llm(query)

    # Build Elasticsearch query from metadata
    should_clauses = []
    filters = []

    # Party names
    if metadata.petitioner_names and metadata.respondent_names:
        for pet in metadata.petitioner_names:
            for resp in metadata.respondent_names:
                should_clauses.append({
                    "bool": {
                        "must": [
                            {"match": {"petitioner_names": {"query": pet, "boost": 20}}},
                            {"match": {"respondent_names": {"query": resp, "boost": 20}}}
                        ]
                    }
                })

    # Topics
    for topic in (metadata.topics or []):
        should_clauses.append({
            "match": {"keywords": {"query": topic, "boost": 15}}
        })

    # Acts/Sections
    for section in (metadata.acts_or_sections or []):
        should_clauses.append({
            "match": {"acts_or_sections_invoked": {"query": section, "boost": 15}}
        })

    # Lexical query
    if metadata.lexical_query:
        should_clauses.append({
            "match": {"page_content": {"query": metadata.lexical_query, "boost": 8}}
        })

    # Fallback to original query
    if not should_clauses:
        should_clauses.append({
            "match": {"page_content": {"query": query, "boost": 5}}
        })

    # Filters
    if metadata.year:
        filters.append({"term": {"year": metadata.year}})
    if metadata.court_name:
        filters.append({"match": {"court_name": metadata.court_name.lower()}})

    es_query = {
        "size": min(metadata.size or size, 20),
        "query": {
            "bool": {
                "should": should_clauses,
                "filter": filters,
                "minimum_should_match": 1
            }
        }
    }

    response = execute_search(es_query)
    return format_results_to_string(response, max_results=size)


def _parse_query_with_llm(query: str) -> CaseMetadata:
    """Use LLM to extract structured metadata from query."""
    prompt_template = """Extract legal search metadata from this query. Return structured data.

Query: {query}

Extract:
- court_name: Court mentioned (e.g., 'supreme court', 'delhi high court')
- petitioner_names: List of petitioner names if mentioned
- respondent_names: List of respondent names if mentioned
- year: Year mentioned (4 digits) or null
- topics: Legal topics (e.g., 'murder', 'bail', 'divorce')
- acts_or_sections: Legal provisions (e.g., 'section 302 ipc', 'article 21')
- lexical_query: Normalized search string
- size: Number of results (1 for specific case, 5 for sections, 10 for topics)"""

    try:
        llm = ChatOpenAI(temperature=0, model="gpt-4o-mini").with_structured_output(CaseMetadata)
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm
        return chain.invoke({"query": query})
    except Exception:
        return CaseMetadata(lexical_query=query, size=10)


@tool(args_schema=AggregationInput)
def aggregation_search(
    topic: Optional[str] = None,
    section: Optional[str] = None,
    group_by: str = "court",
    size: int = 10
) -> str:
    """Get statistics and breakdowns of judgments.

    Use when user asks for:
    - 'How many bail cases per court?'
    - 'Year-wise distribution of Section 138 cases'
    - 'Which courts handled most murder cases?'
    - 'Statistics on cheque bounce cases'

    Args:
        topic: Optional topic to filter
        section: Optional section to filter
        group_by: Field to group by (court, year, judge)
        size: Number of buckets
    """
    filters = []
    if topic:
        filters.append({"match": {"keywords": topic}})
    if section:
        filters.append({"match": {"acts_or_sections_invoked": section}})

    field_map = {
        "court": "court_name.keyword",
        "year": "year",
        "judge": "judge_name.keyword"
    }
    agg_field = field_map.get(group_by, "court_name.keyword")

    query = {
        "size": 0,
        "query": {
            "bool": {
                "filter": filters if filters else [{"match_all": {}}]
            }
        },
        "aggs": {
            "group_agg": {
                "terms": {
                    "field": agg_field,
                    "size": min(size, 50)
                }
            }
        }
    }

    response = execute_search(query)

    buckets = response.get("aggregations", {}).get("group_agg", {}).get("buckets", [])
    total = response.get("hits", {}).get("total", {})
    total_count = total.get("value", 0) if isinstance(total, dict) else total

    result = f"**Statistics Summary**\n\nTotal matching judgments: {total_count}\n\nBreakdown by {group_by}:\n"
    for bucket in buckets:
        result += f"- {bucket['key']}: {bucket['doc_count']} cases\n"

    return result


@tool(args_schema=LandmarkCaseInput)
def search_landmark_cases(
    topic: str,
    court: Optional[str] = "supreme court",
    size: int = 5
) -> str:
    """Search for landmark/important cases on a legal topic.

    Use when user asks for:
    - 'Landmark cases on right to privacy'
    - 'Important judgments on Article 21'
    - 'Leading cases on bail'
    - 'Famous constitutional law cases'

    Prioritizes Constitution bench and larger bench decisions.

    Args:
        topic: Legal topic
        court: Court (defaults to Supreme Court)
        size: Number of results
    """
    query = {
        "size": min(size, 20),
        "query": {
            "bool": {
                "must": [
                    {
                        "match": {
                            "keywords": {
                                "query": topic,
                                "boost": 10
                            }
                        }
                    }
                ],
                "should": [
                    {"range": {"bench_size": {"gte": 5, "boost": 20}}},
                    {"range": {"bench_size": {"gte": 3, "boost": 10}}},
                    {"range": {"citation_count": {"gte": 10, "boost": 15}}}
                ],
                "filter": [
                    {"match": {"court_name": court.lower()}} if court else {"match_all": {}}
                ]
            }
        },
        "sort": [
            {"_score": "desc"},
            {"year": "desc"}
        ]
    }

    response = execute_search(query)
    return format_results_to_string(response, max_results=size)


@tool(args_schema=CaseStatusInput)
def search_by_case_status(
    case_name: Optional[str] = None,
    citation: Optional[str] = None,
    status: str = "overruled",
    size: int = 10
) -> str:
    """Search cases that have been overruled, distinguished, or followed.

    Use when user asks for:
    - 'Cases that overruled XYZ judgment'
    - 'Is this case still good law?'
    - 'Cases that followed ABC vs State'
    - 'Has this judgment been distinguished?'

    Args:
        case_name: Case name to check
        citation: Citation to check
        status: Status type (overruled, distinguished, followed, approved)
        size: Number of results
    """
    must_clauses = []

    must_clauses.append({
        "match": {
            "page_content": {
                "query": status,
                "boost": 5
            }
        }
    })

    if case_name:
        must_clauses.append({
            "match_phrase": {
                "page_content": {
                    "query": case_name,
                    "boost": 10
                }
            }
        })

    if citation:
        must_clauses.append({
            "match_phrase": {
                "page_content": {
                    "query": citation,
                    "boost": 10
                }
            }
        })

    query = {
        "size": min(size, 20),
        "query": {
            "bool": {
                "must": must_clauses
            }
        }
    }

    response = execute_search(query)
    return format_results_to_string(response, max_results=size)
