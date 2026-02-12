# tools/__init__.py
"""
Legal Search Tools Package

A collection of 25 specialized tools for searching legal judgments in Elasticsearch.
Designed for use with LangChain agents using the modern @tool decorator pattern.

Usage:
    from tools import get_all_tools

    tools = get_all_tools()
    # Use tools with your LangChain agent
"""

from typing import List

# Import base utilities
from .base import (
    get_es_client,
    execute_search,
    format_results,
    format_results_to_string,
    generate_s3_link,
    ES_URL,
    ES_INDEX,
    ES_TIMEOUT,
)

# Import party search tools
from .party_search import (
    search_by_party_names,
    fuzzy_name_search,
    search_by_single_party,
)

# Import legal reference search tools
from .legal_reference_search import (
    search_by_act_section,
    search_by_citation,
    search_by_multiple_sections,
    search_by_legal_principle,
)

# Import court search tools
from .court_search import (
    search_by_court,
    search_by_judge,
    search_by_date_range,
    search_by_bench_size,
)

# Import content search tools
from .content_search import (
    search_by_legal_topic,
    search_by_keywords,
    advanced_boolean_search,
    search_by_case_type,
)

# Import advanced search tools
from .advanced_search import (
    search_similar_cases,
    hybrid_search,
    aggregation_search,
    search_landmark_cases,
    search_by_case_status,
)

# Import specialized search tools
from .specialized_search import (
    search_bail_cases,
    search_quashing_cases,
    search_writ_petitions,
    search_criminal_appeals,
)

# Import smart search tool
from .smart_search import smart_case_search

# Note: Agent and pipeline classes should be imported directly:
# from tools.agent import LegalResearchAgent, create_legal_agent, run_query
# from tools.response_agent import ResponseAgent, LegalResearchPipeline, generate_legal_response


def get_all_tools() -> List:
    """
    Get all available legal search tools.

    Returns:
        List of all tool instances ready for use with LangChain agents.
    """
    return [
        # Party Search Tools (3)
        search_by_party_names,
        fuzzy_name_search,
        search_by_single_party,

        # Legal Reference Tools (4)
        search_by_act_section,
        search_by_citation,
        search_by_multiple_sections,
        search_by_legal_principle,

        # Court Search Tools (4)
        search_by_court,
        search_by_judge,
        search_by_date_range,
        search_by_bench_size,

        # Content Search Tools (4)
        search_by_legal_topic,
        search_by_keywords,
        advanced_boolean_search,
        search_by_case_type,

        # Advanced Search Tools (5)
        search_similar_cases,
        hybrid_search,
        aggregation_search,
        search_landmark_cases,
        search_by_case_status,

        # Specialized Search Tools (4)
        search_bail_cases,
        search_quashing_cases,
        search_writ_petitions,
        search_criminal_appeals,

        # Smart Search (1)
        smart_case_search,
    ]


def get_tools_by_category() -> dict:
    """
    Get tools organized by category.

    Returns:
        Dictionary with category names as keys and tool lists as values.
    """
    return {
        "party_search": [
            search_by_party_names,
            fuzzy_name_search,
            search_by_single_party,
        ],
        "legal_reference": [
            search_by_act_section,
            search_by_citation,
            search_by_multiple_sections,
            search_by_legal_principle,
        ],
        "court_search": [
            search_by_court,
            search_by_judge,
            search_by_date_range,
            search_by_bench_size,
        ],
        "content_search": [
            search_by_legal_topic,
            search_by_keywords,
            advanced_boolean_search,
            search_by_case_type,
        ],
        "advanced_search": [
            search_similar_cases,
            hybrid_search,
            aggregation_search,
            search_landmark_cases,
            search_by_case_status,
        ],
        "specialized_search": [
            search_bail_cases,
            search_quashing_cases,
            search_writ_petitions,
            search_criminal_appeals,
        ],
        "smart_search": [
            smart_case_search,
        ],
    }


def get_tool_descriptions() -> str:
    """
    Get formatted descriptions of all tools for agent prompts.

    Returns:
        Formatted string with all tool names and descriptions.
    """
    tools = get_all_tools()
    descriptions = []

    for i, tool in enumerate(tools, 1):
        desc = tool.description.split('\n')[0] if tool.description else "No description"
        descriptions.append(f"{i}. {tool.name}: {desc}")

    return "\n".join(descriptions)


# Export all
__all__ = [
    # Base utilities
    "get_es_client",
    "execute_search",
    "format_results",
    "format_results_to_string",
    "generate_s3_link",
    "ES_URL",
    "ES_INDEX",
    "ES_TIMEOUT",

    # Party search tools
    "search_by_party_names",
    "fuzzy_name_search",
    "search_by_single_party",

    # Legal reference tools
    "search_by_act_section",
    "search_by_citation",
    "search_by_multiple_sections",
    "search_by_legal_principle",

    # Court search tools
    "search_by_court",
    "search_by_judge",
    "search_by_date_range",
    "search_by_bench_size",

    # Content search tools
    "search_by_legal_topic",
    "search_by_keywords",
    "advanced_boolean_search",
    "search_by_case_type",

    # Advanced search tools
    "search_similar_cases",
    "hybrid_search",
    "aggregation_search",
    "search_landmark_cases",
    "search_by_case_status",

    # Specialized search tools
    "search_bail_cases",
    "search_quashing_cases",
    "search_writ_petitions",
    "search_criminal_appeals",

    # Smart search
    "smart_case_search",

    # Helper functions
    "get_all_tools",
    "get_tools_by_category",
    "get_tool_descriptions",

    # Agent and pipeline classes (import directly from tools.agent and tools.response_agent)
    # "LegalResearchAgent",
    # "create_legal_agent",
    # "run_query",
    # "ResponseAgent",
    # "LegalResearchPipeline",
    # "generate_legal_response",
]
