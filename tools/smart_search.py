# tools/smart_search.py
"""
Smart Case Search — exhaustive multi-strategy search tool.

Automatically tries 7+ permutations/combinations of search approaches
to find a case, stopping only when results are found or all strategies
are exhausted.
"""

import re
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, Field
from langchain.tools import tool

from .base import execute_search, format_results_to_string


# ============================================================
# Schema
# ============================================================

class SmartCaseSearchInput(BaseModel):
    """Input for smart case search."""
    query: str = Field(
        description=(
            "Full case reference as provided by the user. "
            "Can be party names, citation, or a combination. "
            "Examples: 'Sona Bala Bora v. Jyotirindra Bhatacharjee, (2005) 4 SCC 501', "
            "'AIR 2020 SC 123', 'Ram Kumar vs State of UP'"
        )
    )
    size: int = Field(default=5, description="Number of results to return")


# ============================================================
# Query Parser
# ============================================================

def _parse_case_query(query: str) -> Dict[str, Any]:
    """
    Parse a case reference query into structured components.

    Extracts: petitioner, respondent, year, citation parts, raw names.
    """
    result = {
        "petitioner": None,
        "respondent": None,
        "year": None,
        "citation_part": None,
        "raw_query": query.strip(),
    }

    text = query.strip()

    # Extract year (4-digit number in parentheses/brackets or standalone)
    year_match = re.search(r'[\(\[]?\s*((?:19|20)\d{2})\s*[\)\]]?', text)
    if year_match:
        result["year"] = int(year_match.group(1))

    # Extract citation part: everything after the last comma before a year,
    # or patterns like "(2005) 4 SCC 501", "AIR 2020 SC 123"
    citation_patterns = [
        r'(\(\d{4}\)\s*\d+\s*\w+\s*\d+)',          # (2005) 4 SCC 501
        r'(\[\d{4}\]\s*\d+\s*\w+\.?\w*\.?\s*\d+)',  # [2005] 3 S.C.R. 454
        r'(AIR\s*\d{4}\s*\w+\s*\d+)',                # AIR 2020 SC 123
        r'(\d{4}\s+SCC\s+OnLine\s+\w+\s+\d+)',      # 2021 SCC OnLine Del 1234
        r'(MANU/\w+/\d+/\d+)',                        # MANU/SC/0123/2020
    ]
    for pattern in citation_patterns:
        cit_match = re.search(pattern, text, re.IGNORECASE)
        if cit_match:
            result["citation_part"] = cit_match.group(1)
            break

    # Extract party names: "Name v. Name" or "Name vs Name" or "Name vs. Name"
    party_pattern = r'^(.+?)\s+(?:v\.?s?\.?|versus)\s+(.+?)(?:\s*[,\(\[\d]|$)'
    party_match = re.match(party_pattern, text, re.IGNORECASE)
    if party_match:
        result["petitioner"] = _clean_party_name(party_match.group(1))
        result["respondent"] = _clean_party_name(party_match.group(2))

    return result


def _clean_party_name(name: str) -> str:
    """Clean up a party name: strip trailing punctuation, extra whitespace."""
    name = name.strip().rstrip(',').rstrip('.').strip()
    # Remove trailing year/citation artifacts
    name = re.sub(r'\s*[\(\[]\s*\d{4}\s*[\)\]]?\s*$', '', name)
    return name.strip()


def _get_name_variations(name: str) -> List[str]:
    """
    Generate variations of a party name for exhaustive searching.

    Examples:
    - "Sona Bala Bora And Ors." → ["Sona Bala Bora And Ors.", "Sona Bala Bora", "Sona Bala"]
    - "State Of Maharashtra" → ["State Of Maharashtra", "Maharashtra", "State"]
    """
    variations = [name]

    # Strip "And Ors", "And Others", "& Ors", "& Anr"
    stripped = re.sub(r'\s+(?:and|&)\s+(?:ors\.?|others?|anr\.?)\s*$', '', name, flags=re.IGNORECASE).strip()
    if stripped != name and stripped:
        variations.append(stripped)

    # For "State of X" → also try just "X" and "State of X"
    state_match = re.match(r'(?:State|Union)\s+(?:of|Of)\s+(.+)', name, re.IGNORECASE)
    if state_match:
        variations.append(state_match.group(1).strip())

    # Try first two words (for long names)
    words = name.split()
    if len(words) > 2:
        variations.append(' '.join(words[:2]))

    # Try just the first word if it's a surname
    if len(words) >= 2 and len(words[0]) > 3:
        variations.append(words[0])

    return variations


# ============================================================
# Search Strategies
# ============================================================

def _strategy_party_both_with_year(parsed: Dict, size: int) -> Tuple[str, Dict]:
    """Strategy 1: Both party names + year boost on citation."""
    if not parsed["petitioner"] or not parsed["respondent"]:
        return "party_both_with_year", {}

    must = [
        {"match": {"petitioner_names": {"query": parsed["petitioner"], "minimum_should_match": "75%", "boost": 10}}},
        {"match": {"respondent_names": {"query": parsed["respondent"], "minimum_should_match": "75%", "boost": 10}}},
    ]
    should = []
    if parsed["year"]:
        for yr in range(parsed["year"] - 2, parsed["year"] + 3):
            should.append({"wildcard": {"citation": f"*{yr}*"}})

    query = {"size": size, "query": {"bool": {"must": must, "should": should}}}
    return "party_both_with_year", query


def _strategy_party_both_no_year(parsed: Dict, size: int) -> Tuple[str, Dict]:
    """Strategy 2: Both party names, no year filter."""
    if not parsed["petitioner"] or not parsed["respondent"]:
        return "party_both_no_year", {}

    must = [
        {"match": {"petitioner_names": {"query": parsed["petitioner"], "minimum_should_match": "75%", "boost": 10}}},
        {"match": {"respondent_names": {"query": parsed["respondent"], "minimum_should_match": "75%", "boost": 10}}},
    ]
    query = {"size": size, "query": {"bool": {"must": must}}}
    return "party_both_no_year", query


def _strategy_party_swapped(parsed: Dict, size: int) -> Tuple[str, Dict]:
    """Strategy 3: Swap petitioner and respondent (sometimes stored reversed)."""
    if not parsed["petitioner"] or not parsed["respondent"]:
        return "party_swapped", {}

    must = [
        {"match": {"petitioner_names": {"query": parsed["respondent"], "minimum_should_match": "75%", "boost": 10}}},
        {"match": {"respondent_names": {"query": parsed["petitioner"], "minimum_should_match": "75%", "boost": 10}}},
    ]
    query = {"size": size, "query": {"bool": {"must": must}}}
    return "party_swapped", query


def _strategy_party_relaxed(parsed: Dict, size: int) -> Tuple[str, Dict]:
    """Strategy 4: Both names as should (not must) — relaxed matching."""
    if not parsed["petitioner"] or not parsed["respondent"]:
        return "party_relaxed", {}

    should = [
        {"match": {"petitioner_names": {"query": parsed["petitioner"], "boost": 10}}},
        {"match": {"respondent_names": {"query": parsed["respondent"], "boost": 10}}},
        {"match": {"petitioner_names": {"query": parsed["respondent"], "boost": 5}}},
        {"match": {"respondent_names": {"query": parsed["petitioner"], "boost": 5}}},
    ]
    query = {"size": size, "query": {"bool": {"should": should, "minimum_should_match": 2}}}
    return "party_relaxed", query


def _strategy_fuzzy_petitioner(parsed: Dict, size: int) -> Tuple[str, Dict]:
    """Strategy 5: Fuzzy search on petitioner name."""
    if not parsed["petitioner"]:
        return "fuzzy_petitioner", {}

    should = [
        {"match": {"petitioner_names": {"query": parsed["petitioner"], "fuzziness": "AUTO", "prefix_length": 2, "boost": 8}}},
        {"match": {"respondent_names": {"query": parsed["petitioner"], "fuzziness": "AUTO", "prefix_length": 2, "boost": 3}}},
    ]
    query = {"size": size, "query": {"bool": {"should": should, "minimum_should_match": 1}}}
    return "fuzzy_petitioner", query


def _strategy_fuzzy_respondent(parsed: Dict, size: int) -> Tuple[str, Dict]:
    """Strategy 6: Fuzzy search on respondent name."""
    if not parsed["respondent"]:
        return "fuzzy_respondent", {}

    should = [
        {"match": {"respondent_names": {"query": parsed["respondent"], "fuzziness": "AUTO", "prefix_length": 2, "boost": 8}}},
        {"match": {"petitioner_names": {"query": parsed["respondent"], "fuzziness": "AUTO", "prefix_length": 2, "boost": 3}}},
    ]
    query = {"size": size, "query": {"bool": {"should": should, "minimum_should_match": 1}}}
    return "fuzzy_respondent", query


def _strategy_name_variations(parsed: Dict, size: int) -> Tuple[str, Dict]:
    """Strategy 7: Try variations of party names (strip 'And Ors', 'State of X', etc.)."""
    if not parsed["petitioner"] and not parsed["respondent"]:
        return "name_variations", {}

    pet_variations = _get_name_variations(parsed["petitioner"]) if parsed["petitioner"] else []
    resp_variations = _get_name_variations(parsed["respondent"]) if parsed["respondent"] else []

    should = []
    for pv in pet_variations[1:]:  # Skip the original (already tried)
        should.append({"match": {"petitioner_names": {"query": pv, "boost": 8, "minimum_should_match": "75%"}}})
    for rv in resp_variations[1:]:
        should.append({"match": {"respondent_names": {"query": rv, "boost": 8, "minimum_should_match": "75%"}}})

    if not should:
        return "name_variations", {}

    query = {"size": size, "query": {"bool": {"should": should, "minimum_should_match": 1}}}
    return "name_variations", query


def _strategy_citation_match(parsed: Dict, size: int) -> Tuple[str, Dict]:
    """Strategy 8: Search by citation text in citation field."""
    if not parsed["citation_part"]:
        return "citation_match", {}

    cleaned = re.sub(r'[\(\)\[\]]', '', parsed["citation_part"])
    should = [
        {"match_phrase": {"citation": {"query": parsed["citation_part"], "boost": 25}}},
        {"match": {"citation": {"query": cleaned, "boost": 15, "minimum_should_match": "50%"}}},
    ]
    query = {"size": size, "query": {"bool": {"should": should, "minimum_should_match": 1}}}
    return "citation_match", query


def _strategy_citation_in_content(parsed: Dict, size: int) -> Tuple[str, Dict]:
    """Strategy 9: Search citation text in page_content."""
    if not parsed["citation_part"]:
        return "citation_in_content", {}

    should = [
        {"match_phrase": {"page_content": {"query": parsed["citation_part"], "boost": 10}}},
    ]
    # Also add party names in content if available
    if parsed["petitioner"]:
        should.append({"match_phrase": {"page_content": {"query": parsed["petitioner"], "boost": 5}}})
    if parsed["respondent"]:
        should.append({"match_phrase": {"page_content": {"query": parsed["respondent"], "boost": 5}}})

    query = {"size": size, "query": {"bool": {"should": should, "minimum_should_match": 1}}}
    return "citation_in_content", query


def _strategy_single_party_petitioner(parsed: Dict, size: int) -> Tuple[str, Dict]:
    """Strategy 10: Search just petitioner name across both fields."""
    if not parsed["petitioner"]:
        return "single_party_petitioner", {}

    should = [
        {"match": {"petitioner_names": {"query": parsed["petitioner"], "boost": 5}}},
        {"match": {"respondent_names": {"query": parsed["petitioner"], "boost": 5}}},
    ]
    query = {"size": size, "query": {"bool": {"should": should, "minimum_should_match": 1}}}
    return "single_party_petitioner", query


def _strategy_single_party_respondent(parsed: Dict, size: int) -> Tuple[str, Dict]:
    """Strategy 11: Search just respondent name across both fields."""
    if not parsed["respondent"]:
        return "single_party_respondent", {}

    should = [
        {"match": {"petitioner_names": {"query": parsed["respondent"], "boost": 5}}},
        {"match": {"respondent_names": {"query": parsed["respondent"], "boost": 5}}},
    ]
    query = {"size": size, "query": {"bool": {"should": should, "minimum_should_match": 1}}}
    return "single_party_respondent", query


def _strategy_fulltext_keyword(parsed: Dict, size: int) -> Tuple[str, Dict]:
    """Strategy 12: Full-text keyword search on entire query."""
    should = [
        {"match": {"page_content": {"query": parsed["raw_query"], "boost": 5}}},
        {"match": {"keywords": {"query": parsed["raw_query"], "boost": 3}}},
    ]
    query = {"size": size, "query": {"bool": {"should": should, "minimum_should_match": 1}}}
    return "fulltext_keyword", query


# All strategies in order of specificity (most specific first)
ALL_STRATEGIES = [
    _strategy_party_both_with_year,     # 1. Both names + year
    _strategy_party_both_no_year,       # 2. Both names, no year
    _strategy_party_swapped,            # 3. Swapped petitioner/respondent
    _strategy_party_relaxed,            # 4. Both names as should (relaxed)
    _strategy_fuzzy_petitioner,         # 5. Fuzzy petitioner
    _strategy_fuzzy_respondent,         # 6. Fuzzy respondent
    _strategy_name_variations,          # 7. Name variations (strip Ors, State of)
    _strategy_citation_match,           # 8. Citation in citation field
    _strategy_citation_in_content,      # 9. Citation in page_content
    _strategy_single_party_petitioner,  # 10. Single party: petitioner
    _strategy_single_party_respondent,  # 11. Single party: respondent
    _strategy_fulltext_keyword,         # 12. Full-text keyword fallback
]


# ============================================================
# Tool Definition
# ============================================================

@tool("smart_case_search", args_schema=SmartCaseSearchInput)
def smart_case_search(query: str, size: int = 5) -> str:
    """Exhaustive multi-strategy case search that tries 7-12 different search approaches.

    Use this as the PRIMARY tool when searching for a specific case by name, citation,
    or any combination. It automatically:
    1. Parses the query to extract party names, year, and citation
    2. Tries 12 different search strategies in order of specificity
    3. Stops at the first strategy that returns results
    4. Reports which strategy succeeded and which were tried

    This tool GUARANTEES an exhaustive search — it tries name swapping, fuzzy matching,
    name variations (strips 'And Ors', 'State of' prefixes), citation matching,
    single-party search, and full-text keyword search.

    Examples:
    - 'Sona Bala Bora v. Jyotirindra Bhatacharjee, (2005) 4 SCC 501'
    - 'AIR 2020 SC 123'
    - 'Ram Kumar vs State of UP'
    - 'Maneka Gandhi vs Union of India'

    Args:
        query: Full case reference (party names, citation, or combination)
        size: Number of results to return
    """
    parsed = _parse_case_query(query)

    # Build the report of what was parsed
    report_lines = [
        "## Smart Case Search",
        f"**Query:** {query}",
        f"**Parsed:** petitioner='{parsed['petitioner']}', respondent='{parsed['respondent']}', "
        f"year={parsed['year']}, citation='{parsed['citation_part']}'",
        "",
        "### Search Strategies Attempted:",
    ]

    strategies_tried = []
    best_result = None

    for strategy_fn in ALL_STRATEGIES:
        strategy_name, es_query = strategy_fn(parsed, min(size, 20))

        # Skip strategies that returned empty query (missing required fields)
        if not es_query:
            report_lines.append(f"  - {strategy_name}: SKIPPED (missing required fields)")
            continue

        try:
            response = execute_search(es_query)
            hit_count = len(response.get("hits", {}).get("hits", []))
            total = response.get("hits", {}).get("total", {})
            total_count = total.get("value", 0) if isinstance(total, dict) else total

            strategies_tried.append(strategy_name)

            if hit_count > 0:
                report_lines.append(f"  - {strategy_name}: FOUND {total_count} results")
                best_result = response
                break
            else:
                report_lines.append(f"  - {strategy_name}: no results")

        except Exception as e:
            report_lines.append(f"  - {strategy_name}: ERROR ({str(e)[:80]})")
            strategies_tried.append(strategy_name)

    report_lines.append("")
    report_lines.append(f"**Total strategies tried:** {len(strategies_tried)}")

    if best_result:
        winning_strategy = strategies_tried[-1]
        report_lines.append(f"**Winning strategy:** {winning_strategy}")
        report_lines.append("")

        # Format the results
        formatted = format_results_to_string(best_result, max_results=size)
        report_lines.append(formatted)
    else:
        report_lines.append("")
        report_lines.append(
            "**No results found after exhaustive search.** "
            "The case may not exist in the database, or the name/citation may be "
            "significantly different from what is indexed. "
            "Try providing more specific details or alternative spellings."
        )

    return "\n".join(report_lines)