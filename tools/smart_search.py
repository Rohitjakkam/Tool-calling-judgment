# tools/smart_search.py
"""
Smart Case Search — exhaustive multi-strategy search tool.

Automatically tries 7+ permutations/combinations of search approaches
to find a case, stopping only when results are found or all strategies
are exhausted.
"""

import re
import time
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, Field
from langchain.tools import tool

from .base import execute_search, format_results_to_string, setup_logger

logger = setup_logger(__name__)


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

    logger.info("Parsed query: petitioner='%s', respondent='%s', year=%s, citation='%s'",
                result["petitioner"], result["respondent"], result["year"], result["citation_part"])
    return result


def _clean_party_name(name: str) -> str:
    """Clean up a party name: strip trailing punctuation, extra whitespace, and court names."""
    name = name.strip().rstrip(',').rstrip('.').strip()
    # Remove trailing year/citation artifacts
    name = re.sub(r'\s*[\(\[]\s*\d{4}\s*[\)\]]?\s*$', '', name)
    # Remove court name qualifiers that get accidentally captured as part of party names
    # E.g., "Government Of India Supreme Court" → "Government Of India"
    # (longer patterns first so they match before shorter ones)
    name = re.sub(
        r'\s+(?:Supreme\s+Court(?:\s+of\s+India)?'
        r'|High\s+Court(?:\s+of\s+\w+)?'
        r'|(?:from|in|at|before)\s+(?:Supreme|High|District)\s+Court'
        r'|(?:Supreme|High|District|Sessions|Family|Consumer)\s+Court)',
        '', name, flags=re.IGNORECASE
    ).strip()
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


# ============================================================
# Year Validation Helper
# ============================================================

def _check_year_in_results(response: dict, requested_year: int, tolerance: int = 1) -> bool:
    """
    Check if any search result matches the requested year.

    Checks both the citation field and page_content (first 500 chars)
    for the year string, since the ES 'year' field is often corrupted.

    Args:
        response: ES search response
        requested_year: The year the user asked for
        tolerance: Accept years within ± this range (default ±1)

    Returns:
        True if at least one result matches the year
    """
    acceptable_years = set(
        str(yr) for yr in range(requested_year - tolerance, requested_year + tolerance + 1)
    )
    logger.debug("Year validation: checking for years %s in %d results", acceptable_years, len(response.get("hits", {}).get("hits", [])))

    for hit in response.get("hits", {}).get("hits", []):
        source = hit.get("_source", {})

        # Check citation field
        citation = str(source.get("citation", ""))
        for yr_str in acceptable_years:
            if yr_str in citation:
                logger.debug("  Year %s MATCHED in citation: '%s'", yr_str, citation[:80])
                return True

        # Check beginning of page_content
        content = str(source.get("page_content", ""))[:500]
        for yr_str in acceptable_years:
            if yr_str in content:
                logger.debug("  Year %s MATCHED in page_content (first 500 chars)", yr_str)
                return True

        logger.debug("  No year match in hit: citation='%s'", citation[:80])

    logger.info("Year validation FAILED: none of %d results contain year %d (±%d)", len(response.get("hits", {}).get("hits", [])), requested_year, tolerance)
    return False


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
    """Exhaustive multi-strategy case search that tries up to 13 different search approaches.

    Use this as the PRIMARY tool when searching for a specific case by name, citation,
    or any combination. It automatically:
    1. Parses the query to extract party names, year, and citation
    2. Tries 12 different database strategies in order of specificity
    3. If ALL database strategies fail, uses Strategy 13: scrapes indiankanoon.org as internet fallback
    4. Stops at the first strategy that returns results
    5. Reports which strategy succeeded and which were tried

    This tool GUARANTEES an exhaustive search — it tries name swapping, fuzzy matching,
    name variations (strips 'And Ors', 'State of' prefixes), citation matching,
    single-party search, full-text keyword search, AND live internet scraping as final fallback.

    Examples:
    - 'Sona Bala Bora v. Jyotirindra Bhatacharjee, (2005) 4 SCC 501'
    - 'AIR 2020 SC 123'
    - 'Ram Kumar vs State of UP'
    - 'Maneka Gandhi vs Union of India'

    Args:
        query: Full case reference (party names, citation, or combination)
        size: Number of results to return
    """
    overall_start = time.time()
    logger.info("=" * 60)
    logger.info("SMART CASE SEARCH started | query='%s' | size=%d", query, size)
    logger.info("=" * 60)

    parsed = _parse_case_query(query)
    requested_year = parsed["year"]

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
    best_result_year_matched = False  # Track if year actually matches
    best_result_strategy = None  # Track which strategy gave best_result
    best_result_total_count = 0  # Track how specific the best result is
    ik_found = False  # Track if Indian Kanoon fallback succeeded

    # Strategies 1-3 (party_both_with_year, party_both_no_year, party_swapped) are
    # "high-confidence" — they match BOTH party names. If they find a small result
    # set (≤10), that's almost certainly the right case even if the year field is
    # corrupted. Later broad strategies (single_party, fulltext) should NOT override
    # a high-confidence result just because they happen to match the year.
    HIGH_CONFIDENCE_STRATEGIES = {
        "party_both_with_year", "party_both_no_year", "party_swapped",
    }
    HIGH_CONFIDENCE_MAX_TOTAL = 10  # If both-party match finds ≤10 results, trust it

    for strategy_fn in ALL_STRATEGIES:
        strategy_name, es_query = strategy_fn(parsed, min(size, 20))

        # Skip strategies that returned empty query (missing required fields)
        if not es_query:
            logger.debug("Strategy '%s': SKIPPED (missing required fields)", strategy_name)
            report_lines.append(f"  - {strategy_name}: SKIPPED (missing required fields)")
            continue

        try:
            logger.info("Strategy '%s': executing ES query...", strategy_name)
            strategy_start = time.time()
            response = execute_search(es_query)
            strategy_elapsed = time.time() - strategy_start
            hit_count = len(response.get("hits", {}).get("hits", []))
            total = response.get("hits", {}).get("total", {})
            total_count = total.get("value", 0) if isinstance(total, dict) else total

            strategies_tried.append(strategy_name)

            if hit_count > 0:
                logger.info("Strategy '%s': %d hits (total %d) in %.2fs", strategy_name, hit_count, total_count, strategy_elapsed)

                # If the user requested a specific year, validate that at least
                # one result is from that year (check citation field, not year field)
                if requested_year:
                    year_matched = _check_year_in_results(response, requested_year)
                    if year_matched:
                        # Before declaring WINNER: if we already have a high-confidence
                        # result from an earlier specific strategy (both-party match with
                        # ≤10 results), don't let a broad strategy override it.
                        # Broad strategies (single_party, fulltext) with 1000s of results
                        # can false-positive on year because they return so many documents.
                        if (best_result is not None
                                and best_result_strategy in HIGH_CONFIDENCE_STRATEGIES
                                and best_result_total_count <= HIGH_CONFIDENCE_MAX_TOTAL
                                and strategy_name not in HIGH_CONFIDENCE_STRATEGIES
                                and total_count > HIGH_CONFIDENCE_MAX_TOTAL):
                            logger.info(
                                "Strategy '%s': year matched but IGNORING — already have "
                                "high-confidence result from '%s' (%d results). "
                                "This broad strategy (%d results) likely false-positive.",
                                strategy_name, best_result_strategy,
                                best_result_total_count, total_count
                            )
                            report_lines.append(
                                f"  - {strategy_name}: {total_count} results (year matched "
                                f"but skipped — high-confidence result already found via "
                                f"{best_result_strategy})"
                            )
                            continue

                        logger.info("Strategy '%s': WINNER — year %d matched", strategy_name, requested_year)
                        report_lines.append(f"  - {strategy_name}: FOUND {total_count} results (year {requested_year} matched)")
                        best_result = response
                        best_result_year_matched = True
                        best_result_strategy = strategy_name
                        best_result_total_count = total_count
                        break
                    else:
                        # Results found but WRONG year — save as fallback, keep trying
                        logger.warning("Strategy '%s': %d results but WRONG YEAR (wanted %d) — saving as fallback", strategy_name, total_count, requested_year)
                        report_lines.append(f"  - {strategy_name}: {total_count} results but NONE from year {requested_year}")
                        if best_result is None:
                            best_result = response  # Save as fallback (best we have so far)
                            best_result_strategy = strategy_name
                            best_result_total_count = total_count
                        continue  # Keep trying other strategies
                else:
                    # No specific year requested — any results are good
                    logger.info("Strategy '%s': WINNER — %d results (no year filter)", strategy_name, total_count)
                    report_lines.append(f"  - {strategy_name}: FOUND {total_count} results")
                    best_result = response
                    best_result_year_matched = True
                    best_result_strategy = strategy_name
                    best_result_total_count = total_count
                    break
            else:
                logger.debug("Strategy '%s': 0 hits in %.2fs", strategy_name, strategy_elapsed)
                report_lines.append(f"  - {strategy_name}: no results")

        except Exception as e:
            logger.error("Strategy '%s': ERROR — %s", strategy_name, str(e)[:120])
            report_lines.append(f"  - {strategy_name}: ERROR ({str(e)[:80]})")
            strategies_tried.append(strategy_name)

    report_lines.append("")
    report_lines.append(f"**Total strategies tried:** {len(strategies_tried)}")

    # Decide: use ES results or fall through to Indian Kanoon
    # If year was requested but NO strategy matched the year → try Indian Kanoon first
    use_internet_fallback = (best_result is None) or (requested_year and not best_result_year_matched)
    logger.info("Decision: use_internet_fallback=%s | best_result=%s | year_matched=%s",
                use_internet_fallback, best_result is not None, best_result_year_matched)

    if best_result and not use_internet_fallback:
        winning_strategy = strategies_tried[-1]
        report_lines.append(f"**Winning strategy:** {winning_strategy}")
        report_lines.append("")

        # Format the results
        formatted = format_results_to_string(best_result, max_results=size)
        report_lines.append(formatted)
    else:
        # Strategy 13: Indian Kanoon internet fallback
        report_lines.append("")
        if requested_year and best_result and not best_result_year_matched:
            report_lines.append("### Strategy 13: Indian Kanoon Internet Fallback")
            report_lines.append(
                f"Database found results but NONE from year {requested_year}. "
                f"Searching indiankanoon.org for the {requested_year} case..."
            )
        else:
            report_lines.append("### Strategy 13: Indian Kanoon Internet Fallback")
            report_lines.append("All 12 database strategies failed. Searching indiankanoon.org...")
        report_lines.append("")

        ik_found = False
        try:
            logger.info("Strategy 13: Starting Indian Kanoon internet fallback...")
            ik_start = time.time()
            from .indiankanoon_scraper import scrape_indiankanoon_search
            ik_result = scrape_indiankanoon_search(query, size)
            ik_elapsed = time.time() - ik_start
            logger.info("Indian Kanoon scrape completed in %.2fs", ik_elapsed)
            strategies_tried.append("indiankanoon_scrape")

            # Check for failure indicators in the result
            ik_failed_indicators = [
                "Case not found on Indian Kanoon",
                "blocking automated requests",
                "Could not complete the search",
            ]
            if any(indicator in ik_result for indicator in ik_failed_indicators):
                logger.warning("Indian Kanoon scrape: FAILED — no matching case found")
                report_lines.append(f"  - indiankanoon_scrape: no results")
                # Include IK's error details in the report
                for line in ik_result.split("\n"):
                    if "ERROR" in line or "blocking" in line.lower() or "HTTP" in line:
                        logger.warning("  IK detail: %s", line.strip())
                        report_lines.append(f"    {line.strip()}")
            else:
                logger.info("Indian Kanoon scrape: SUCCESS — case found on indiankanoon.org")
                report_lines.append(f"  - indiankanoon_scrape: FOUND on indiankanoon.org")
                ik_found = True

        except Exception as e:
            strategies_tried.append("indiankanoon_scrape")
            ik_result = None
            logger.error("Indian Kanoon scrape: EXCEPTION — %s", str(e)[:150])
            report_lines.append(f"  - indiankanoon_scrape: ERROR ({str(e)[:100]})")

        report_lines.append("")
        report_lines.append(f"**Total strategies tried:** {len(strategies_tried)}")
        report_lines.append("")

        if ik_found:
            report_lines.append(f"**Winning strategy:** indiankanoon_scrape (internet)")
            report_lines.append("")
            report_lines.append(ik_result)
        elif best_result:
            # Indian Kanoon also failed, but we have old ES results — show them as fallback
            report_lines.append(
                f"**Indian Kanoon did not find a {requested_year} case either.** "
                f"Showing closest matches from the database instead:"
            )
            report_lines.append("")
            formatted = format_results_to_string(best_result, max_results=size)
            report_lines.append(formatted)
        else:
            report_lines.append(
                "**No results found after exhaustive search (database + internet).** "
                "The case may not exist, or the name/citation may be "
                "significantly different from what is indexed."
            )

    overall_elapsed = time.time() - overall_start
    logger.info("=" * 60)
    logger.info("SMART CASE SEARCH completed in %.2fs | strategies_tried=%d | result=%s",
                overall_elapsed, len(strategies_tried),
                "FOUND" if (best_result and best_result_year_matched) or ik_found else "NOT_FOUND")
    logger.info("=" * 60)

    return "\n".join(report_lines)