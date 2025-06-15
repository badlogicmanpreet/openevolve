"""
Utilities for code parsing, diffing, and manipulation
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def parse_evolve_blocks(code: str) -> List[Tuple[int, int, str]]:
    """
    Parse evolve blocks from code

    Args:
        code: Source code with evolve blocks

    Returns:
        List of tuples (start_line, end_line, block_content)
    """
    lines = code.split("\n")
    blocks = []

    in_block = False
    start_line = -1
    block_content = []

    for i, line in enumerate(lines):
        if "# EVOLVE-BLOCK-START" in line:
            in_block = True
            start_line = i
            block_content = []
        elif "# EVOLVE-BLOCK-END" in line and in_block:
            in_block = False
            blocks.append((start_line, i, "\n".join(block_content)))
        elif in_block:
            block_content.append(line)

    return blocks


def apply_diff(original_code: str, diff_text: str) -> str:
    """
    Apply a diff to the original code

    Args:
        original_code: Original source code
        diff_text: Diff in the SEARCH/REPLACE format

    Returns:
        Modified code
    """
    # Split into lines for easier processing
    original_lines = original_code.split("\n")
    result_lines = original_lines.copy()

    # Extract diff blocks
    diff_blocks = extract_diffs(diff_text)
    
    if not diff_blocks:
        logger.warning("No diff blocks found, returning original code")
        return original_code

    # Apply each diff block
    applied_count = 0
    for i, (search_text, replace_text) in enumerate(diff_blocks):
        search_lines = search_text.split("\n")
        replace_lines = replace_text.split("\n")

        # Find where the search pattern starts in the original code
        match_found = False
        for j in range(len(result_lines) - len(search_lines) + 1):
            # Try exact match first
            if result_lines[j : j + len(search_lines)] == search_lines:
                # Replace the matched section
                result_lines[j : j + len(search_lines)] = replace_lines
                match_found = True
                applied_count += 1
                logger.debug(f"Applied diff block {i+1}/{len(diff_blocks)}")
                break
        
        # If exact match failed, try with stripped whitespace
        if not match_found:
            search_lines_stripped = [line.strip() for line in search_lines]
            for j in range(len(result_lines) - len(search_lines) + 1):
                result_lines_stripped = [result_lines[k].strip() for k in range(j, j + len(search_lines))]
                if result_lines_stripped == search_lines_stripped:
                    # Replace the matched section
                    result_lines[j : j + len(search_lines)] = replace_lines
                    match_found = True
                    applied_count += 1
                    logger.debug(f"Applied diff block {i+1}/{len(diff_blocks)} (with whitespace tolerance)")
                    break
        
        if not match_found:
            logger.warning(f"Could not find match for diff block {i+1}: {search_text[:50]}...")

    logger.info(f"Applied {applied_count}/{len(diff_blocks)} diff blocks successfully")
    return "\n".join(result_lines)


def extract_diffs(diff_text: str) -> List[Tuple[str, str]]:
    """
    Extract diff blocks from the diff text

    Args:
        diff_text: Diff in the SEARCH/REPLACE format

    Returns:
        List of tuples (search_text, replace_text)
    """
    # Enhanced regex patterns to handle variations
    patterns = [
        # Standard format with optional spaces
        r"<{7}\s*SEARCH\s*\n(.*?)\n\s*={7}\s*\n(.*?)\n\s*>{7}\s*REPLACE\s*",
        # Case insensitive version
        r"<{7}\s*(?:search|SEARCH)\s*\n(.*?)\n\s*={3,}\s*\n(.*?)\n\s*>{7}\s*(?:replace|REPLACE)\s*",
        # Flexible separator length (3-7 characters)
        r"<{3,7}\s*(?:search|SEARCH)\s*\n(.*?)\n\s*={3,7}\s*\n(.*?)\n\s*>{3,7}\s*(?:replace|REPLACE)\s*",
        # Extra flexible with variable whitespace
        r"<+\s*(?:search|SEARCH)\s*\n(.*?)\n\s*=+\s*\n(.*?)\n\s*>+\s*(?:replace|REPLACE)\s*",
    ]
    
    for pattern in patterns:
        diff_blocks = re.findall(pattern, diff_text, re.DOTALL | re.IGNORECASE)
        if diff_blocks:
            # Clean up the matches - strip whitespace but preserve code structure
            cleaned_blocks = []
            for search, replace in diff_blocks:
                # Strip leading/trailing whitespace but preserve internal structure
                search_clean = search.strip()
                replace_clean = replace.strip()
                
                # Skip empty blocks
                if search_clean and replace_clean:
                    cleaned_blocks.append((search_clean, replace_clean))
            
            if cleaned_blocks:
                logger.debug(f"Found {len(cleaned_blocks)} diff blocks using pattern")
                return cleaned_blocks
    
    # Log parsing failure for debugging
    logger.warning("No valid diff blocks found in LLM response")
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"LLM response text: {diff_text[:500]}...")
    
    return []


def parse_full_rewrite(llm_response: str, language: str = "python") -> Optional[str]:
    """
    Extract a full rewrite from an LLM response

    Args:
        llm_response: Response from the LLM
        language: Programming language

    Returns:
        Extracted code or None if not found
    """
    code_block_pattern = r"```" + language + r"\n(.*?)```"
    matches = re.findall(code_block_pattern, llm_response, re.DOTALL)

    if matches:
        return matches[0].strip()

    # Fallback to any code block
    code_block_pattern = r"```(.*?)```"
    matches = re.findall(code_block_pattern, llm_response, re.DOTALL)

    if matches:
        return matches[0].strip()

    # Fallback to plain text
    return llm_response


def format_diff_summary(diff_blocks: List[Tuple[str, str]]) -> str:
    """
    Create a human-readable summary of the diff

    Args:
        diff_blocks: List of (search_text, replace_text) tuples

    Returns:
        Summary string
    """
    summary = []

    for i, (search_text, replace_text) in enumerate(diff_blocks):
        search_lines = search_text.strip().split("\n")
        replace_lines = replace_text.strip().split("\n")

        # Create a short summary
        if len(search_lines) == 1 and len(replace_lines) == 1:
            summary.append(f"Change {i+1}: '{search_lines[0]}' to '{replace_lines[0]}'")
        else:
            search_summary = (
                f"{len(search_lines)} lines" if len(search_lines) > 1 else search_lines[0]
            )
            replace_summary = (
                f"{len(replace_lines)} lines" if len(replace_lines) > 1 else replace_lines[0]
            )
            summary.append(f"Change {i+1}: Replace {search_summary} with {replace_summary}")

    return "\n".join(summary)


def calculate_edit_distance(code1: str, code2: str) -> int:
    """
    Calculate the Levenshtein edit distance between two code snippets

    Args:
        code1: First code snippet
        code2: Second code snippet

    Returns:
        Edit distance (number of operations needed to transform code1 into code2)
    """
    if code1 == code2:
        return 0

    # Simple implementation of Levenshtein distance
    m, n = len(code1), len(code2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i

    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if code1[i - 1] == code2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )

    return dp[m][n]


def extract_code_language(code: str) -> str:
    """
    Try to determine the language of a code snippet

    Args:
        code: Code snippet

    Returns:
        Detected language or "unknown"
    """
    # Look for Apex-specific patterns first (most specific to least specific)
    apex_patterns = [
        r"\[SELECT\s+.*\s+FROM\s+\w+.*\]",  # SOQL queries in brackets
        r"List<\w+>",  # Generic List declarations
        r"Map<\w+,\s*\w+>",  # Generic Map declarations
        r"Set<\w+>",  # Generic Set declarations
        r"(Account|Contact|Opportunity|Lead|Case)\s+\w+",  # Standard Salesforce objects
        r"System\.(debug|assert)",  # System class methods
        r"(trigger|webservice|testmethod|istest)",  # Apex keywords (case insensitive)
        r"Database\.(insert|update|delete|query)",  # Database DML operations
        r"Schema\.\w+",  # Schema namespace
        r"ApexPages\.\w+",  # Visualforce integration
        r"(public|private|global)\s+(static\s+)?(void|String|Integer|Boolean|Decimal)",  # Apex method signatures
    ]
    
    apex_match_count = 0
    for pattern in apex_patterns:
        if re.search(pattern, code, re.MULTILINE | re.IGNORECASE):
            apex_match_count += 1
    
    # Look for other common language signatures first to avoid false positives
    if re.search(r"^(import|from|def|class)\s", code, re.MULTILINE):
        return "python"
    elif re.search(r"^(package|import java|public class)", code, re.MULTILINE):
        # Check if it's really Java vs Apex
        java_specific = re.search(r"(System\.out\.println|String\[\]\s+args|main\s*\()", code)
        if java_specific:
            return "java"
        # If no Java-specific patterns but has Apex patterns, it's likely Apex
        elif apex_match_count >= 2:
            logger.debug(f"Detected Apex (not Java) with {apex_match_count} pattern matches")
            return "apex"
        else:
            return "java"  # Default to Java if uncertain
    elif re.search(r"^(#include|int main|void main)", code, re.MULTILINE):
        return "cpp"
    elif re.search(r"^(function|var|let|const|console\.log)", code, re.MULTILINE):
        return "javascript"
    elif re.search(r"^(module|fn|let mut|impl)", code, re.MULTILINE):
        return "rust"
    elif re.search(r"^(SELECT|CREATE TABLE|INSERT INTO)", code, re.MULTILINE):
        return "sql"
    
    # If we have 2 or more Apex-specific matches and no other language detected, it's likely Apex
    if apex_match_count >= 2:
        logger.debug(f"Detected Apex language with {apex_match_count} pattern matches")
        return "apex"

    # Fallback: if we had at least 1 strong Apex pattern and no other language detected
    strong_apex_patterns = [
        r"\[SELECT\s+.*\s+FROM\s+\w+.*\]",  # SOQL queries
        r"(Account|Contact|Opportunity|Lead|Case)\s+\w+",  # Salesforce objects
        r"(trigger|webservice|testmethod|istest)",  # Apex-specific keywords
    ]
    
    strong_apex_matches = 0
    for pattern in strong_apex_patterns:
        if re.search(pattern, code, re.MULTILINE | re.IGNORECASE):
            strong_apex_matches += 1
    
    if strong_apex_matches >= 1:
        logger.debug(f"Fallback Apex detection with {strong_apex_matches} strong pattern matches")
        return "apex"

    return "unknown"
