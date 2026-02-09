"""
Configuration and utility functions for the pipeline.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class SampleResult:
    """Result for a single sample."""
    sample: Dict[str, Any]
    predicted_label: Any
    ground_truth: Any
    reasoning: str = ""
    scientists_reasoning: Dict[str, str] = field(default_factory=dict)
    judges_verdict: Dict[str, Dict] = field(default_factory=dict)
    retry_count: int = 0
    is_correct: bool = False
    processing_time: float = 0.0
    # Additional fields for detailed tracing
    gat_prob: Optional[float] = None
    gat_label: Optional[int] = None
    answer: str = ""
    agents_input_prompts: Dict[str, str] = field(default_factory=dict)
    judges_input_prompts: Dict[str, str] = field(default_factory=dict)

    @property
    def status(self) -> str:
        """Return 'passed' if all judges passed, 'failed' otherwise."""
        if not self.judges_verdict:
            return "passed"  # No judges ran = passed
        for judge_result in self.judges_verdict.values():
            if isinstance(judge_result, dict) and judge_result.get("verdict") == "problematic":
                return "failed"
        return "passed"

    @property
    def final_problematic_count(self) -> int:
        """Count how many judges reported 'problematic'."""
        count = 0
        for judge_result in self.judges_verdict.values():
            if isinstance(judge_result, dict) and judge_result.get("verdict") == "problematic":
                count += 1
        return count


# =============================================================================
# Color Output
# =============================================================================


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def label_to_text(label: int, task_type: str = "task1") -> str:
    """Convert numeric label to text."""
    if task_type == "task1":
        return "upregulated" if label == 1 else "downregulated"
    else:
        return "yes" if label == 1 else "no"


def colorize_prediction(pred: str, gt_label: int, task_type: str = "task1") -> str:
    """Colorize prediction based on correctness."""
    if task_type == "task1":
        is_correct = (pred == "upregulated" and gt_label == 1) or (pred == "downregulated" and gt_label == 0)
    else:
        is_correct = (pred == "yes" and gt_label == 1) or (pred == "no" and gt_label == 0)
    
    color = Colors.GREEN if is_correct else Colors.RED
    return f"{color}{pred}{Colors.RESET}"


def colorize_verdict(verdict: str) -> str:
    """Colorize judge verdict."""
    if verdict == "not-problematic":
        return f"{Colors.GREEN}{verdict}{Colors.RESET}"
    else:
        return f"{Colors.RED}{verdict}{Colors.RESET}"


# =============================================================================
# Task 2 Label Parsing
# =============================================================================


def parse_llm_label_task2(text: str, allow_uncertain: bool = False) -> str:
    """Parse LLM response to extract Yes/No answer for task2.
    
    Args:
        text: LLM response text
        allow_uncertain: If True, return 'uncertain' when answer is ambiguous
    
    Returns:
        'yes', 'no', or 'uncertain' (if allow_uncertain=True)
    """
    text_lower = text.lower()
    
    # Check for explicit yes/no patterns
    yes_patterns = ["answer: yes", "answer is yes", "yes.", "\"yes\"", "'yes'", 
                    "the answer is yes", "my answer is yes", "conclude yes",
                    "final answer: yes", "final answer is yes"]
    no_patterns = ["answer: no", "answer is no", "no.", "\"no\"", "'no'",
                   "the answer is no", "my answer is no", "conclude no",
                   "final answer: no", "final answer is no"]
    uncertain_patterns = ["uncertain", "cannot determine", "not enough information",
                         "unable to determine", "insufficient"]
    
    for pattern in yes_patterns:
        if pattern in text_lower:
            return "yes"
    for pattern in no_patterns:
        if pattern in text_lower:
            return "no"
    
    if allow_uncertain:
        for pattern in uncertain_patterns:
            if pattern in text_lower:
                return "uncertain"
    
    # Simple fallback: check if 'yes' or 'no' appears more prominently
    yes_count = text_lower.count(" yes") + text_lower.count("yes ")
    no_count = text_lower.count(" no") + text_lower.count("no ")
    
    if yes_count > no_count:
        return "yes"
    elif no_count > yes_count:
        return "no"
    
    # Default behavior based on allow_uncertain
    return "uncertain" if allow_uncertain else "no"


# =============================================================================
# GAT Context Building
# =============================================================================


def build_gat_context(gat_pred: Optional[Dict], task_type: str = "task1") -> str:
    """Build GAT prediction context string."""
    if gat_pred is None:
        return "GAT prediction: Not available"
    
    if task_type == "task1":
        pred_label = "upregulated" if gat_pred.get("predicted_label", 0) == 1 else "downregulated"
        confidence = gat_pred.get("confidence", 0.5)
        return f"GAT prediction: {pred_label} (confidence: {confidence:.2%})"
    else:
        pred_label = "yes" if gat_pred.get("predicted_label", 0) == 1 else "no"
        confidence = gat_pred.get("confidence", 0.5)
        return f"GAT prediction: {pred_label} (confidence: {confidence:.2%})"


# =============================================================================
# Domain Knowledge
# =============================================================================


MOA_KNOWLEDGE: Dict[str, str] = {}
COMPOUND_KNOWLEDGE: Dict[str, str] = {}


def load_domain_knowledge(path: str) -> tuple:
    """Load domain knowledge from JSON file."""
    global MOA_KNOWLEDGE, COMPOUND_KNOWLEDGE
    
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            MOA_KNOWLEDGE = data.get("moa_knowledge", {})
            COMPOUND_KNOWLEDGE = data.get("compound_knowledge", {})
            logger.info(f"Loaded domain knowledge: {len(MOA_KNOWLEDGE)} MoAs, {len(COMPOUND_KNOWLEDGE)} compounds")
        except Exception as e:
            logger.warning(f"Failed to load domain knowledge: {e}")
    else:
        logger.warning(f"Domain knowledge file not found: {path}")
    
    return MOA_KNOWLEDGE, COMPOUND_KNOWLEDGE


def get_moa_knowledge(moa: str) -> str:
    """Get knowledge about a mechanism of action."""
    return MOA_KNOWLEDGE.get(moa, MOA_KNOWLEDGE.get(moa.lower(), ""))


def get_compound_knowledge(compound: str) -> str:
    """Get knowledge about a compound."""
    return COMPOUND_KNOWLEDGE.get(compound, COMPOUND_KNOWLEDGE.get(compound.lower(), ""))
