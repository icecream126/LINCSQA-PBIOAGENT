"""
Judge Agent classes for Multi-Scientist Multi-Judge Pipeline.

Contains:
- HistoryLeakageChecker: Verifies reasoning does not rely on direction labels from previous cases
- ReasoningTargetVerifier: Confirms reasoning refers to correct cell line, perturbation, and target gene
- ReasoningAnswerConsistencyChecker: Checks whether reasoning direction matches final answer
- ReasoningLogicChecker: Evaluates internal logical consistency of the reasoning
"""

import json
import logging
from typing import Dict, Tuple

from utils.prompt import return_prompt

logger = logging.getLogger(__name__)


class HistoryLeakageChecker:
    """Verifies that reasoning does not rely on direction labels from previous cases."""

    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.system_prompt = return_prompt("history_leakage_judge_system_prompt")
        self.user_prompt_template = return_prompt("history_leakage_judge_user_prompt")

    def build_prompt(
        self,
        history_summary: str,
        reasoning: str,
        answer: str,
    ) -> Tuple[str, str]:
        """Build system and user prompts."""
        user_prompt = self.user_prompt_template.format(
            history_summary=history_summary if history_summary else "No previous history available.",
            final_reasoning=reasoning,
            final_answer=answer,
        )
        return self.system_prompt, user_prompt

    def parse_response(self, response: str) -> Dict:
        """Parse judge response."""
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                parsed = json.loads(response[json_start:json_end])
                return {
                    "verdict": parsed.get("verdict", "not-problematic").lower(),
                    "feedback": parsed.get("feedback", ""),
                }
        except (json.JSONDecodeError, KeyError):
            pass
        return {"verdict": "not-problematic", "feedback": ""}


class ReasoningTargetVerifier:
    """Confirms that reasoning refers to the correct cell line, perturbation, and target gene."""

    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.system_prompt = return_prompt("grounding_judge_system_prompt")
        self.user_prompt_template = return_prompt("grounding_judge_user_prompt")

    def build_prompt(
        self,
        cell_line: str,
        pert: str,
        target_gene: str,
        reasoning: str,
        answer: str,
    ) -> Tuple[str, str]:
        """Build system and user prompts."""
        user_prompt = self.user_prompt_template.format(
            cell_line=cell_line,
            pert_or_moa=pert,
            target_gene=target_gene,
            final_reasoning=reasoning,
            final_answer=answer,
        )
        return self.system_prompt, user_prompt

    def parse_response(self, response: str) -> Dict:
        """Parse judge response."""
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                parsed = json.loads(response[json_start:json_end])
                return {
                    "verdict": parsed.get("verdict", "not-problematic").lower(),
                    "feedback": parsed.get("feedback", ""),
                }
        except (json.JSONDecodeError, KeyError):
            pass
        return {"verdict": "not-problematic", "feedback": ""}


class ReasoningAnswerConsistencyChecker:
    """Checks whether the reasoning direction matches the final answer."""

    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.system_prompt = return_prompt("consistency_judge_system_prompt")
        self.user_prompt_template = return_prompt("consistency_judge_user_prompt")
        self.user_prompt_template_task2 = return_prompt("consistency_judge_user_prompt_task2")

    def build_prompt(
        self,
        reasoning: str,
        answer: str,
        task_type: str = "task1",
        question: str = "",
        ground_truth_direction: str = "",
    ) -> Tuple[str, str]:
        """Build system and user prompts."""
        if task_type == "task2" and question:
            user_prompt = self.user_prompt_template_task2.format(
                question=question,
                ground_truth_direction=ground_truth_direction,
                final_reasoning=reasoning,
                final_answer=answer,
            )
        else:
            user_prompt = self.user_prompt_template.format(
                final_reasoning=reasoning,
                final_answer=answer,
            )
        return self.system_prompt, user_prompt

    def parse_response(self, response: str) -> Dict:
        """Parse judge response."""
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                parsed = json.loads(response[json_start:json_end])
                return {
                    "verdict": parsed.get("verdict", "not-problematic").lower(),
                    "feedback": parsed.get("feedback", ""),
                }
        except (json.JSONDecodeError, KeyError):
            pass
        return {"verdict": "not-problematic", "feedback": ""}


class ReasoningLogicChecker:
    """Evaluates internal logical consistency of the reasoning."""

    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.system_prompt = return_prompt("judge_system_prompt")
        self.user_prompt_template = return_prompt("judge_user_prompt")

    def build_prompt(
        self,
        question: str,
        answer: str,
        reasoning: str,
    ) -> Tuple[str, str]:
        """Build system and user prompts."""
        user_prompt = self.user_prompt_template.format(
            question=question,
            answer=answer,
            reasoning=reasoning,
        )
        return self.system_prompt, user_prompt

    def parse_response(self, response: str) -> Dict:
        """Parse judge response."""
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                parsed = json.loads(response[json_start:json_end])
                return {
                    "verdict": parsed.get("verdict", "not-problematic").lower(),
                    "feedback": parsed.get("feedback", ""),
                }
        except (json.JSONDecodeError, KeyError):
            pass
        return {"verdict": "not-problematic", "feedback": ""}
