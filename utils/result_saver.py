"""
Result saving utilities for the pipeline.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List

from utils.config import SampleResult

logger = logging.getLogger(__name__)


def _clean_path_component(s: str) -> str:
    """Clean a string to be used as a path component."""
    if not s:
        return "unknown"
    return s.replace("/", "-").replace(" ", "_").replace("(", "").replace(")", "").replace(":", "-")


def save_results_json(
    results: List[SampleResult],
    args,
    task_type: str,
    organ: str,
) -> None:
    """
    Save results to JSON files.

    Structure:
        results/{project_name}/{model_name}/{organ}/{task_type}/{test_case_id}/{gt_moa}/{compound}/{candidate_moa}_{seed}.json

    Args:
        results: List of SampleResult objects
        args: Command line arguments (must have results_dir, project_name, model_name, seed)
        task_type: Task type ('task1' or 'task2')
        organ: Organ name
    """
    model_name_clean = args.model_name.replace("/", "-")

    # Group results by (test_case_id, gt_moa, compound, candidate_moa)
    grouped = {}
    for r in results:
        sample = r.sample
        test_case_id = sample.get("test_case_id", "unknown")
        gt_moa = sample.get("gt_moa", "unknown")
        compound = sample.get("compound", sample.get("pert", "unknown"))
        candidate_moa = sample.get("candidate_moa", "unknown")
        key = (test_case_id, gt_moa, compound, candidate_moa)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(r)

    for (test_case_id, gt_moa, compound, candidate_moa), group_results in grouped.items():
        test_case_id_clean = _clean_path_component(test_case_id)
        gt_moa_clean = _clean_path_component(gt_moa)
        compound_clean = _clean_path_component(compound)
        candidate_moa_clean = _clean_path_component(candidate_moa)

        # New path structure: results/{project_name}/{model_name}/{organ}/{task_type}/{test_case_id}/{gt_moa}/{compound}/{candidate_moa}_{seed}.json
        output_dir = os.path.join(
            args.results_dir,
            args.project_name,
            model_name_clean,
            organ,
            task_type,
            test_case_id_clean,
            gt_moa_clean,
            compound_clean,
        )
        os.makedirs(output_dir, exist_ok=True)

        # Build result list in requested format
        result_list = []
        for r in group_results:
            sample = r.sample
            # Extract agents input prompts
            agents_prompts = r.agents_input_prompts if hasattr(r, 'agents_input_prompts') else {}
            judges_prompts = r.judges_input_prompts if hasattr(r, 'judges_input_prompts') else {}

            # Extract judges verdicts and feedbacks
            judges_verdict = r.judges_verdict if hasattr(r, 'judges_verdict') else {}
            history_leakage = judges_verdict.get("history_leakage", {})
            target_verifier = judges_verdict.get("target_verifier", {})
            consistency = judges_verdict.get("consistency", {})
            logic = judges_verdict.get("logic", {})

            # Convert ground_truth and predicted_label based on task type
            answer = r.answer if hasattr(r, 'answer') else ""
            if task_type == "task1":
                # task1: ground_truth is 1 for upregulated, 0 for downregulated (already stored correctly)
                # predicted_label: 1 if "upregulated", 0 if "downregulated"
                ground_truth_val = r.ground_truth
                if isinstance(answer, str):
                    predicted_label_val = 1 if answer.lower() == "upregulated" else 0
                else:
                    predicted_label_val = r.predicted_label
            else:
                # task2: ground_truth is always 1
                # predicted_label: 1 if "yes", 0 if "no"
                ground_truth_val = 1
                if isinstance(answer, str):
                    predicted_label_val = 1 if answer.lower() == "yes" else 0
                else:
                    predicted_label_val = 1 if r.predicted_label == "yes" else 0

            result_dict = {
                "gene": sample.get("gene", ""),
                "pert": sample.get("pert", ""),
                "cell_line": sample.get("cell_line", ""),
                "organ": sample.get("organ", organ),
                "ground_truth": ground_truth_val,
                "predicted_label": predicted_label_val,
                "gt_moa": sample.get("gt_moa", ""),
                "candidate_moa": sample.get("candidate_moa", ""),
                "compound": sample.get("compound", ""),
                "test_case_id": sample.get("test_case_id", ""),
                "gat_prob": r.gat_prob if hasattr(r, 'gat_prob') else None,
                "gat_label": r.gat_label if hasattr(r, 'gat_label') else None,
                "retry_count": r.retry_count,
                "final_problematic_count": r.final_problematic_count if hasattr(r, 'final_problematic_count') else 0,
                "status": r.status,
                # Agent prompts and reasoning
                "context_agent_input_prompt": agents_prompts.get("context", ""),
                "context_agent_reasoning": r.scientists_reasoning.get("context", "") if hasattr(r, 'scientists_reasoning') else "",
                "mechanism_agent_input_prompt": agents_prompts.get("mechanism", ""),
                "mechanism_agent_reasoning": r.scientists_reasoning.get("mechanism", "") if hasattr(r, 'scientists_reasoning') else "",
                "network_agent_input_prompt": agents_prompts.get("network", ""),
                "network_agent_reasoning": r.scientists_reasoning.get("network", "") if hasattr(r, 'scientists_reasoning') else "",
                "integration_agent_input_prompt": agents_prompts.get("integration", ""),
                "integration_agent_raw_output": r.reasoning,
                "integration_agent_reasoning": r.reasoning,
                # Judge prompts, verdicts, and feedbacks
                "history_leakage_input_prompt": judges_prompts.get("history_leakage", ""),
                "history_leakage_verdict": history_leakage.get("verdict", "not-problematic"),
                "history_leakage_feedback": history_leakage.get("feedback", ""),
                "target_verifier_input_prompt": judges_prompts.get("target_verifier", ""),
                "target_verifier_verdict": target_verifier.get("verdict", "not-problematic"),
                "target_verifier_feedback": target_verifier.get("feedback", ""),
                "consistency_checker_input_prompt": judges_prompts.get("consistency", ""),
                "consistency_checker_verdict": consistency.get("verdict", "not-problematic"),
                "consistency_checker_feedback": consistency.get("feedback", ""),
                "logic_checker_input_prompt": judges_prompts.get("logic", ""),
                "logic_checker_verdict": logic.get("verdict", "not-problematic"),
                "logic_checker_feedback": logic.get("feedback", ""),
            }
            result_list.append(result_dict)

        filename = f"{candidate_moa_clean}_{args.seed}.json"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result_list, f, indent=2, ensure_ascii=False)

        logger.debug(f"  Saved {len(result_list)} results to {filepath}")


def save_metrics_json(
    metrics: Dict[str, Any],
    args,
    task_type: str,
    organ: str,
    filename: str = "metrics.json",
) -> str:
    """
    Save metrics to JSON file.

    Structure:
        results/{project_name}/{model_name}/{organ}/{task_type}/{filename}

    Args:
        metrics: Metrics dictionary
        args: Command line arguments (must have results_dir, project_name, model_name)
        task_type: Task type ('task1' or 'task2')
        organ: Organ name
        filename: Output filename

    Returns:
        Path to saved file
    """
    model_name_clean = args.model_name.replace("/", "-")

    output_dir = os.path.join(
        args.results_dir,
        args.project_name,
        model_name_clean,
        organ,
        task_type,
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    # Add timestamp
    metrics["saved_at"] = datetime.now().isoformat()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved metrics to {output_path}")
    return output_path
