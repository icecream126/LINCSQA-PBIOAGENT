"""
Data loader module for sorted pathway data.

Provides functions and classes for loading data from sorted pathway JSON files
and creating efficient batching iterators for cross-file processing.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional


logger = logging.getLogger(__name__)


@dataclass
class SortedSampleInfo:
    """
    Sample info with source file tracking for cross-file batching.
    
    Attributes:
        sample: Sample data dictionary.
        source_file: Full path to source JSON file.
        source_key: Unique key: {organ}/{task_type}/{test_case_id}/{gt_moa}/{compound}/{candidate_moa}.
        index_in_file: Position in the source file (for maintaining order).
        organ: Organ/tissue type.
        gt_moa: Ground truth MoA.
        test_case_id: Test case identifier.
        compound: Compound name.
        candidate_moa: Candidate MoA being evaluated.
    """
    sample: Dict
    source_file: str
    source_key: str
    index_in_file: int
    organ: str
    gt_moa: str
    test_case_id: str = ""
    compound: str = ""
    candidate_moa: str = ""


def load_sorted_pathway_data(
    sorted_pathway_dir: str,
    model_name: str,
    task_type: str,
    seed: int = 0,
    organs: List[str] = None,
    case_study: str = None,
) -> Tuple[List[SortedSampleInfo], Dict[str, List[SortedSampleInfo]]]:
    """
    Load data from sorted pathway JSON files (combined_score format).

    Path pattern for task1:
        {sorted_pathway_dir}/{model_name}/{organ}/task1/{test_case_id}/{gt_moa}/{compound}/{candidate_moa}_{seed}.json
    
    Path pattern for task2 (no organ level, task2 folder at case_study level):
        {sorted_pathway_dir}/{model_name}/{case_study}/task2/{test_case_id}/{gt_moa}/{compound}/{candidate_moa}_{seed}.json

    Args:
        sorted_pathway_dir: Base directory for sorted pathway data.
        model_name: Model name (with / replaced by _).
        task_type: Task type (task1 or task2).
        seed: Seed value to select specific JSON file (default: 0).
        organs: List of organs to load (for task1). Ignored for task2.
        case_study: Case study name for task2 (e.g., 'case1_braf').

    Returns:
        all_samples: List of all SortedSampleInfo objects.
        samples_by_source: Dict mapping source_key to list of samples from that file.
    """
    model_name_clean = model_name.replace("/", "_")
    
    # For task2, include case_study in the path
    if task_type == "task2" and case_study:
        base_path = os.path.join(sorted_pathway_dir, model_name_clean, case_study)
    else:
        base_path = os.path.join(sorted_pathway_dir, model_name_clean)

    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Model directory not found: {base_path}")

    all_samples = []
    samples_by_source = {}

    # Task2 has different structure: no organ folder, task2 folder directly under case_study
    if task_type == "task2":
        # Path: {base_path}/task2/{test_case_id}/{gt_moa}/{compound}/{candidate_moa}_{seed}.json
        task_path = os.path.join(base_path, "task2")
        
        if not os.path.exists(task_path):
            logger.warning(f"Task2 path not found: {task_path}")
            return all_samples, samples_by_source
        
        organ = "case_study"  # Use placeholder for task2
        
        # Iterate through test_case_id folders
        for test_case_id in os.listdir(task_path):
            test_case_path = os.path.join(task_path, test_case_id)
            if not os.path.isdir(test_case_path):
                continue

            # Iterate through gt_moa folders
            for gt_moa in os.listdir(test_case_path):
                gt_moa_path = os.path.join(test_case_path, gt_moa)
                if not os.path.isdir(gt_moa_path):
                    continue

                # Iterate through compound folders
                for compound in os.listdir(gt_moa_path):
                    compound_path = os.path.join(gt_moa_path, compound)
                    if not os.path.isdir(compound_path):
                        continue

                    # Find the JSON file with matching seed
                    for json_file in os.listdir(compound_path):
                        if not json_file.endswith(f"_{seed}.json"):
                            continue

                        candidate_moa = json_file[:-(len(f"_{seed}.json"))]
                        json_path = os.path.join(compound_path, json_file)
                        source_key = f"{organ}/{task_type}/{test_case_id}/{gt_moa}/{compound}/{candidate_moa}"

                        try:
                            with open(json_path, "r", encoding="utf-8") as f:
                                samples_data = json.load(f)
                        except Exception as e:
                            logger.error(f"Failed to load {json_path}: {e}")
                            continue

                        source_samples = []
                        for idx, sample_data in enumerate(samples_data):
                            sample = dict(sample_data)
                            sample["organ"] = organ
                            sample["gt_moa"] = gt_moa.replace("_", " ")
                            sample["source_key"] = source_key
                            sample["source_file"] = json_path
                            sample["test_case_id"] = test_case_id

                            if "compound" not in sample:
                                sample["compound"] = compound
                            if "candidate_moa" not in sample:
                                sample["candidate_moa"] = candidate_moa.replace("_", " ")

                            sample_info = SortedSampleInfo(
                                sample=sample,
                                source_file=json_path,
                                source_key=source_key,
                                index_in_file=idx,
                                organ=organ,
                                gt_moa=gt_moa.replace("_", " "),
                                test_case_id=test_case_id,
                                compound=compound,
                                candidate_moa=candidate_moa.replace("_", " "),
                            )
                            source_samples.append(sample_info)
                            all_samples.append(sample_info)

                        samples_by_source[source_key] = source_samples

        logger.info(f"Loaded {len(all_samples)} samples from {len(samples_by_source)} source files (task2)")
        return all_samples, samples_by_source

    # Task1: Original structure with organ folders
    if organs is None:
        organs = [
            d for d in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, d)) and d != "integrated"
        ]

    for organ in organs:
        task_path = os.path.join(base_path, organ, task_type)

        if not os.path.exists(task_path):
            logger.warning(f"Task path not found: {task_path}")
            continue

        for test_case_id in os.listdir(task_path):
            test_case_path = os.path.join(task_path, test_case_id)
            if not os.path.isdir(test_case_path):
                continue

            for gt_moa in os.listdir(test_case_path):
                gt_moa_path = os.path.join(test_case_path, gt_moa)
                if not os.path.isdir(gt_moa_path):
                    continue

                for compound in os.listdir(gt_moa_path):
                    compound_path = os.path.join(gt_moa_path, compound)
                    if not os.path.isdir(compound_path):
                        continue

                    for json_file in os.listdir(compound_path):
                        if not json_file.endswith(f"_{seed}.json"):
                            continue

                        candidate_moa = json_file[:-(len(f"_{seed}.json"))]
                        json_path = os.path.join(compound_path, json_file)
                        source_key = f"{organ}/{task_type}/{test_case_id}/{gt_moa}/{compound}/{candidate_moa}"

                        try:
                            with open(json_path, "r", encoding="utf-8") as f:
                                samples_data = json.load(f)
                        except Exception as e:
                            logger.error(f"Failed to load {json_path}: {e}")
                            continue

                        source_samples = []
                        for idx, sample_data in enumerate(samples_data):
                            sample = dict(sample_data)
                            sample["organ"] = organ
                            sample["gt_moa"] = gt_moa.replace("_", " ")
                            sample["source_key"] = source_key
                            sample["source_file"] = json_path
                            sample["test_case_id"] = test_case_id

                            if "compound" not in sample:
                                sample["compound"] = compound
                            if "candidate_moa" not in sample:
                                sample["candidate_moa"] = candidate_moa.replace("_", " ")

                            sample_info = SortedSampleInfo(
                                sample=sample,
                                source_file=json_path,
                                source_key=source_key,
                                index_in_file=idx,
                                organ=organ,
                                gt_moa=gt_moa.replace("_", " "),
                                test_case_id=test_case_id,
                                compound=compound,
                                candidate_moa=candidate_moa.replace("_", " "),
                            )
                            source_samples.append(sample_info)
                            all_samples.append(sample_info)

                        samples_by_source[source_key] = source_samples

    logger.info(f"Loaded {len(all_samples)} samples from {len(samples_by_source)} source files")
    return all_samples, samples_by_source


class CrossFileBatchIterator:
    """
    Iterator that creates batches with one sample from each active source file.

    Each batch contains exactly one sample from each candidate_moa.json file
    until that file is exhausted, then it is removed and replaced by the next
    available source file to keep batch size steady.
    Samples are processed in their sorted order within each file.
    """

    def __init__(
        self,
        samples_by_source: Dict[str, List[SortedSampleInfo]],
        batch_size: int
    ):
        """
        Args:
            samples_by_source: Dict mapping source_key to list of samples from that file
            batch_size: Number of source files to include per batch
        """
        self.batch_size = max(1, batch_size)

        # Track current position in each source file
        self.source_keys = list(samples_by_source.keys())
        self.samples_by_source = {k: list(v) for k, v in samples_by_source.items()}
        self.positions = {k: 0 for k in self.source_keys}
        self.source_queue = list(self.source_keys)
        self.active_sources: List[str] = []

        # Total samples
        self.total_samples = sum(len(v) for v in samples_by_source.values())
        self.processed = 0
        self.total_batches = self._estimate_total_batches()

    def _prune_exhausted_sources(self) -> None:
        """Remove active sources that have been fully consumed."""
        if not self.active_sources:
            return
        self.active_sources = [
            k for k in self.active_sources
            if self.positions[k] < len(self.samples_by_source[k])
        ]

    def _fill_active_sources(self) -> None:
        """Fill active sources up to batch_size from the remaining queue."""
        while len(self.active_sources) < self.batch_size and self.source_queue:
            self.active_sources.append(self.source_queue.pop(0))

    def _estimate_total_batches(self) -> int:
        """Estimate total batches based on current source lengths and batch size."""
        if not self.source_keys:
            return 0

        lengths = [len(self.samples_by_source[k]) for k in self.source_keys]
        batch_size = min(self.batch_size, len(lengths))
        active = lengths[:batch_size]
        queue = lengths[batch_size:]
        batches = 0

        while active:
            batches += 1
            next_active = []
            for remaining in active:
                remaining -= 1
                if remaining > 0:
                    next_active.append(remaining)
            while len(next_active) < batch_size and queue:
                next_active.append(queue.pop(0))
            active = next_active

        return batches

    def __iter__(self):
        return self

    def __next__(self) -> List[SortedSampleInfo]:
        """Return next batch of samples."""
        if self.processed >= self.total_samples:
            raise StopIteration

        self._prune_exhausted_sources()
        self._fill_active_sources()
        if not self.active_sources:
            raise StopIteration

        batch = []

        # Add one sample from each active source file
        for source_key in list(self.active_sources):
            pos = self.positions[source_key]
            samples = self.samples_by_source[source_key]
            if pos >= len(samples):
                continue
            batch.append(samples[pos])
            self.positions[source_key] = pos + 1
            self.processed += 1

        self._prune_exhausted_sources()
        self._fill_active_sources()

        if not batch:
            raise StopIteration

        return batch

    def __len__(self) -> int:
        """Estimate number of batches."""
        return self.total_batches
