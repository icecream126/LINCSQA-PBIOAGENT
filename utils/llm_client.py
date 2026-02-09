"""
LLM Client module for vLLM server interaction.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

from openai import OpenAI

logger = logging.getLogger(__name__)


class VLLMClient:
    """Client for vLLM server with OpenAI-compatible API."""

    def __init__(
        self,
        port: int = 8000,
        model: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        temperature: float = 0.6,
        max_tokens: int = 4096,
        max_workers: int = 16,
        seed: int = 0,
        api_key: str = "EMPTY",
    ):
        self.port = port
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_workers = max_workers
        self.seed = seed
        
        base_url = f"http://localhost:{port}/v1"
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate completion from messages."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                seed=self.seed,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

    def generate_batch(
        self,
        messages_batch: List[List[Dict[str, str]]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> List[str]:
        """Generate completions for a batch of message sequences (sequential)."""
        results = []
        for messages in messages_batch:
            try:
                result = self.generate(messages, temperature, max_tokens)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch item generation failed: {e}")
                results.append("")
        return results

    def parallel_generate(
        self,
        tasks: List[Tuple[Any, ...]],
        task_fn: Callable[..., Any],
    ) -> List[Any]:
        """Execute tasks in parallel using ThreadPoolExecutor.
        
        Args:
            tasks: List of tuples, each containing arguments for task_fn
            task_fn: Function to execute for each task
            
        Returns:
            List of results in the same order as tasks
        """
        results = [None] * len(tasks)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(task_fn, *task_args): idx
                for idx, task_args in enumerate(tasks)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Task {idx} failed: {e}")
                    results[idx] = None
                    
        return results
