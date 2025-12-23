"""Utility functions for ScienceWorld evaluation."""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Set, Optional


def get_timestamp() -> str:
    """Get current timestamp string.
    
    Returns:
        ISO format timestamp string.
    """
    return datetime.now().isoformat()


def game_result_to_dict(result: Any) -> Dict[str, Any]:
    """Convert GameResult to dictionary for serialization.
    
    Args:
        result: GameResult object.
        
    Returns:
        Dictionary representation.
    """
    return {
        "episode_id": result.episode_id,
        "task_id": result.task_id,
        "task_name": result.task_name,
        "variation": result.variation,
        "success": result.success,
        "score": result.score,
        "steps": result.steps,
        "goal": result.goal,
        "actions": result.actions,
        "observations": result.observations,
        "thoughts": result.thoughts,
        "error": result.error,
        "used_memories": result.used_memories,
    }


def compute_summary(results: List[Any]) -> Dict[str, Any]:
    """Compute summary statistics from results.
    
    Args:
        results: List of GameResult objects.
        
    Returns:
        Summary dictionary.
    """
    if not results:
        return {
            "total_episodes": 0,
            "successes": 0,
            "success_rate": 0.0,
            "avg_score": 0.0,
            "avg_steps": 0.0,
            "success_avg_steps": 0.0,
            "by_task_id": {},
        }
    
    total = len(results)
    successes = sum(1 for r in results if r.success)
    success_steps = sum(r.steps for r in results if r.success)
    total_steps = sum(r.steps for r in results)
    total_score = sum(r.score for r in results)
    
    # Group by task_id
    by_task_id: Dict[str, Dict[str, Any]] = {}
    for r in results:
        task_id = r.task_id
        if task_id not in by_task_id:
            by_task_id[task_id] = {
                "task_name": r.task_name,
                "total": 0,
                "successes": 0,
                "total_score": 0,
                "total_steps": 0,
            }
        by_task_id[task_id]["total"] += 1
        by_task_id[task_id]["total_score"] += r.score
        by_task_id[task_id]["total_steps"] += r.steps
        if r.success:
            by_task_id[task_id]["successes"] += 1
    
    # Compute per-task stats
    for task_id, stats in by_task_id.items():
        stats["success_rate"] = stats["successes"] / stats["total"] if stats["total"] > 0 else 0
        stats["avg_score"] = stats["total_score"] / stats["total"] if stats["total"] > 0 else 0
        stats["avg_steps"] = stats["total_steps"] / stats["total"] if stats["total"] > 0 else 0
    
    return {
        "total_episodes": total,
        "successes": successes,
        "success_rate": successes / total if total > 0 else 0,
        "avg_score": total_score / total if total > 0 else 0,
        "avg_steps": total_steps / total if total > 0 else 0,
        "success_avg_steps": success_steps / successes if successes > 0 else 0,
        "by_task_id": by_task_id,
    }


def save_results(
    results: List[Any],
    config_dict: Dict[str, Any],
    output_path: str,
    model_name: str,
) -> None:
    """Save evaluation results to JSON file.
    
    Args:
        results: List of GameResult objects.
        config_dict: Configuration dictionary.
        output_path: Path to output file.
        model_name: Model name.
    """
    summary = compute_summary(results)
    
    output = {
        "model": model_name,
        "timestamp": get_timestamp(),
        "config": config_dict,
        "summary": summary,
        "results": [game_result_to_dict(r) for r in results],
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load checkpoint from file.
    
    Args:
        checkpoint_path: Path to checkpoint file.
        
    Returns:
        Checkpoint data with completed_episode_ids and results.
    """
    if not Path(checkpoint_path).exists():
        return {
            "completed_episode_ids": set(),
            "results": [],
        }
    
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return {
            "completed_episode_ids": set(data.get("completed_episode_ids", [])),
            "results": data.get("results", []),
        }
    except (json.JSONDecodeError, KeyError) as e:
        return {
            "completed_episode_ids": set(),
            "results": [],
        }


def save_checkpoint(
    checkpoint_path: str,
    completed_episode_ids: Set[str],
    results: List[Dict[str, Any]],
) -> None:
    """Save checkpoint to file.
    
    Args:
        checkpoint_path: Path to checkpoint file.
        completed_episode_ids: Set of completed episode IDs.
        results: List of result dictionaries.
    """
    data = {
        "completed_episode_ids": list(completed_episode_ids),
        "results": results,
        "timestamp": get_timestamp(),
    }
    
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def generate_run_id(config: Any) -> str:
    """Generate a stable run ID based on configuration.

    The run ID changes when any parameter that affects results changes.

    Args:
        config: Configuration object.

    Returns:
        Run ID string.
    """
    # All parameters that affect experiment results
    key_params = {
        # LLM parameters
        "model": config.llm.model,
        "temperature": config.llm.temperature,
        "max_tokens": config.llm.max_tokens,
        "enable_thinking": config.llm.enable_thinking,
        # Test parameters
        "split": config.test.split,
        "task_ids": sorted(config.test.task_ids) if config.test.task_ids else None,
        "num_episodes": config.test.num_episodes,
        "seed": config.test.seed,
        "max_steps": config.test.max_steps,
        "simplifications": config.test.simplifications,
        # Prompt parameters
        "use_few_shot": config.prompt.use_few_shot,
        "history_length": config.prompt.history_length,
        # Memory parameters
        "memory_enabled": config.memory.enabled,
        "memory_mode": config.memory.mode,
    }

    # Add memory-specific parameters if enabled
    if config.memory.enabled and config.memory.mode != "baseline":
        key_params.update({
            "embedding_model": config.memory.embedding_model,
            "top_k": config.memory.top_k,
            "similarity_threshold": config.memory.similarity_threshold,
        })

        # Add MaTTS parameters if enabled
        if config.memory.matts.enabled:
            key_params.update({
                "matts_enabled": True,
                "matts_sample_n": config.memory.matts.sample_n,
                "matts_temperature": config.memory.matts.temperature,
                "matts_enable_thinking": config.memory.matts.enable_thinking,
            })

    params_hash = hashlib.md5(
        json.dumps(key_params, sort_keys=True).encode()
    ).hexdigest()[:8]

    model_short = config.llm.model.split("/")[-1]
    task_str = "all" if not config.test.task_ids else f"t{len(config.test.task_ids)}"

    # Add memory mode suffix if enabled
    memory_suffix = ""
    if config.memory.enabled:
        mode_short = {
            "baseline": "base",
            "retrieve_only": "ret",
            "retrieve_and_extract": "retex"
        }
        memory_suffix = f"_mem{mode_short.get(config.memory.mode, config.memory.mode[:3])}"

        # Add MaTTS indicator if enabled
        if config.memory.matts.enabled:
            memory_suffix += "_matts"

    return f"{model_short}_{config.test.split}_{task_str}{memory_suffix}_{params_hash}"

