"""Configuration management for ScienceWorld evaluation."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class LLMConfig:
    """LLM service configuration."""
    api_base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = ""
    model: str = "qwen/qwen3-8b"
    temperature: float = 0.3
    max_tokens: int = 1024
    timeout: int = 60
    # Qwen3 thinking mode (for vLLM deployment): True/False/None(not set)
    enable_thinking: Optional[bool] = None


@dataclass
class RetryConfig:
    """Retry configuration."""
    max_retries: int = 3
    retry_interval: float = 1.0
    max_retry_interval: float = 30.0


@dataclass
class TestConfig:
    """Test configuration."""
    num_episodes: int = 5  # episodes per task variation
    # e.g., ["1-1", "4-1"] or None for all
    task_ids: Optional[List[str]] = None
    split: str = "dev"  # train/dev/test
    seed: int = 42
    max_steps: int = 50
    simplifications: str = "easy"  # easy, or comma-separated options


@dataclass
class PromptConfig:
    """Prompt configuration."""
    use_few_shot: bool = True
    history_length: int = 20


@dataclass
class RuntimeConfig:
    """Runtime configuration."""
    save_interval: int = 1
    output_dir: str = "results"
    debug: bool = False


@dataclass
class MaTTSConfig:
    """MaTTS (Memory-aware Test-Time Scaling) configuration."""
    enabled: bool = False
    sample_n: int = 3
    temperature: float = 0.7
    max_tokens: int = 1024


@dataclass
class MemoryConfig:
    """Memory system configuration."""
    enabled: bool = False
    mode: str = "baseline"  # baseline | retrieve_only | retrieve_and_extract

    # Storage configuration
    memory_dir: str = "memory_banks"
    task_name: str = "scienceworld"

    # Embedding model configuration
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_device: str = "cpu"

    # Retrieval parameters
    top_k: int = 1
    similarity_threshold: float = 0.5

    # MaTTS configuration
    matts: MaTTSConfig = field(default_factory=MaTTSConfig)

    def should_retrieve(self) -> bool:
        """Check if retrieval is enabled."""
        return self.enabled and self.mode in ("retrieve_only", "retrieve_and_extract")

    def should_extract(self) -> bool:
        """Check if extraction is enabled."""
        return self.enabled and self.mode == "retrieve_and_extract"

    def needs_memory_system(self) -> bool:
        """Check if memory system components need to be initialized."""
        return self.enabled and self.mode != "baseline"


@dataclass
class Config:
    """Main configuration class."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    test: TestConfig = field(default_factory=TestConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "Config":
        """Create configuration from dictionary."""
        config = cls()

        if "llm" in data:
            config.llm = LLMConfig(**data["llm"])
        if "retry" in data:
            config.retry = RetryConfig(**data["retry"])
        if "test" in data:
            config.test = TestConfig(**data["test"])
        if "prompt" in data:
            config.prompt = PromptConfig(**data["prompt"])
        if "runtime" in data:
            config.runtime = RuntimeConfig(**data["runtime"])
        if "memory" in data:
            memory_data = data["memory"].copy()
            # Handle nested matts config
            if "matts" in memory_data:
                memory_data["matts"] = MaTTSConfig(**memory_data["matts"])
            else:
                memory_data["matts"] = MaTTSConfig()
            config.memory = MemoryConfig(**memory_data)

        return config

    def validate(self) -> None:
        """Validate configuration."""
        # Check API key
        api_key = self.llm.api_key or os.environ.get(
            "OPENROUTER_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "API key not set. Set it in config or OPENROUTER_API_KEY/OPENAI_API_KEY environment variable."
            )
        self.llm.api_key = api_key

        # Check split
        valid_splits = ["train", "dev", "test"]
        if self.test.split not in valid_splits:
            raise ValueError(
                f"Invalid split: {self.test.split}. Must be one of {valid_splits}")

        # Check task_ids format if provided
        if self.test.task_ids is not None:
            for task_id in self.test.task_ids:
                if not self._is_valid_task_id(task_id):
                    raise ValueError(
                        f"Invalid task ID format: {task_id}. "
                        "Expected format like '1-1', '2-3', etc."
                    )

        # Validate simplifications
        valid_simplifications = {
            "easy", "teleportAction", "openDoors",
            "selfWateringFlowerPots", "noElectricalAction", "openContainers"
        }
        if self.test.simplifications:
            for s in self.test.simplifications.split(","):
                s = s.strip()
                if s and s not in valid_simplifications:
                    raise ValueError(
                        f"Invalid simplification: {s}. Must be one of {valid_simplifications}"
                    )

        # Create output directory
        Path(self.runtime.output_dir).mkdir(parents=True, exist_ok=True)

        # Validate memory configuration
        valid_memory_modes = ["baseline",
                              "retrieve_only", "retrieve_and_extract"]
        if self.memory.mode not in valid_memory_modes:
            raise ValueError(
                f"Invalid memory mode: {self.memory.mode}. "
                f"Must be one of {valid_memory_modes}"
            )

        # Create memory directory if memory is enabled
        if self.memory.enabled:
            Path(self.memory.memory_dir).mkdir(parents=True, exist_ok=True)

    def _is_valid_task_id(self, task_id: str) -> bool:
        """Check if task_id is in valid format (e.g., '1-1', '10-2')."""
        # Valid task IDs based on ScienceWorld's 30 tasks
        from .environment import TASK_MAPPING
        return task_id in TASK_MAPPING

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "llm": {
                "api_base_url": self.llm.api_base_url,
                "model": self.llm.model,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
                "timeout": self.llm.timeout,
                "enable_thinking": self.llm.enable_thinking,
            },
            "retry": {
                "max_retries": self.retry.max_retries,
                "retry_interval": self.retry.retry_interval,
                "max_retry_interval": self.retry.max_retry_interval,
            },
            "test": {
                "num_episodes": self.test.num_episodes,
                "task_ids": self.test.task_ids,
                "split": self.test.split,
                "seed": self.test.seed,
                "max_steps": self.test.max_steps,
                "simplifications": self.test.simplifications,
            },
            "prompt": {
                "use_few_shot": self.prompt.use_few_shot,
                "history_length": self.prompt.history_length,
            },
            "runtime": {
                "save_interval": self.runtime.save_interval,
                "output_dir": self.runtime.output_dir,
                "debug": self.runtime.debug,
            },
            "memory": {
                "enabled": self.memory.enabled,
                "mode": self.memory.mode,
                "memory_dir": self.memory.memory_dir,
                "task_name": self.memory.task_name,
                "embedding_model": self.memory.embedding_model,
                "embedding_device": self.memory.embedding_device,
                "top_k": self.memory.top_k,
                "similarity_threshold": self.memory.similarity_threshold,
                "matts": {
                    "enabled": self.memory.matts.enabled,
                    "sample_n": self.memory.matts.sample_n,
                    "temperature": self.memory.matts.temperature,
                    "max_tokens": self.memory.matts.max_tokens,
                },
            },
        }


def load_config(config_path: Optional[str] = None) -> Config:
    """Load and validate configuration.

    Args:
        config_path: Path to YAML config file. If None, uses default config.

    Returns:
        Validated Config object.
    """
    if config_path is None:
        # Use default config
        default_path = Path(__file__).parent.parent / "config" / "default.yaml"
        if default_path.exists():
            config = Config.from_yaml(str(default_path))
        else:
            config = Config()
    else:
        config = Config.from_yaml(config_path)

    config.validate()
    return config
