"""Logging utilities for ScienceWorld evaluation."""

import logging
import sys
from typing import Optional


class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground colors
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright variants
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"

    @classmethod
    def success(cls, text: str) -> str:
        """Format text as success (green)."""
        return f"{cls.BRIGHT_GREEN}{text}{cls.RESET}"

    @classmethod
    def error(cls, text: str) -> str:
        """Format text as error (red)."""
        return f"{cls.BRIGHT_RED}{text}{cls.RESET}"

    @classmethod
    def warning(cls, text: str) -> str:
        """Format text as warning (yellow)."""
        return f"{cls.BRIGHT_YELLOW}{text}{cls.RESET}"

    @classmethod
    def info(cls, text: str) -> str:
        """Format text as info (cyan)."""
        return f"{cls.BRIGHT_CYAN}{text}{cls.RESET}"

    @classmethod
    def highlight(cls, text: str) -> str:
        """Format text as highlight (magenta)."""
        return f"{cls.BRIGHT_MAGENTA}{text}{cls.RESET}"

    @classmethod
    def dim(cls, text: str) -> str:
        """Format text as dimmed."""
        return f"{cls.DIM}{text}{cls.RESET}"


class CleanDebugFormatter(logging.Formatter):
    """Clean formatter for debug log files - only timestamp and message."""

    def format(self, record):
        # Only show timestamp and message for clean logs
        timestamp = self.formatTime(record, self.datefmt)
        return f"[{timestamp}] {record.getMessage()}"


def setup_logging(debug: bool = False, log_file: Optional[str] = None) -> None:
    """Setup logging configuration.

    Args:
        debug: Whether to enable debug logging.
        log_file: Optional path to log file.
    """
    # Root logger level
    root_level = logging.DEBUG if debug else logging.INFO

    # Configure root logger
    handlers = []

    # Console handler - always INFO level (don't print DEBUG to terminal)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    handlers.append(console_handler)

    # File handler if specified - DEBUG level for full logs
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        # Use clean formatter for debug logs
        file_format = CleanDebugFormatter(datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_format)
        handlers.append(file_handler)

    logging.basicConfig(
        level=root_level,
        handlers=handlers,
        force=True,
    )

    # Suppress noisy loggers - only show errors
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("openai").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    logging.getLogger("py4j").setLevel(logging.ERROR)
    logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)
    logging.getLogger("scienceworld").setLevel(logging.ERROR)
    logging.getLogger("scienceworld.scienceworld").setLevel(logging.ERROR)
    logging.getLogger("tenacity").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)


def log_episode_start(episode_id: str, task_desc: str) -> None:
    """Log episode start for debug mode."""
    logger = logging.getLogger(__name__)
    logger.debug("")
    logger.debug("=" * 80)
    logger.debug(f"EPISODE: {episode_id}")
    logger.debug(f"TASK: {task_desc}")
    logger.debug("=" * 80)


def log_episode_end(episode_id: str, success: bool, score: float, steps: int) -> None:
    """Log episode end for debug mode."""
    logger = logging.getLogger(__name__)
    result = "SUCCESS" if success else "FAILED"
    logger.debug("")
    logger.debug(
        f"EPISODE END: {episode_id} | {result} | Score: {score} | Steps: {steps}")
    logger.debug("=" * 80)


def log_step_interaction(
    step: int,
    user_prompt: str,
    response: str,
    action: str,
    observation: str,
    system_prompt: str = "",
) -> None:
    """Log step interaction for debug mode.

    Clean format showing only essential information:
    - System prompt (only on first step)
    - User prompt (full content)
    - LLM response (full content)
    - Parsed action and observation
    """
    logger = logging.getLogger(__name__)

    logger.debug("")
    logger.debug(f">>> AGENT STEP {step}")

    # Log system prompt only on first step
    if step == 1 and system_prompt:
        logger.debug("")
        logger.debug("[SYSTEM PROMPT]")
        logger.debug(system_prompt)

    # Log full user prompt
    logger.debug("")
    logger.debug("[USER PROMPT]")
    logger.debug(user_prompt)

    # Log full LLM response
    logger.debug("")
    logger.debug("[LLM RESPONSE]")
    logger.debug(response)

    # Log parsed action and observation
    logger.debug("")
    logger.debug(f"[PARSED] Action: {action}")
    logger.debug(f"[RESULT] {observation}")
    logger.debug("")


def log_memory_extraction(
    task_id: str,
    system_prompt: str,
    user_prompt: str,
    response: str,
    success: bool = True,
    num_items: int = 0,
) -> None:
    """Log memory extraction interaction for debug mode.

    Args:
        task_id: Task identifier.
        system_prompt: System prompt for extraction.
        user_prompt: User prompt (extraction request).
        response: LLM response.
        success: Whether extraction was successful.
        num_items: Number of items extracted.
    """
    logger = logging.getLogger(__name__)

    logger.debug("")
    logger.debug(f">>> MEMORY EXTRACTION: {task_id}")

    logger.debug("")
    logger.debug("[SYSTEM PROMPT]")
    logger.debug(system_prompt)

    logger.debug("")
    logger.debug("[USER PROMPT]")
    logger.debug(user_prompt)

    logger.debug("")
    logger.debug("[LLM RESPONSE]")
    logger.debug(response)

    if success:
        logger.debug("")
        logger.debug(f"[RESULT] Extracted {num_items} memory items")
    logger.debug("")


def log_system_prompt(system_prompt: str) -> None:
    """Log system prompt for debug mode."""
    logger = logging.getLogger(__name__)
    logger.debug("")
    logger.debug(">>> SYSTEM PROMPT")
    logger.debug(system_prompt)
    logger.debug("")


def format_progress(
    completed: int,
    total: int,
    successes: int,
    success_steps: int,
) -> str:
    """Format progress string for display."""
    rate = successes / completed if completed > 0 else 0
    avg_steps = success_steps / successes if successes > 0 else 0

    return (
        f"[{completed}/{total}] "
        f"SR: {Colors.info(f'{rate:.1%}')} "
        f"({successes}/{completed}) "
        f"AvgSteps: {avg_steps:.1f}"
    )


def format_episode_result(
    episode_id: str,
    success: bool,
    score: float,
    steps: int,
) -> str:
    """Format episode result for display."""
    if success:
        result_str = Colors.success("✓")
    else:
        result_str = Colors.error("✗")

    return f"{result_str} {episode_id} | Score: {score:.0f} | Steps: {steps}"
