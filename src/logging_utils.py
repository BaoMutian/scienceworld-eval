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
        file_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_format)
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=root_level,
        handlers=handlers,
        force=True,
    )
    
    # Suppress noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("py4j").setLevel(logging.ERROR)
    logging.getLogger("scienceworld").setLevel(logging.WARNING)


def log_episode_start(episode_id: str, task_desc: str) -> None:
    """Log episode start for debug mode.
    
    Args:
        episode_id: Episode identifier.
        task_desc: Task description.
    """
    logger = logging.getLogger(__name__)
    logger.debug("=" * 60)
    logger.debug(f"EPISODE START: {episode_id}")
    logger.debug(f"Task: {task_desc}")
    logger.debug("=" * 60)


def log_episode_end(episode_id: str, success: bool, score: float, steps: int) -> None:
    """Log episode end for debug mode.
    
    Args:
        episode_id: Episode identifier.
        success: Whether episode was successful.
        score: Final score.
        steps: Number of steps taken.
    """
    logger = logging.getLogger(__name__)
    result = "SUCCESS" if success else "FAILED"
    logger.debug("-" * 60)
    logger.debug(f"EPISODE END: {episode_id}")
    logger.debug(f"Result: {result}, Score: {score}, Steps: {steps}")
    logger.debug("-" * 60)


def log_step_interaction(
    step: int,
    user_prompt: str,
    response: str,
    action: str,
    observation: str,
) -> None:
    """Log step interaction for debug mode.
    
    Args:
        step: Step number.
        user_prompt: User prompt sent to LLM.
        response: LLM response.
        action: Parsed action.
        observation: Environment observation.
    """
    logger = logging.getLogger(__name__)
    logger.debug(f"\n--- Step {step} ---")
    logger.debug(f"USER PROMPT:\n{user_prompt[:500]}...")
    logger.debug(f"LLM RESPONSE:\n{response}")
    logger.debug(f"PARSED ACTION: {action}")
    logger.debug(f"OBSERVATION: {observation}")


def log_system_prompt(system_prompt: str) -> None:
    """Log system prompt for debug mode.
    
    Args:
        system_prompt: System prompt content.
    """
    logger = logging.getLogger(__name__)
    logger.debug("=" * 60)
    logger.debug("SYSTEM PROMPT")
    logger.debug("=" * 60)
    logger.debug(system_prompt)
    logger.debug("=" * 60)


def format_progress(
    completed: int,
    total: int,
    successes: int,
    success_steps: int,
) -> str:
    """Format progress string for display.
    
    Args:
        completed: Number of completed episodes.
        total: Total number of episodes.
        successes: Number of successful episodes.
        success_steps: Total steps in successful episodes.
        
    Returns:
        Formatted progress string.
    """
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
    """Format episode result for display.
    
    Args:
        episode_id: Episode identifier.
        success: Whether episode was successful.
        score: Final score.
        steps: Number of steps.
        
    Returns:
        Formatted result string.
    """
    if success:
        result_str = Colors.success("✓")
    else:
        result_str = Colors.error("✗")
    
    return f"{result_str} {episode_id} | Score: {score:.0f} | Steps: {steps}"

