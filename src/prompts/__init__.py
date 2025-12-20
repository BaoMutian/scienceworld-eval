"""Prompt templates for ScienceWorld evaluation."""

from .system import (
    get_system_prompt,
    get_system_prompt_with_memory,
    build_user_prompt,
    extract_task_description,
)
from .few_shot import FEW_SHOT_EXAMPLES

