"""System prompt and user prompt builder for ScienceWorld evaluation."""

from typing import List, Tuple, Optional, TYPE_CHECKING

from .few_shot import FEW_SHOT_EXAMPLES, get_task_specific_examples

if TYPE_CHECKING:
    from ..memory import RetrievedMemory

# Base system prompt
_SYSTEM_PROMPT_BASE = """You are an intelligent agent operating in a virtual science laboratory environment. Your goal is to complete science experiment tasks by interacting with objects, using equipment, and applying scientific knowledge.

==================================================
ENVIRONMENT OVERVIEW
==================================================
This environment simulates a household with multiple rooms (kitchen, outside, workshop, etc.) containing various objects, equipment, and living things. Tasks cover topics like:
- Phase changes (boiling, melting, freezing)
- Temperature measurement
- Electrical circuits and conductivity
- Classification of living/non-living things
- Plant growth
- Chemistry (mixing substances)
- Biology (life stages, genetics)
- Physics (inclined planes, friction)

==================================================
AVAILABLE COMMANDS (25 actions)
==================================================
Navigation:
  - look around                    : Describe the current room
  - look at [object]               : Describe an object in detail
  - look in [object]               : Describe a container's contents
  - go to [location]               : Move to a new location (e.g., "go to kitchen")
  - teleport to [location]         : Teleport to a specific room (if enabled)

Object Manipulation:
  - pick up [object]               : Move an object to the inventory
  - put down [object]              : Drop an inventory item
  - move [object] to [location]    : Move an object to a container
  - focus on [object]              : Signal intent on a task object

Container Operations:
  - open/close [container]         : Open/close a container
  - pour [liquid] into [container] : Pour a liquid into a container
  - dunk [object] into [liquid]    : Dunk a container into a liquid
  - mix [object]                   : Chemically mix a container

Equipment/Device Operations:
  - activate [device]              : Activate/turn on a device
  - deactivate [device]            : Deactivate/turn off a device
  - use [object] [on target]       : Use a device/item
  - connect [obj1] to [obj2]       : Connect electrical components
  - disconnect [object]            : Disconnect electrical components
  - read [object]                  : Read a note or book

Other Actions:
  - eat [object]                   : Eat a food item
  - flush [object]                 : Flush a toilet
  - wait                           : Wait for 10 time steps (for slow processes)
  - wait1                          : Wait for 1 time step (for fine control)
  - inventory                      : List agent's inventory
  - task                           : Describe current task

==================================================
OUTPUT FORMAT (REQUIRED)
==================================================
You MUST respond in EXACTLY this format:

Think: <your reasoning about the current situation and next step>

Action: <exact command from the list above>

IMPORTANT:
- Always include both "Think:" and "Action:" sections
- The action must be a valid command with exact object names
- If stuck, use "check valid actions" to see available options
- You CAN carry multiple objects at once
- Phase changes may require time to complete"""

# System prompt with few-shot examples
SYSTEM_PROMPT_WITH_EXAMPLES = _SYSTEM_PROMPT_BASE + """

==================================================
EXAMPLE DEMONSTRATIONS
==================================================
The following examples show how to complete various science tasks:

""" + FEW_SHOT_EXAMPLES

# System prompt without examples
SYSTEM_PROMPT = _SYSTEM_PROMPT_BASE


def get_system_prompt(use_few_shot: bool = True, task_name: Optional[str] = None) -> str:
    """Get system prompt with or without few-shot examples.

    Args:
        use_few_shot: Whether to include few-shot examples.
        task_name: Optional task name for task-specific examples.

    Returns:
        System prompt string.
    """
    if not use_few_shot:
        return SYSTEM_PROMPT

    if task_name:
        task_examples = get_task_specific_examples(task_name)
        return _SYSTEM_PROMPT_BASE + """

==================================================
EXAMPLE DEMONSTRATIONS
==================================================
The following examples show how to complete similar tasks:

""" + task_examples

    return SYSTEM_PROMPT_WITH_EXAMPLES


def _format_trajectory_for_memory(trajectory: List[dict]) -> str:
    """Format trajectory for memory display.

    Args:
        trajectory: List of action-observation pairs.

    Returns:
        Formatted trajectory string (abbreviated).
    """
    if not trajectory:
        return "(empty)"

    max_show = 6
    lines = []

    if len(trajectory) <= max_show:
        for step in trajectory:
            action = step.get("action", "")
            lines.append(f"  > {action}")
    else:
        for step in trajectory[:3]:
            action = step.get("action", "")
            lines.append(f"  > {action}")
        lines.append(f"  ... ({len(trajectory) - 6} more steps) ...")
        for step in trajectory[-3:]:
            action = step.get("action", "")
            lines.append(f"  > {action}")

    return "\n".join(lines)


def _format_memory_items(memory_items: List) -> str:
    """Format memory items for display.

    Args:
        memory_items: List of MemoryEntry objects.

    Returns:
        Formatted memory items string.
    """
    if not memory_items:
        return ""

    lines = ["  Key Insights:"]
    for item in memory_items:
        lines.append(f"    - {item.title}: {item.description}")
        if item.content:
            content = item.content[:200] + \
                "..." if len(item.content) > 200 else item.content
            lines.append(f"      {content}")

    return "\n".join(lines)


def build_memory_section(retrieved_memories: List["RetrievedMemory"]) -> str:
    """Build the memory section for system prompt.

    Args:
        retrieved_memories: List of RetrievedMemory objects.

    Returns:
        Formatted memory section string.
    """
    if not retrieved_memories:
        return ""

    parts = [
        "",
        "==================================================",
        "RELEVANT EXPERIENCE FROM SIMILAR TASKS",
        "==================================================",
        "Below are experiences from past science experiments that may help with your current task.",
        "Use them as reference when relevant, but adapt to the specific situation.",
        "",
    ]

    for i, rm in enumerate(retrieved_memories, 1):
        result_str = "SUCCESS" if rm.is_success else "FAILED"
        parts.append(
            f"[Experience #{i}] (Similarity: {rm.similarity:.2f}, Result: {result_str})")
        parts.append(f"  Goal: {rm.query}")

        parts.append(f"  Actions taken:")
        parts.append(_format_trajectory_for_memory(rm.trajectory))

        if rm.memory_items:
            parts.append(_format_memory_items(rm.memory_items))

        parts.append("")

    return "\n".join(parts)


def get_system_prompt_with_memory(
    use_few_shot: bool = True,
    retrieved_memories: Optional[List["RetrievedMemory"]] = None,
    task_name: Optional[str] = None,
) -> str:
    """Get system prompt with optional few-shot examples and retrieved memories.

    Args:
        use_few_shot: Whether to include few-shot examples.
        retrieved_memories: Optional list of retrieved memories to include.
        task_name: Optional task name for task-specific examples.

    Returns:
        System prompt string.
    """
    base_prompt = get_system_prompt(use_few_shot, task_name)

    if not retrieved_memories:
        return base_prompt

    memory_section = build_memory_section(retrieved_memories)

    # Insert memory section before OUTPUT FORMAT section
    output_format_marker = "==================================================\nOUTPUT FORMAT"

    if output_format_marker in base_prompt:
        idx = base_prompt.find(output_format_marker)
        return base_prompt[:idx] + memory_section + "\n" + base_prompt[idx:]
    else:
        return base_prompt + memory_section


def build_user_prompt(
    task_description: str,
    history: List[Tuple[str, str]],
    current_observation: str,
    history_length: int = 20,
) -> str:
    """Build user prompt with task, history, and current observation.

    Args:
        task_description: The task goal description.
        history: List of (action, observation) tuples.
        current_observation: The most recent observation.
        history_length: Number of recent history entries to include.

    Returns:
        Formatted user prompt string.
    """
    parts = []

    # Add current task
    parts.append("==================================================")
    parts.append("YOUR CURRENT TASK")
    parts.append("==================================================")
    parts.append(f"Goal: {task_description}")
    parts.append("")
    parts.append("Hints:")
    parts.append("  - Type 'check valid actions' if you're unsure what to do")
    parts.append("  - Type 'inventory' to check what you're carrying")
    parts.append("  - Type 'look around' to observe your surroundings")
    parts.append("  - Use 'wait' command if a process needs time to complete")
    parts.append("")

    # Add recent history
    parts.append("==================================================")
    parts.append("RECENT HISTORY")
    parts.append("==================================================")

    recent_history = history[-history_length:] if len(
        history) > history_length else history

    if recent_history:
        for action, observation in recent_history:
            parts.append(f"Action: {action}")
            # Truncate long observations
            obs_display = observation[:500] + \
                "..." if len(observation) > 500 else observation
            parts.append(f"Observation: {obs_display}")
            parts.append("")

    # Add current observation
    parts.append("Current Observation:")
    parts.append(current_observation)
    parts.append("")

    # Reminder
    parts.append("==================================================")
    parts.append("YOUR TURN")
    parts.append("==================================================")
    parts.append(
        "Based on the task goal and current observation, decide your next action.")
    parts.append("Remember to use the exact format: Think: ... Action: ...")

    return "\n".join(parts)


def extract_task_description(initial_observation: str, task_desc_from_env: str = "") -> str:
    """Extract task description from observation or environment.

    Args:
        initial_observation: The initial environment observation.
        task_desc_from_env: Task description from environment info.

    Returns:
        Task description string.
    """
    # Prefer task description from environment
    if task_desc_from_env:
        return task_desc_from_env.strip()

    # Look for "Your task is to:" pattern in observation
    lines = initial_observation.split("\n")
    for line in lines:
        if "your task is to" in line.lower():
            return line.strip()

    # Return observation as fallback
    return initial_observation.strip()[:200]
