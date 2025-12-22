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
This environment simulates a household with 10 interconnected locations containing various objects, equipment, and living things. Tasks cover topics like:
- Phase changes (boiling, melting, freezing)
- Temperature measurement
- Electrical circuits and conductivity
- Classification of living/non-living things
- Plant growth
- Chemistry (mixing substances)
- Biology (life stages, genetics)
- Physics (inclined planes, friction)

Locations:
- Kitchen       : This room is equipped with a fridge, stove, and sink, commonly used for thermodynamics experiments
- Bathroom      : A domestic area containing a sink and a toilet, often used for navigation or finding specific household items
- Workshop:     : This location houses various electrical components, such as batteries and wires
- Art Studio    : This room contains paints and artistic materials, serving as the primary site for chemical mixing and color-creation tasks
- Greenhouse    : A specialized environment for biological experiments
- Outside       : This outdoor space includes natural elements like soil and ponds
- Living Room   : A furnished area with bookshelves and paintings, frequently used for classification tasks or locating declarative knowledge in books
- Bedroom       : A standard room within the house theme that contains furniture such as a bed and is used for navigation and object search
- Hallway       : This area serves as the central connecting hub that allows agents to move between different locations in the house
- Foundry       : An industrial-themed location that features a large forge and is used for complex material-based experiments

==================================================
AVAILABLE COMMANDS
==================================================
Navigation:
  - look around                    : Describe the current room
  - look at [object]               : Describe an object in detail
  - look in [object]               : Describe a container's contents
  - go to [location]               : Move to a new location
  - teleport to [location]         : Teleport to a specific location

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
OUTPUT FORMAT
==================================================
You MUST respond in EXACTLY this format:

Think: <your reasoning about the current situation and next step>

Action: <exact command from the list above>

IMPORTANT:
- Always include both "Think:" and "Action:" sections
- The action must be a valid command with exact object names
- You CAN carry multiple objects at once"""

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

    Shows all actions in the trajectory (no observation, to keep it concise).

    Args:
        trajectory: List of action-observation pairs.

    Returns:
        Formatted trajectory string with all actions listed.
    """
    if not trajectory:
        return "(empty)"

    lines = []
    for i, step in enumerate(trajectory, 1):
        action = step.get("action", "")
        lines.append(f"  {i}. {action}")

    return "\n".join(lines)


def _format_memory_items(memory_items: List) -> str:
    """Format memory items for display.

    Shows full content without truncation for LLM to learn from.

    Args:
        memory_items: List of MemoryEntry objects.

    Returns:
        Formatted memory items string with full content.
    """
    if not memory_items:
        return ""

    lines = ["  Key Insights:"]
    for item in memory_items:
        lines.append(f"    - {item.title}: {item.description}")
        if item.content:
            # Full content, no truncation
            lines.append(f"      {item.content}")

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
    initial_observation: str = "",
    history_length: int = 20,
) -> str:
    """Build user prompt with task, history, and current observation.

    Args:
        task_description: The task goal description.
        history: List of (action, observation_after_action) tuples.
        current_observation: The most recent observation (after last action, or initial if no actions).
        initial_observation: The initial environment observation before any actions.
        history_length: Number of recent history entries to include.

    Returns:
        Formatted user prompt string.

    Note:
        - The last entry in history has the same observation as current_observation,
          so we don't show the observation for the last action in history.
        - If history can fit within history_length, we also show the initial observation.
    """
    parts = []

    # Add current task
    parts.append("==================================================")
    parts.append("YOUR CURRENT TASK")
    parts.append("==================================================")
    parts.append(f"Goal: {task_description}")
    parts.append("")
    parts.append("Hints:")
    parts.append("  - Type 'inventory' to check what you're carrying")
    parts.append("  - Type 'look around' to observe your surroundings")
    parts.append("  - Use 'wait' command if a process needs time to complete")
    parts.append(
        "  - Use 'teleport' command (if enabled) to quickly move to a specific location")
    parts.append("")

    # Add recent history
    parts.append("==================================================")
    parts.append("RECENT HISTORY")
    parts.append("==================================================")

    # Determine which history entries to include
    if len(history) > history_length:
        recent_history = history[-history_length:]
        include_initial = False
    else:
        recent_history = history
        include_initial = True  # Can fit initial observation

    # Show initial observation if we have room and it exists
    if include_initial and initial_observation:
        parts.append("Initial Observation:")
        parts.append(initial_observation)
        parts.append("")

    # Add action-observation pairs (full observations, no truncation)
    # Don't show observation for the last action (it's the same as current_observation)
    if recent_history:
        for i, (action, observation) in enumerate(recent_history):
            parts.append(f"Action: {action}")
            # Show observation for all but the last entry
            if i < len(recent_history) - 1:
                parts.append(f"Observation: {observation}")
                parts.append("")
            else:
                # Last action - observation will be shown as current observation
                parts.append("")

    # Add current observation (the result of the last action, or initial if no actions)
    parts.append("==================================================")
    parts.append("CURRENT OBSERVATION")
    parts.append("==================================================")
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
