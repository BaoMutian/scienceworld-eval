"""Prompt templates for memory extraction."""

from typing import List, Dict

# ScienceWorld environment context for extraction prompts
ENVIRONMENT_CONTEXT = """## Environment Background

ScienceWorld is a text-based interactive environment for testing scientific reasoning abilities.
The agent navigates rooms, manipulates objects, and performs science experiments through text commands.

### Key Environment Features:
- **Navigation**: Rooms connected by doors/portals (kitchen, living room, outside, etc.)
- **Objects**: Various containers, substances, tools, and living things
- **State Changes**: Objects can change state (solid/liquid/gas), temperature, location
- **Scientific Tasks**: Include melting, boiling, freezing, mixing, growing plants, electrical circuits, etc.

### Available Action Types:
- Movement: go to [location], teleport to [location]
- Observation: look around, look at [object], look in [container], examine [object]
- Manipulation: pick up [object], put [object] in [container], open/close [container]
- Interaction: activate/deactivate [device], use [tool] on [object], connect/disconnect [objects]
- Task-specific: focus on [object], pour [liquid] in [container], mix [container], eat [food]
- Waiting: wait, wait1 (for time-dependent processes like heating, growing)

### Common Challenges:
- Finding objects requires systematic room exploration
- Temperature changes require heat sources (stove, fireplace) or cooling (freezer, outside)
- Many tasks require specific sequences (e.g., focus -> action -> wait)
- Electrical tasks need proper circuit connections
"""

# Prompt for extracting strategies from successful trajectories
EXTRACTION_PROMPT_SUCCESS = """You are an expert at analyzing science experiment trajectories and extracting reusable reasoning strategies.

{environment_context}

## Task Context
- Task Type: {task_type}
- Task Goal: {goal}
- Result: SUCCESS

## Trajectory
{trajectory}

## Instructions
Analyze this SUCCESSFUL trajectory and extract 1-3 reusable strategies that contributed to success.
For each strategy, provide:
1. **title**: A short, descriptive name (e.g., "Heat Source Selection", "Systematic Object Search")
2. **description**: A one-sentence summary of when this strategy applies
3. **content**: Detailed actionable insight including specific commands or patterns that work

Focus on:
- Key decision points that led to success
- Efficient patterns or shortcuts discovered
- Scientific reasoning that could apply to similar tasks
- Specific action sequences that are reusable

## Output Format
Return a JSON array of strategy objects:
```json
[
  {{
    "title": "Strategy Name",
    "description": "When to use this strategy",
    "content": "Detailed explanation of the strategy and how to apply it, including specific commands"
  }}
]
```

Output ONLY the JSON array, no additional text."""

# Prompt for extracting lessons from failed trajectories
EXTRACTION_PROMPT_FAILURE = """You are an expert at analyzing science experiment trajectories and extracting lessons from failures.

{environment_context}

## Task Context
- Task Type: {task_type}
- Task Goal: {goal}
- Result: FAILED

## Trajectory
{trajectory}

## Instructions
Analyze this FAILED trajectory and extract 1-3 preventive lessons that could help avoid similar failures.
For each lesson, provide:
1. **title**: A short, descriptive name (e.g., "Avoid Skipping Focus Step", "Check Container First")
2. **description**: A one-sentence summary of the pitfall to avoid
3. **content**: Detailed explanation of what went wrong and how to prevent it

Focus on:
- Critical mistakes or wrong assumptions
- Inefficient patterns that wasted steps
- Missing scientific knowledge that caused the failure
- What the correct approach should have been

## Output Format
Return a JSON array of lesson objects:
```json
[
  {{
    "title": "Lesson Name",
    "description": "Pitfall to avoid",
    "content": "What went wrong and how to prevent it, with correct approach"
  }}
]
```

Output ONLY the JSON array, no additional text."""

# System prompt for MaTTS contrastive extraction
MATTS_SYSTEM_PROMPT = """You are an expert in science experiment execution and reasoning analysis.

You will be given a user query (task goal) and multiple trajectories showing how an agent attempted the task. 
Some trajectories may be successful, and others may have failed.

## Guidelines

Your goal is to compare and contrast these trajectories to identify the most useful and generalizable strategies as memory items.

Use self-contrast reasoning:
- Identify patterns and strategies that consistently led to success.
- Identify mistakes or inefficiencies from failed trajectories and formulate preventative strategies.
- Prefer strategies that generalize beyond specific objects or exact wording.

## Important notes
- Think first: Why did some trajectories succeed while others failed?
- You can extract at most 5 memory items from all trajectories combined.
- Do not repeat similar or overlapping items.
- Do not mention specific object names or exact room layouts — focus on generalizable behaviors and reasoning patterns.
- Make sure each memory item captures actionable and transferable insights.

## Output Format
Your output must strictly follow this JSON format:
```json
[
  {
    "title": "<the title of the memory item>",
    "description": "<one sentence summary of the memory item>",
    "content": "<1-5 sentences describing the insights learned to successfully accomplish the task>"
  }
]
```

Output ONLY the JSON array, no additional text."""

# User prompt template for MaTTS
MATTS_USER_PROMPT_TEMPLATE = """## Environment Background

ScienceWorld is a text-based interactive environment for testing scientific reasoning abilities.
The agent navigates rooms, manipulates objects, and performs science experiments through text commands.

### Key Environment Features:
- **Navigation**: Rooms connected by doors/portals (kitchen, living room, outside, greenhouse, workshop, etc.)
- **Objects**: Various containers (cups, pots, jars), substances (water, ice, salt), tools, and living things
- **State Changes**: Objects can change state (solid/liquid/gas), temperature, location
- **Scientific Tasks**: Include melting, boiling, freezing, mixing, growing plants, electrical circuits, etc.

### Available Action Types:
- Movement: go to [location], teleport to [location]
- Observation: look around, look at [object], look in [container]
- Manipulation: pick up [object], put [object] in [container], open/close [container]
- Interaction: activate/deactivate [device], use [tool] on [object], pour [liquid] into [container]
- Task-specific: focus on [object], mix [container], wait/wait1
- Special: connect [component] to [component], disconnect [component]

### Common Challenges:
- Finding target objects requires systematic room exploration
- Temperature changes require appropriate heat/cold sources
- Many tasks require specific sequences (e.g., focus -> action -> wait)
- Time-dependent processes need proper wait commands

---

## Task Goal
{goal}

## Task Type
{task_type}

---

## Trajectories
{trajectories}

---

Based on the above trajectories, extract reusable strategies and lessons. Focus on what made successful attempts work and what caused failures."""

# Legacy prompt kept for backward compatibility
EXTRACTION_PROMPT_CONTRASTIVE = MATTS_USER_PROMPT_TEMPLATE


def format_trajectory(trajectory: List[Dict[str, str]]) -> str:
    """Format trajectory for prompt.

    Args:
        trajectory: List of action-observation pairs.

    Returns:
        Formatted trajectory string.
    """
    lines = []
    for i, step in enumerate(trajectory, 1):
        action = step.get("action", "")
        observation = step.get("observation", "")
        # Truncate very long observations for extraction (full content not needed)
        if len(observation) > 500:
            observation = observation[:500] + "..."
        lines.append(f"Step {i}:")
        lines.append(f"  Action: {action}")
        lines.append(f"  Observation: {observation}")
        lines.append("")
    return "\n".join(lines)


def format_trajectory_full(
    trajectory: List[Dict[str, str]],
    initial_observation: str = "",
) -> str:
    """Format trajectory with complete information for MaTTS.

    Args:
        trajectory: List of action-observation pairs.
        initial_observation: Initial environment observation.

    Returns:
        Formatted trajectory string with full details.
    """
    lines = []
    
    # Include initial observation if provided
    if initial_observation:
        lines.append("Initial Observation:")
        # Truncate very long initial observations
        if len(initial_observation) > 800:
            lines.append(f"  {initial_observation[:800]}...")
        else:
            lines.append(f"  {initial_observation}")
        lines.append("")
    
    # Format each step with full details
    for i, step in enumerate(trajectory, 1):
        action = step.get("action", "")
        observation = step.get("observation", "")
        
        lines.append(f"Step {i}:")
        lines.append(f"  Action: {action}")
        # Keep observations reasonably sized for context
        if len(observation) > 600:
            lines.append(f"  Observation: {observation[:600]}...")
        else:
            lines.append(f"  Observation: {observation}")
        lines.append("")
    
    return "\n".join(lines)


def format_multiple_trajectories(
    trajectories: List[Dict],
) -> str:
    """Format multiple trajectories for contrastive extraction.

    Args:
        trajectories: List of trajectory dicts with keys:
            - 'trajectory': List of action-observation pairs
            - 'is_success': Whether this attempt succeeded
            - 'score': Final score (0-100)
            - 'steps': Number of steps taken
            - 'initial_observation': (optional) Initial environment state
            - 'goal': (optional) Task goal

    Returns:
        Formatted string with all trajectories and their context.
    """
    lines = []
    
    for i, traj_data in enumerate(trajectories, 1):
        is_success = traj_data.get("is_success", False)
        score = traj_data.get("score", 0)
        steps = traj_data.get("steps", len(traj_data.get("trajectory", [])))
        result = "SUCCESS" if is_success else "FAILED"
        
        # Header with trajectory metadata
        lines.append(f"{'='*60}")
        lines.append(f"### Trajectory {i}: {result}")
        lines.append(f"- Final Score: {score}")
        lines.append(f"- Total Steps: {steps}")
        lines.append(f"{'='*60}")
        lines.append("")
        
        # Format the full trajectory
        initial_obs = traj_data.get("initial_observation", "")
        trajectory = traj_data.get("trajectory", [])
        lines.append(format_trajectory_full(trajectory, initial_obs))
        lines.append("")
    
    return "\n".join(lines)


def build_extraction_prompt(
    task_type: str,
    goal: str,
    trajectory: List[Dict[str, str]],
    is_success: bool,
) -> str:
    """Build extraction prompt for a single trajectory.

    Args:
        task_type: Type of the task (task_name).
        goal: Task goal description.
        trajectory: List of action-observation pairs.
        is_success: Whether the task was successful.

    Returns:
        Formatted prompt string.
    """
    template = EXTRACTION_PROMPT_SUCCESS if is_success else EXTRACTION_PROMPT_FAILURE
    formatted_trajectory = format_trajectory(trajectory)

    return template.format(
        environment_context=ENVIRONMENT_CONTEXT,
        task_type=task_type,
        goal=goal,
        trajectory=formatted_trajectory,
    )


def build_contrastive_extraction_prompt(
    task_type: str,
    goal: str,
    trajectories: List[Dict],
) -> str:
    """Build extraction prompt for multiple trajectories (MaTTS).

    Args:
        task_type: Type of the task (task_name).
        goal: Task goal description.
        trajectories: List of trajectory dicts with full context.

    Returns:
        Formatted user prompt string.
    """
    formatted_trajectories = format_multiple_trajectories(trajectories)

    return MATTS_USER_PROMPT_TEMPLATE.format(
        task_type=task_type,
        goal=goal,
        trajectories=formatted_trajectories,
    )


def get_matts_system_prompt() -> str:
    """Get the system prompt for MaTTS contrastive extraction.
    
    Returns:
        System prompt string.
    """
    return MATTS_SYSTEM_PROMPT
