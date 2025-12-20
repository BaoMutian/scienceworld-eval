"""ScienceWorld environment wrapper."""

import logging
from typing import List, Dict, Any, Optional, Tuple

from scienceworld import ScienceWorldEnv as SWEnv

logger = logging.getLogger(__name__)

# Task ID to name mapping (30 tasks across 10 topics)
TASK_MAPPING = {
    # Topic 1: Matter - Phase Changes
    "1-1": "boil",
    "1-2": "melt",
    "1-3": "freeze",
    "1-4": "change-the-state-of-matter-of",
    # Topic 2: Measurement
    "2-1": "use-thermometer",
    "2-2": "measure-melting-point-known-substance",
    "2-3": "measure-melting-point-unknown-substance",
    # Topic 3: Electricity
    "3-1": "power-component",
    "3-2": "power-component-renewable-vs-nonrenewable-energy",
    "3-3": "test-conductivity",
    "3-4": "test-conductivity-of-unknown-substances",
    # Topic 4: Classification
    "4-1": "find-living-thing",
    "4-2": "find-non-living-thing",
    "4-3": "find-plant",
    "4-4": "find-animal",
    # Topic 5: Biology - Plant Growth
    "5-1": "grow-plant",
    "5-2": "grow-fruit",
    # Topic 6: Chemistry
    "6-1": "chemistry-mix",
    "6-2": "chemistry-mix-paint-secondary-color",
    "6-3": "chemistry-mix-paint-tertiary-color",
    # Topic 7: Biology - Lifespan
    "7-1": "lifespan-longest-lived",
    "7-2": "lifespan-shortest-lived",
    "7-3": "lifespan-longest-lived-then-shortest-lived",
    # Topic 8: Biology - Life Stages
    "8-1": "identify-life-stages-1",
    "8-2": "identify-life-stages-2",
    # Topic 9: Forces
    "9-1": "inclined-plane-determine-angle",
    "9-2": "inclined-plane-friction-named-surfaces",
    "9-3": "inclined-plane-friction-unnamed-surfaces",
    # Topic 10: Biology - Genetics
    "10-1": "mendelian-genetics-known-plant",
    "10-2": "mendelian-genetics-unknown-plant",
}

# Reverse mapping
TASK_NAME_TO_ID = {v: k for k, v in TASK_MAPPING.items()}

# Tasks that require electrical actions (cannot use noElectricalAction simplification)
ELECTRICAL_TASKS = {"3-1", "3-2", "3-3", "3-4"}

# Simplification presets
SIMPLIFICATION_PRESETS = {
    "easy": ["teleportAction", "openDoors", "selfWateringFlowerPots", "noElectricalAction"],
}


def parse_simplifications(simplifications_str: str, task_id: Optional[str] = None) -> List[str]:
    """Parse simplifications string into list.
    
    Args:
        simplifications_str: Comma-separated simplifications or preset name.
        task_id: Optional task ID to check for electrical task constraints.
        
    Returns:
        List of simplification names.
    """
    if not simplifications_str:
        return []
    
    simplifications_str = simplifications_str.strip()
    
    # Check for preset
    if simplifications_str in SIMPLIFICATION_PRESETS:
        result = SIMPLIFICATION_PRESETS[simplifications_str].copy()
    else:
        result = [s.strip() for s in simplifications_str.split(",") if s.strip()]
    
    # Remove noElectricalAction for electrical tasks
    if task_id and task_id in ELECTRICAL_TASKS:
        if "noElectricalAction" in result:
            result.remove("noElectricalAction")
            logger.debug(f"Removed noElectricalAction for electrical task {task_id}")
    
    return result


def get_task_id_from_name(task_name: str) -> str:
    """Get task ID from task name.
    
    Args:
        task_name: Task name (e.g., "boil", "melt").
        
    Returns:
        Task ID (e.g., "1-1").
    """
    return TASK_NAME_TO_ID.get(task_name, "unknown")


def get_task_name_from_id(task_id: str) -> str:
    """Get task name from task ID.
    
    Args:
        task_id: Task ID (e.g., "1-1").
        
    Returns:
        Task name (e.g., "boil").
    """
    return TASK_MAPPING.get(task_id, "unknown")


def get_topic_from_task_id(task_id: str) -> int:
    """Get topic number from task ID.
    
    Args:
        task_id: Task ID (e.g., "1-1").
        
    Returns:
        Topic number (1-10).
    """
    try:
        return int(task_id.split("-")[0])
    except (ValueError, IndexError):
        return 0


class ScienceWorldEnv:
    """Wrapper for ScienceWorld environment."""

    # Special action for getting valid actions
    CHECK_VALID_ACTIONS = "check valid actions"

    def __init__(self, simplifications_str: str = "easy"):
        """Initialize ScienceWorld environment.
        
        Args:
            simplifications_str: Simplifications preset or comma-separated list.
        """
        self.simplifications_str = simplifications_str
        self.env: Optional[SWEnv] = None
        self.current_task_name: Optional[str] = None
        self.current_task_id: Optional[str] = None
        self.current_variation: Optional[int] = None
        self.valid_actions: List[str] = []
        self._env_initialized = False

    def _ensure_env(self) -> None:
        """Ensure environment is initialized."""
        if self.env is None:
            self.env = SWEnv()
            self._env_initialized = True

    def get_task_names(self) -> List[str]:
        """Get list of all available task names.
        
        Returns:
            List of task names.
        """
        self._ensure_env()
        return self.env.getTaskNames()

    def get_variations(self, task_name: str, split: str = "dev") -> List[int]:
        """Get available variations for a task.
        
        Args:
            task_name: Name of the task.
            split: Data split (train/dev/test).
            
        Returns:
            List of variation indices.
        """
        self._ensure_env()
        
        # Need to load the task first to get its variations
        self.env.load(task_name, 0, "")
        
        # Use appropriate method based on split
        if split == "train":
            return list(self.env.getVariationsTrain())
        elif split == "test":
            return list(self.env.getVariationsTest())
        else:  # dev
            return list(self.env.getVariationsDev())

    def get_all_tasks(self) -> Dict[str, str]:
        """Get mapping of all task IDs to names.
        
        Returns:
            Dictionary mapping task IDs to task names.
        """
        return TASK_MAPPING.copy()

    def reset(
        self,
        task_name: str,
        variation_idx: int,
        simplifications_str: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Reset environment with a specific task and variation.
        
        Args:
            task_name: Name of the task.
            variation_idx: Variation index.
            simplifications_str: Optional override for simplifications.
            
        Returns:
            Tuple of (initial_observation, info_dict).
        """
        self._ensure_env()
        
        self.current_task_name = task_name
        self.current_task_id = get_task_id_from_name(task_name)
        self.current_variation = variation_idx
        
        # Parse simplifications to list, then join back to string
        simpl_str = simplifications_str if simplifications_str is not None else self.simplifications_str
        simplifications = parse_simplifications(simpl_str, self.current_task_id)
        simplifications_param = ",".join(simplifications) if simplifications else ""
        
        # Load task - simplifications should be a comma-separated string
        self.env.load(task_name, variation_idx, simplifications_param)
        
        # Get initial observation
        obs, info = self.env.reset()
        
        # Store valid actions
        self.valid_actions = info.get("valid", [])
        
        # Get task description
        task_desc = info.get("taskDesc", "")
        
        return obs, {
            "valid": self.valid_actions,
            "score": info.get("score", 0),
            "taskDesc": task_desc,
            "task_name": task_name,
            "task_id": self.current_task_id,
            "variation": variation_idx,
            "simplifications": simplifications,
        }

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Execute action in environment.
        
        Args:
            action: Action string to execute.
            
        Returns:
            Tuple of (observation, reward, done, info).
        """
        if self.env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        # Handle special "check valid actions" command
        action_lower = action.lower().strip()
        if action_lower == self.CHECK_VALID_ACTIONS:
            valid_actions_str = "Valid actions:\n" + "\n".join(f"  - {cmd}" for cmd in self.valid_actions)
            return valid_actions_str, 0, False, {
                "valid": self.valid_actions,
                "score": 0,
                "done": False,
            }
        
        # Execute action in environment
        obs, reward, done, info = self.env.step(action)
        
        # Update valid actions
        self.valid_actions = info.get("valid", [])
        
        # Check if task is complete (score == 100)
        score = info.get("score", 0)
        is_complete = score >= 100
        
        return obs, reward, done or is_complete, {
            "valid": self.valid_actions,
            "score": score,
            "done": done or is_complete,
            "is_complete": is_complete,
        }

    def get_task_description(self) -> str:
        """Get current task description.
        
        Returns:
            Task description string.
        """
        if self.env is None:
            return ""
        return self.env.getTaskDescription()

    def get_score(self) -> float:
        """Get current score.
        
        Returns:
            Current score (0-100).
        """
        if self.env is None:
            return 0
        return self.env.getScore()

    def get_valid_actions(self) -> List[str]:
        """Get list of currently valid actions.
        
        Returns:
            List of valid action strings.
        """
        return self.valid_actions.copy()

    def close(self) -> None:
        """Close the environment."""
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
            self.env = None
            self._env_initialized = False


def get_episode_id(task_id: str, variation: int, episode: int) -> str:
    """Generate a unique episode ID.
    
    Args:
        task_id: Task ID (e.g., "1-1").
        variation: Variation index.
        episode: Episode number.
        
    Returns:
        Unique episode ID string.
    """
    return f"{task_id}_v{variation}_e{episode}"


def parse_episode_id(episode_id: str) -> Tuple[str, int, int]:
    """Parse episode ID to extract components.
    
    Args:
        episode_id: Episode ID string.
        
    Returns:
        Tuple of (task_id, variation, episode).
    """
    parts = episode_id.split("_")
    task_id = parts[0]
    variation = int(parts[1][1:])  # Remove 'v' prefix
    episode = int(parts[2][1:])    # Remove 'e' prefix
    return task_id, variation, episode

