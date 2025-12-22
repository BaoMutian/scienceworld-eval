"""ReAct Agent for ScienceWorld evaluation."""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, TYPE_CHECKING

from .llm_client import LLMClient
from .environment import ScienceWorldEnv, get_episode_id
from .prompts import (
    get_system_prompt,
    get_system_prompt_with_memory,
    build_user_prompt,
    extract_task_description,
)
from .logging_utils import (
    Colors,
    log_episode_start,
    log_episode_end,
    log_step_interaction,
    format_episode_result,
)

if TYPE_CHECKING:
    from .memory import RetrievedMemory

logger = logging.getLogger(__name__)


@dataclass
class EpisodeResult:
    """Result of a single episode run."""
    episode_id: str
    task_id: str
    task_name: str
    variation: int
    success: bool
    score: float
    steps: int
    goal: str = ""
    actions: List[str] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    thoughts: List[str] = field(default_factory=list)
    error: Optional[str] = None
    # Memory-related fields
    used_memories: List[Dict[str, Any]] = field(default_factory=list)


class ReActAgent:
    """ReAct-style agent for ScienceWorld tasks."""

    # Action keywords for fallback parsing (matching ScienceWorld's 25 actions)
    ACTION_KEYWORDS = (
        # Navigation
        "look around", "look at", "look in", "go to", "teleport to",
        # Object manipulation
        "pick up", "put down", "move", "focus on",
        # Container operations
        "open", "close", "pour", "dunk", "mix",
        # Equipment/device operations
        "activate", "deactivate", "use", "connect", "disconnect", "read",
        # Other actions
        "eat", "flush", "wait", "inventory", "task",
    )

    def __init__(
        self,
        llm_client: LLMClient,
        use_few_shot: bool = True,
        history_length: int = 20,
        debug: bool = False,
        retrieved_memories: Optional[List["RetrievedMemory"]] = None,
        task_name: Optional[str] = None,
    ):
        """Initialize ReAct agent.

        Args:
            llm_client: LLM client for generating responses.
            use_few_shot: Whether to include few-shot examples.
            history_length: Number of history entries to include.
            debug: Whether to enable debug logging.
            retrieved_memories: Optional list of retrieved memories to use.
            task_name: Optional task name for task-specific prompts.
        """
        self.llm_client = llm_client
        self.history_length = history_length
        self.debug = debug
        self.retrieved_memories = retrieved_memories or []

        # Build system prompt with optional memories
        self.system_prompt = get_system_prompt_with_memory(
            use_few_shot=use_few_shot,
            retrieved_memories=self.retrieved_memories,
            task_name=task_name,
        )

    def parse_response(self, response: str) -> Tuple[str, str]:
        """Parse LLM response to extract thought and action."""
        thought = ""
        action = ""

        # Extract Think section
        think_match = re.search(
            r"Think(?:ing)?:\s*(.+?)(?=Action:|$)",
            response, re.DOTALL | re.IGNORECASE
        )
        if think_match:
            thought = think_match.group(1).strip()

        # Extract Action section
        action_match = re.search(
            r"Action:\s*(.+?)(?=Think|Thought|$)",
            response, re.DOTALL | re.IGNORECASE
        )
        if action_match:
            action = action_match.group(1).strip().split("\n")[0].strip()
            # Clean action: remove trailing parenthetical comments like "(this will...)"
            action = re.sub(r'\s*\([^)]*\)\s*$', '', action).strip()

        # Fallback: look for action-like lines
        if not action:
            for line in response.split("\n"):
                line_lower = line.lower().strip()
                if any(line_lower.startswith(kw) for kw in self.ACTION_KEYWORDS):
                    action = line.strip()
                    # Clean action
                    action = re.sub(r'\s*\([^)]*\)\s*$', '', action).strip()
                    break

        # Last resort: use last non-empty line
        if not action:
            lines = [l.strip() for l in response.split("\n") if l.strip()]
            if lines:
                action = lines[-1]
                action = re.sub(r'\s*\([^)]*\)\s*$', '', action).strip()

        return thought, action

    def run_episode(
        self,
        env: ScienceWorldEnv,
        initial_obs: str,
        info: Dict[str, Any],
        max_steps: int = 50,
        episode_num: int = 0,
    ) -> EpisodeResult:
        """Run a single episode with the agent."""
        task_desc = extract_task_description(
            initial_obs, info.get("taskDesc", ""))
        task_name = info.get("task_name", "")
        task_id = info.get("task_id", "")
        variation = info.get("variation", 0)

        episode_id = get_episode_id(task_id, variation, episode_num)

        # Record used memories
        used_memories = [rm.get_summary() for rm in self.retrieved_memories]

        result = EpisodeResult(
            episode_id=episode_id,
            task_id=task_id,
            task_name=task_name,
            variation=variation,
            success=False,
            score=0,
            steps=0,
            goal=task_desc,
            used_memories=used_memories,
        )

        # History stores (action, observation_after_action) pairs
        # build_user_prompt will handle not duplicating the last observation
        history: List[Tuple[str, str]] = []
        initial_observation = initial_obs  # Save for prompt building
        current_obs = initial_obs
        result.observations.append(current_obs)

        if self.debug:
            # Log to file only
            log_episode_start(episode_id, task_desc)
            # Print to terminal
            print(f"\n{Colors.highlight('='*50)}")
            print(f"{Colors.info('Episode:')} {episode_id}")
            print(
                f"{Colors.dim('Goal:')} {task_desc[:150]}{'...' if len(task_desc) > 150 else ''}")
            if self.retrieved_memories:
                print(f"{Colors.dim('Retrieved memories:')}")
                for rm in self.retrieved_memories:
                    status = Colors.success(
                        '✓') if rm.is_success else Colors.warning('✗')
                    titles = [item.title for item in rm.memory_items[:2]]
                    print(
                        f"  {status} sim={rm.similarity:.2f} | {', '.join(titles)}")
            print(f"{Colors.highlight('-'*50)}")

        try:
            for step in range(max_steps):
                user_prompt = build_user_prompt(
                    task_description=task_desc,
                    history=history,
                    current_observation=current_obs,
                    initial_observation=initial_observation,
                    history_length=self.history_length,
                )

                response = self.llm_client.chat_simple(
                    system_prompt=self.system_prompt,
                    user_prompt=user_prompt,
                )

                thought, action = self.parse_response(response)
                result.thoughts.append(thought)
                result.actions.append(action)

                obs, reward, done, step_info = env.step(action)
                result.observations.append(obs)

                if self.debug:
                    # Log full prompt/response to file only
                    log_step_interaction(
                        step=step + 1,
                        user_prompt=user_prompt,
                        response=response,
                        action=action,
                        observation=obs,
                        system_prompt=self.system_prompt if step == 0 else "",
                    )
                    # Print concise info to terminal
                    obs_preview = obs.replace('\n', ' ')[:80]
                    print(f"  [{step + 1:2d}] {Colors.info(action)}")
                    print(
                        f"      {Colors.dim('->')} {obs_preview}{'...' if len(obs) > 80 else ''}")

                # Add to history after LLM call (action, result_of_action)
                history.append((action, obs))
                current_obs = obs
                result.steps = step + 1
                result.score = step_info.get("score", 0)

                if step_info.get("is_complete", False):
                    result.success = True
                    if self.debug:
                        print(
                            f"  {Colors.success('>>> Task completed!')} Score: {result.score}")
                    break

                if done:
                    if self.debug:
                        print(
                            f"  {Colors.warning('>>> Episode ended')} Score: {result.score}")
                    break

        except Exception as e:
            result.error = str(e)
            logger.error(f"Error during episode {episode_id}: {e}")

        if self.debug:
            # Log to file
            log_episode_end(episode_id, result.success,
                            result.score, result.steps)
            # Print summary to terminal
            status = Colors.success(
                'SUCCESS') if result.success else Colors.error('FAILED')
            print(f"{Colors.highlight('-'*50)}")
            print(
                f"  Result: {status} | Score: {result.score:.0f} | Steps: {result.steps}")

        return result


def run_single_episode(
    task_name: str,
    variation_idx: int,
    episode_num: int,
    llm_client: LLMClient,
    simplifications: str = "easy",
    use_few_shot: bool = True,
    history_length: int = 20,
    max_steps: int = 50,
    debug: bool = False,
    retrieved_memories: Optional[List["RetrievedMemory"]] = None,
) -> EpisodeResult:
    """Run a single episode from scratch.

    Args:
        task_name: Name of the task.
        variation_idx: Variation index.
        episode_num: Episode number.
        llm_client: LLM client for generating responses.
        simplifications: Simplifications string.
        use_few_shot: Whether to include few-shot examples.
        history_length: Number of history entries to include.
        max_steps: Maximum steps per episode.
        debug: Whether to enable debug logging.
        retrieved_memories: Optional list of retrieved memories to use.

    Returns:
        EpisodeResult with execution results.
    """
    env = None
    try:
        env = ScienceWorldEnv(simplifications_str=simplifications)
        obs, info = env.reset(task_name, variation_idx)

        agent = ReActAgent(
            llm_client=llm_client,
            use_few_shot=use_few_shot,
            history_length=history_length,
            debug=debug,
            retrieved_memories=retrieved_memories,
            task_name=task_name,
        )

        return agent.run_episode(
            env, obs, info,
            max_steps=max_steps,
            episode_num=episode_num
        )

    except Exception as e:
        logger.error(
            f"Error running episode for {task_name} v{variation_idx}: {e}")
        from .environment import get_task_id_from_name
        task_id = get_task_id_from_name(task_name)
        episode_id = get_episode_id(task_id, variation_idx, episode_num)
        return EpisodeResult(
            episode_id=episode_id,
            task_id=task_id,
            task_name=task_name,
            variation=variation_idx,
            success=False,
            score=0,
            steps=0,
            goal="",
            error=str(e),
        )
    finally:
        if env:
            env.close()
