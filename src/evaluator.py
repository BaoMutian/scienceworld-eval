"""Evaluator for ScienceWorld with checkpoint support and memory integration."""

import logging
import random
from pathlib import Path
from typing import List, Set, Optional, Dict, Any

from tqdm import tqdm

from .config import Config
from .llm_client import LLMClient
from .environment import (
    ScienceWorldEnv,
    TASK_MAPPING,
    get_task_name_from_id,
    get_episode_id,
)
from .agent import EpisodeResult, run_single_episode
from .prompts import get_system_prompt, extract_task_description
from .utils import (
    game_result_to_dict,
    compute_summary,
    save_results,
    load_checkpoint,
    save_checkpoint,
    get_timestamp,
    generate_run_id,
)
from .logging_utils import (
    Colors,
    setup_logging,
    log_system_prompt,
    format_progress,
    format_episode_result,
)

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluator for ScienceWorld tasks with optional memory support."""

    def __init__(self, config: Config):
        """Initialize evaluator."""
        self.config = config
        self.llm_client = LLMClient(config.llm, config.retry)

        # State
        self._completed_episode_ids: Set[str] = set()
        self._results: List[EpisodeResult] = []
        self._success_count = 0
        self._success_steps = 0

        # Setup paths
        self.output_dir = Path(config.runtime.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.run_id = generate_run_id(config)
        self.checkpoint_path = self.output_dir / \
            f"{self.run_id}_checkpoint.json"
        self.results_path = self.output_dir / f"{self.run_id}_results.json"
        self.debug_log_path = self.output_dir / f"{self.run_id}_debug.log"

        if config.runtime.debug:
            setup_logging(debug=True, log_file=str(self.debug_log_path))

        # Initialize memory components if enabled
        self.memory_store = None
        self.memory_retriever = None
        self.memory_extractor = None
        self._init_memory()

    def _init_memory(self) -> None:
        """Initialize memory components if needed.

        Only initializes components when mode is not 'baseline'.
        baseline mode = no memory system at all.
        """
        if not self.config.memory.needs_memory_system():
            logger.debug(
                f"Memory mode is '{self.config.memory.mode}', skipping initialization")
            return

        try:
            from .memory import (
                EmbeddingModel,
                MemoryStore,
                MemoryRetriever,
                MemoryExtractor,
            )

            # Initialize embedding model
            embedding_model = EmbeddingModel(
                model_name=self.config.memory.embedding_model,
                device=self.config.memory.embedding_device,
            )

            # Initialize memory store
            self.memory_store = MemoryStore(
                memory_dir=self.config.memory.memory_dir,
                task_name=self.config.memory.task_name,
                embedding_model=embedding_model,
            )

            # Initialize retriever if needed
            if self.config.memory.should_retrieve():
                self.memory_retriever = MemoryRetriever(
                    store=self.memory_store,
                    embedding_model=embedding_model,
                    top_k=self.config.memory.top_k,
                    similarity_threshold=self.config.memory.similarity_threshold,
                )

            # Initialize extractor if needed
            if self.config.memory.should_extract():
                self.memory_extractor = MemoryExtractor(
                    llm_client=self.llm_client,
                    temperature=self.config.llm.temperature,
                    max_tokens=self.config.llm.max_tokens,
                )

            logger.info(
                f"Memory system initialized: mode={self.config.memory.mode}, "
                f"store_size={self.memory_store.size()}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize memory system: {e}")
            logger.warning("Falling back to baseline mode (no memory)")
            self.memory_store = None
            self.memory_retriever = None
            self.memory_extractor = None

    def _load_checkpoint(self) -> None:
        """Load checkpoint if exists."""
        checkpoint = load_checkpoint(str(self.checkpoint_path))
        self._completed_episode_ids = checkpoint["completed_episode_ids"]

        for r in checkpoint.get("results", []):
            result = EpisodeResult(
                episode_id=r["episode_id"],
                task_id=r["task_id"],
                task_name=r["task_name"],
                variation=r["variation"],
                success=r["success"],
                score=r["score"],
                steps=r["steps"],
                goal=r.get("goal", ""),
                actions=r.get("actions", []),
                observations=r.get("observations", []),
                thoughts=r.get("thoughts", []),
                error=r.get("error"),
                used_memories=r.get("used_memories", []),
            )
            self._results.append(result)
            if result.success:
                self._success_count += 1
                self._success_steps += result.steps

        if self._completed_episode_ids:
            print(
                f"{Colors.info('Checkpoint found:')} {len(self._completed_episode_ids)} episodes completed")

    def _save_checkpoint(self) -> None:
        """Save current checkpoint."""
        save_checkpoint(
            str(self.checkpoint_path),
            self._completed_episode_ids,
            [game_result_to_dict(r) for r in self._results],
        )

    def _add_result(self, result: EpisodeResult) -> None:
        """Add a result."""
        self._results.append(result)
        self._completed_episode_ids.add(result.episode_id)
        if result.success:
            self._success_count += 1
            self._success_steps += result.steps

    def get_task_schedule(self) -> List[Dict[str, Any]]:
        """Get list of (task_id, task_name, variation, episode) tuples to run.

        Returns:
            List of dicts with task info.
        """
        env = ScienceWorldEnv(self.config.test.simplifications)

        # Get task IDs to run
        if self.config.test.task_ids:
            task_ids = self.config.test.task_ids
        else:
            task_ids = list(TASK_MAPPING.keys())

        schedule = []

        for task_id in task_ids:
            task_name = get_task_name_from_id(task_id)
            if task_name == "unknown":
                logger.warning(f"Unknown task ID: {task_id}, skipping")
                continue

            # Get variations for this task
            try:
                variations = env.get_variations(
                    task_name, self.config.test.split)
            except Exception as e:
                logger.warning(
                    f"Failed to get variations for {task_name}: {e}")
                continue

            if not variations:
                logger.warning(
                    f"No variations found for {task_name} in {self.config.test.split}")
                continue

            # Shuffle variations based on seed (use deterministic hash)
            # Note: Python's hash() is non-deterministic for strings across runs
            # Use a simple deterministic hash instead
            task_seed = self.config.test.seed + sum(ord(c) for c in task_id)
            random.seed(task_seed)
            shuffled_vars = variations.copy()
            random.shuffle(shuffled_vars)

            # Limit to num_episodes variations (1 episode per variation for efficiency)
            # This gives num_episodes total per task
            selected_vars = shuffled_vars[:self.config.test.num_episodes]

            for var_idx, variation in enumerate(selected_vars):
                schedule.append({
                    "task_id": task_id,
                    "task_name": task_name,
                    "variation": variation,
                    "episode": 0,  # Single episode per variation
                    "episode_id": get_episode_id(task_id, variation, 0),
                })

        env.close()

        # Shuffle schedule with seed for reproducibility
        random.seed(self.config.test.seed)
        random.shuffle(schedule)

        return schedule

    def _retrieve_memories(self, task_name: str, goal: str) -> list:
        """Retrieve relevant memories for a task.

        Args:
            task_name: Name of the task.
            goal: Task goal description.

        Returns:
            List of RetrievedMemory objects.
        """
        if not self.memory_retriever:
            return []

        try:
            # Retrieve memories based on goal similarity
            retrieved = self.memory_retriever.retrieve(goal)

            # Display retrieval info
            if retrieved:
                for rm in retrieved:
                    result_tag = Colors.success(
                        "✓") if rm.is_success else Colors.warning("✗")
                    tqdm.write(
                        f"  {Colors.info('📚 Memory:')} {result_tag} "
                        f"sim={rm.similarity:.2f} | {rm.memory_items[0].title if rm.memory_items else 'No title'}"
                    )

            if self.config.runtime.debug and retrieved:
                logger.debug(
                    f"Retrieved {len(retrieved)} memories for goal: {goal[:50]}..."
                )

            return retrieved

        except Exception as e:
            logger.error(f"Memory retrieval failed: {e}")
            return []

    def _extract_and_store_memory(self, result: EpisodeResult) -> None:
        """Extract memory from episode result and store it.

        Args:
            result: Episode result to extract memory from.
        """
        if not self.memory_extractor or not self.memory_store:
            return

        try:
            # Build trajectory from result
            trajectory = []
            for i, action in enumerate(result.actions):
                obs = result.observations[i + 1] if i + \
                    1 < len(result.observations) else ""
                trajectory.append({
                    "action": action,
                    "observation": obs,
                })

            # Extract memory
            memory = self.memory_extractor.extract(
                task_id=result.episode_id,
                task_type=result.task_name,
                goal=result.goal,
                trajectory=trajectory,
                is_success=result.success,
            )

            if memory:
                self.memory_store.add(memory)
                # Display extraction info
                result_tag = Colors.success(
                    "✓") if memory.is_success else Colors.warning("✗")
                item_titles = [item.title for item in memory.memory_items[:2]]
                titles_str = ", ".join(item_titles)
                if len(memory.memory_items) > 2:
                    titles_str += f" +{len(memory.memory_items) - 2}"
                tqdm.write(
                    f"  {Colors.info('💡 Extracted:')} {result_tag} "
                    f"{len(memory.memory_items)} items | {titles_str}"
                )

                if self.config.runtime.debug:
                    logger.debug(
                        f"Extracted and stored memory {memory.memory_id} "
                        f"({len(memory.memory_items)} items)"
                    )

        except Exception as e:
            tqdm.write(f"  {Colors.error('⚠ Extract failed:')} {str(e)[:50]}")
            logger.error(
                f"Memory extraction failed for {result.episode_id}: {e}")
            # Don't propagate - extraction failure shouldn't stop evaluation

    def _build_trajectory_data(self, result: EpisodeResult) -> Dict[str, Any]:
        """Build complete trajectory data from episode result for MaTTS.

        Args:
            result: Episode result to extract trajectory from.

        Returns:
            Dictionary with full trajectory context.
        """
        # Build trajectory from result
        trajectory = []
        for i, action in enumerate(result.actions):
            obs = result.observations[i + 1] if i + \
                1 < len(result.observations) else ""
            trajectory.append({
                "action": action,
                "observation": obs,
            })

        # Get initial observation (first observation before any action)
        initial_obs = result.observations[0] if result.observations else ""

        return {
            "trajectory": trajectory,
            "is_success": result.success,
            "score": result.score,
            "steps": result.steps,
            "initial_observation": initial_obs,
            "goal": result.goal,
            "episode_id": result.episode_id,
        }

    def _run_matts_episode(
        self,
        task_info: Dict[str, Any],
        sample_idx: int,
    ) -> EpisodeResult:
        """Run a single MaTTS sample episode with higher temperature.

        Args:
            task_info: Task information dict.
            sample_idx: Sample index for identification.

        Returns:
            Episode result.
        """
        task_name = task_info["task_name"]
        variation = task_info["variation"]

        env = None
        try:
            # Create environment
            env = ScienceWorldEnv(self.config.test.simplifications)
            obs, info = env.reset(task_name, variation)
            goal = extract_task_description(obs, info.get("taskDesc", ""))

            # Retrieve memories (if any exist already)
            retrieved_memories = self._retrieve_memories(
                task_name, goal) if goal else []

            # Create agent with higher temperature for diverse sampling
            from .agent import ReActAgent

            agent = ReActAgent(
                llm_client=self.llm_client,
                use_few_shot=self.config.prompt.use_few_shot,
                history_length=self.config.prompt.history_length,
                debug=self.config.runtime.debug,
                retrieved_memories=retrieved_memories,
                task_name=task_name,
            )

            # Run episode
            result = agent.run_episode(
                env, obs, info,
                max_steps=self.config.test.max_steps,
                episode_num=sample_idx,
            )

            return result

        except Exception as e:
            logger.error(
                f"MaTTS sample {sample_idx} failed for {task_info['task_id']}: {e}")
            from .agent import EpisodeResult as ER
            return ER(
                episode_id=f"{task_info['task_id']}_v{variation}_s{sample_idx}",
                task_id=task_info["task_id"],
                task_name=task_name,
                variation=variation,
                success=False,
                score=0,
                steps=0,
                goal="",
                error=str(e),
            )
        finally:
            if env:
                env.close()

    def _run_matts_contrastive(self, task_info: Dict[str, Any]) -> Optional[EpisodeResult]:
        """Run MaTTS contrastive extraction for a task.

        Runs 1 main episode + N extra samples for contrastive extraction.
        The main episode result is used for evaluation statistics.

        Args:
            task_info: Task information dict.

        Returns:
            Main episode result (first sample), used for evaluation.
        """
        task_id = task_info["task_id"]
        task_name = task_info["task_name"]
        variation = task_info["variation"]
        extra_n = self.config.memory.matts.sample_n  # Extra samples for comparison
        total_samples = 1 + extra_n  # Main (1) + Extra (n)

        print(f"\n{Colors.highlight('='*50)}")
        print(f"{Colors.info('MaTTS Contrastive Extraction')}")
        print(f"  Task: {task_id} ({task_name}) v{variation}")
        print(f"  Samples: 1 main + {extra_n} extra = {total_samples} total")
        print(f"{Colors.highlight('='*50)}")

        # Collect trajectory samples: main (1) + extra (n)
        trajectories_data: List[Dict[str, Any]] = []
        results: List[EpisodeResult] = []

        for sample_idx in range(total_samples):
            is_main = sample_idx == 0
            label = "Main" if is_main else f"Extra {sample_idx}"
            print(
                f"\n{Colors.dim(f'--- {label} ({sample_idx + 1}/{total_samples}) ---')}")

            result = self._run_matts_episode(task_info, sample_idx)
            results.append(result)

            # Build trajectory data with full context
            traj_data = self._build_trajectory_data(result)
            trajectories_data.append(traj_data)

            # Display sample result
            status = Colors.success(
                "✓ SUCCESS") if result.success else Colors.error("✗ FAILED")
            marker = Colors.info("[EVAL]") if is_main else ""
            print(
                f"  Result: {status} | Score: {result.score} | Steps: {result.steps} {marker}")

        # Summarize all samples
        success_count = sum(1 for r in results if r.success)
        print(f"\n{Colors.info('Sample Summary:')}")
        print(f"  Total: {total_samples} (1 main + {extra_n} extra)")
        print(
            f"  Success: {Colors.success(str(success_count))}/{total_samples}")
        print(
            f"  Avg Score: {sum(r.score for r in results) / len(results):.1f}")

        # Main result for evaluation
        main_result = results[0]
        main_status = Colors.success(
            "SUCCESS") if main_result.success else Colors.error("FAILED")
        print(
            f"  {Colors.info('Main (Eval):')} {main_status} | Score: {main_result.score}")

        # Run contrastive extraction using all trajectories
        if self.memory_extractor and self.memory_store and trajectories_data:
            print(f"\n{Colors.info('Running contrastive extraction...')}")

            # Get goal from main result
            goal = main_result.goal

            # Use MaTTS-specific thinking mode
            memory = self.memory_extractor.extract_contrastive(
                task_id=f"{task_id}_v{variation}_matts",
                task_type=task_name,
                goal=goal,
                trajectories=trajectories_data,
                enable_thinking=self.config.memory.matts.enable_thinking,
            )

            if memory:
                self.memory_store.add(memory)
                print(
                    f"  {Colors.success('✓ Extracted')} {len(memory.memory_items)} high-quality items:")
                for item in memory.memory_items:
                    print(f"    • {Colors.info(item.title)}")
                    print(f"      {Colors.dim(item.description)}")
            else:
                print(f"  {Colors.warning('⚠ No memory items extracted')}")

        print(f"{Colors.highlight('='*50)}\n")

        # Record retrieval statistics for memories used in main result
        if main_result.used_memories and self.memory_store:
            memory_ids = [
                m.get("memory_id") for m in main_result.used_memories
                if m.get("memory_id")
            ]
            if memory_ids:
                self.memory_store.record_retrievals(memory_ids, main_result.success)

        # Return the main result (first sample) for evaluation statistics
        return main_result

    def _run_episode(self, task_info: Dict[str, Any]) -> EpisodeResult:
        """Run a single episode with optional memory support.

        If MaTTS is enabled, runs multiple samples and does contrastive extraction.
        Otherwise, runs a single episode with standard extraction.
        """
        # Check if MaTTS should be used
        if self.config.memory.should_use_matts():
            return self._run_matts_contrastive(task_info)

        # Standard single episode
        task_name = task_info["task_name"]
        variation = task_info["variation"]
        episode = task_info["episode"]

        env = None
        try:
            # Create single environment for the entire episode
            env = ScienceWorldEnv(self.config.test.simplifications)
            obs, info = env.reset(task_name, variation)
            goal = extract_task_description(obs, info.get("taskDesc", ""))

            # Retrieve relevant memories
            retrieved_memories = self._retrieve_memories(
                task_name, goal) if goal else []

            # Create agent and run
            from .agent import ReActAgent, EpisodeResult as ER

            agent = ReActAgent(
                llm_client=self.llm_client,
                use_few_shot=self.config.prompt.use_few_shot,
                history_length=self.config.prompt.history_length,
                debug=self.config.runtime.debug,
                retrieved_memories=retrieved_memories,
                task_name=task_name,
            )

            result = agent.run_episode(
                env, obs, info,
                max_steps=self.config.test.max_steps,
                episode_num=episode
            )

            # Record retrieval statistics for memories used
            if retrieved_memories and self.memory_store:
                memory_ids = [rm.memory_id for rm in retrieved_memories]
                self.memory_store.record_retrievals(memory_ids, result.success)

            # Extract and store memory if enabled (standard extraction)
            if self.config.memory.should_extract():
                self._extract_and_store_memory(result)

            return result

        except Exception as e:
            logger.error(
                f"Error running episode {task_info['episode_id']}: {e}")
            from .agent import EpisodeResult as ER
            return ER(
                episode_id=task_info["episode_id"],
                task_id=task_info["task_id"],
                task_name=task_name,
                variation=variation,
                success=False,
                score=0,
                steps=0,
                goal="",
                error=str(e),
            )
        finally:
            if env:
                env.close()

    def run(self) -> None:
        """Run the evaluation."""
        # Print header
        print()
        print(Colors.highlight("=" * 60))
        print(Colors.highlight("  ScienceWorld Evaluation"))
        print(Colors.highlight("=" * 60))
        print(f"  Model:    {Colors.info(self.config.llm.model)}")
        print(f"  Split:    {Colors.info(self.config.test.split)}")
        print(
            f"  Tasks:    {Colors.info(str(self.config.test.task_ids or 'all'))}")
        print(
            f"  Variations: {Colors.info(str(self.config.test.num_episodes))} per task")
        print(f"  Simplify: {Colors.info(self.config.test.simplifications)}")
        print(f"  Run ID:   {Colors.dim(self.run_id)}")

        # Print memory info
        if self.config.memory.enabled:
            print(Colors.dim("-" * 40))
            print(f"  Memory:   {Colors.info(self.config.memory.mode)}")
            if self.memory_store:
                stats = self.memory_store.get_stats()
                print(
                    f"  Bank:     {Colors.info(str(stats['total_memories']))} memories")
            else:
                print(f"  Bank:     {Colors.warning('Not initialized')}")

            # Print MaTTS info if enabled
            if self.config.memory.should_use_matts():
                matts = self.config.memory.matts
                print(Colors.dim("-" * 40))
                print(f"  {Colors.info('MaTTS Enabled:')}")
                print(f"    Samples:     {matts.sample_n} per task")
                print(f"    Temperature: {matts.temperature}")
                if matts.enable_thinking is not None:
                    thinking_str = Colors.success(
                        "ON") if matts.enable_thinking else Colors.dim("OFF")
                    print(f"    Thinking:    {thinking_str}")

        print(Colors.highlight("=" * 60))
        print()

        if self.config.runtime.debug:
            log_system_prompt(get_system_prompt(
                self.config.prompt.use_few_shot))

        self._load_checkpoint()

        # Get task schedule
        schedule = self.get_task_schedule()
        total_episodes = len(schedule)

        # Filter out already completed episodes
        remaining = [
            t for t in schedule
            if t["episode_id"] not in self._completed_episode_ids
        ]

        if not remaining:
            print(Colors.success("All episodes already completed!"))
        else:
            print(f"Total episodes: {Colors.info(str(total_episodes))}")
            if self._completed_episode_ids:
                print(
                    f"{Colors.success('Resuming:')} {len(self._completed_episode_ids)} done, "
                    f"{Colors.warning(str(len(remaining)))} remaining"
                )
            else:
                print(f"Remaining: {Colors.warning(str(len(remaining)))}")
            print()

            completed_since_save = 0

            # Sequential evaluation with progress bar
            with tqdm(
                remaining,
                desc="Evaluating",
                unit="ep",
                ncols=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            ) as pbar:
                for task_info in pbar:
                    episode_id = task_info["episode_id"]

                    # Skip if already completed
                    if episode_id in self._completed_episode_ids:
                        continue

                    try:
                        result = self._run_episode(task_info)
                        self._add_result(result)
                        completed_since_save += 1

                        # Log progress
                        completed = len(self._results)
                        progress_str = format_progress(
                            completed, total_episodes, self._success_count, self._success_steps
                        )
                        result_str = format_episode_result(
                            result.episode_id, result.success, result.score, result.steps
                        )
                        tqdm.write(f"{progress_str} | {result_str}")

                        # Save checkpoint periodically
                        if completed_since_save >= self.config.runtime.save_interval:
                            self._save_checkpoint()
                            completed_since_save = 0

                    except Exception as e:
                        logger.error(f"Error processing {episode_id}: {e}")

        # Final save
        self._save_checkpoint()

        timestamp = get_timestamp().replace(":", "-")
        final_results_path = self.output_dir / \
            f"{self.run_id}_{timestamp}_results.json"

        save_results(
            results=self._results,
            config_dict=self.config.to_dict(),
            output_path=str(final_results_path),
            model_name=self.config.llm.model,
        )

        save_results(
            results=self._results,
            config_dict=self.config.to_dict(),
            output_path=str(self.results_path),
            model_name=self.config.llm.model,
        )

        # Print summary
        self._print_summary(final_results_path)

    def _print_summary(self, final_results_path: Path) -> None:
        """Print evaluation summary."""
        summary = compute_summary(self._results)

        print()
        print(Colors.highlight("=" * 60))
        print(Colors.highlight("  EVALUATION COMPLETE"))
        print(Colors.highlight("=" * 60))
        print()

        rate_color = (
            Colors.BRIGHT_GREEN if summary["success_rate"] >= 0.7
            else Colors.BRIGHT_YELLOW if summary["success_rate"] >= 0.5
            else Colors.BRIGHT_RED
        )
        print(f"  Total episodes:  {summary['total_episodes']}")
        print(
            f"  Successes:       {Colors.success(str(summary['successes']))}")
        print(
            f"  Success rate:    {rate_color}{summary['success_rate']:.2%}{Colors.RESET}")
        print(f"  Avg score:       {summary['avg_score']:.1f}")
        print(f"  Avg steps:       {summary['avg_steps']:.1f}")
        print(f"  Success avg:     {summary['success_avg_steps']:.1f}")

        if summary["by_task_id"]:
            print()
            print(Colors.dim("-" * 40))
            print("  By Task ID:")
            for task_id, stats in sorted(summary["by_task_id"].items()):
                type_rate_color = (
                    Colors.BRIGHT_GREEN if stats["success_rate"] >= 0.7
                    else Colors.BRIGHT_YELLOW if stats["success_rate"] >= 0.5
                    else Colors.BRIGHT_RED
                )
                print(
                    f"    {task_id:6s} ({stats['task_name'][:20]:20s}) "
                    f"{type_rate_color}{stats['successes']:2d}/{stats['total']:2d} "
                    f"({stats['success_rate']:.0%}){Colors.RESET} "
                    f"avg_score={stats['avg_score']:.1f}"
                )

        # Print memory statistics if enabled
        if self.config.memory.enabled and self.memory_store:
            print()
            print(Colors.dim("-" * 40))
            print("  Memory Statistics:")
            stats = self.memory_store.get_stats()
            print(f"    Total memories:   {stats['total_memories']}")
            print(f"    Success memories: {stats['success_memories']}")
            print(f"    Failure memories: {stats['failure_memories']}")
            # Retrieval statistics
            if stats['total_retrievals'] > 0:
                rate = stats['avg_retrieval_success_rate']
                rate_color = (
                    Colors.BRIGHT_GREEN if rate >= 0.7
                    else Colors.BRIGHT_YELLOW if rate >= 0.5
                    else Colors.BRIGHT_RED
                )
                print(f"    Total retrievals: {stats['total_retrievals']}")
                print(
                    f"    Retrieval success rate: {rate_color}{rate:.2%}{Colors.RESET}"
                )

        print()
        print(Colors.highlight("=" * 60))
        print(f"  Results: {Colors.info(str(final_results_path))}")
        print(f"  Checkpoint: {Colors.dim(str(self.checkpoint_path))}")
        if self.memory_store:
            print(
                f"  Memory bank: {Colors.dim(str(self.memory_store.memories_path))}")
        print(Colors.highlight("=" * 60))
        print()


def run_evaluation(config: Config) -> None:
    """Run evaluation with the given configuration."""
    Evaluator(config).run()
