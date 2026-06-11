from __future__ import annotations

import csv
import inspect
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch

from src.config.config import BaseConfig, DataRuntimeConfig, ModelConfig, RunConfig, SystemConfig
from src.env.crowdsourcing_env import CrowdsourcingEnv
from src.env.entity import SubTask, Transition
from src.llm.function_templates import TemplateFunctionSet
from src.net.agent import Agent


_CONFIG_SOURCE_SAVED = False


@dataclass
class TrainingResult:
    log_folder: str
    avg_revenue: float
    avg_success_tasks: float
    stability: float
    episode_revenues: list[float]
    episode_success_tasks: list[int]


@dataclass
class TrainOptions:
    episodes: int
    seed: int | None = None
    enable_llm_augmentation: bool = False
    enable_credit_net: bool = False
    function_set: TemplateFunctionSet | None = None
    log_prefix: str = ""
    log_folder: str | None = None
    dataset_paths: dict[str, str] | None = None
    out_file_path: str | None = None
    append_out_file: bool = False
    csv_output_dir: str | None = None
    run_label: str = ""
    save_model_id: str | None = None


def _build_log_folder(prefix: str = "", log_folder: str | None = None) -> str:
    if log_folder is not None:
        os.makedirs(log_folder, exist_ok=True)
        return log_folder

    now = datetime.now()
    ts = now.strftime("%m%d_%H%M%S")
    folder_name = f"{prefix}_{ts}" if prefix else ts
    log_folder = os.path.join(BaseConfig.logs_dir, folder_name)
    os.makedirs(log_folder, exist_ok=True)
    return log_folder


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _should_collect_feature_stats(episode_idx: int) -> bool:
    if not RunConfig.enable_feature_stats_print:
        return False
    interval = RunConfig.feature_stats_episode_interval
    return interval <= 0 or episode_idx % interval == 0


def _init_episode_feature_stats():
    return {
        "entity_worker_vectors": [],
        "entity_task_vectors": [],
        "entity_subtask_vectors": [],
        "state_custom_vectors": [],
        "ob_custom_vectors": [],
        "ob_subtask_vectors": [],
        "ob_task_vectors": [],
    }


def _accumulate_episode_feature_stats(episode_stats, state, observations, env: CrowdsourcingEnv) -> None:
    # 每个时间步：统计所有 idle workers 的 info。
    episode_stats["state_custom_vectors"].append(list(state.customs))
    episode_stats["entity_worker_vectors"].extend([list(w_info[0]) for w_info in state.workers_infos])

    # 每个时间步：统计所有 unmatched tasks 的 info。
    episode_stats["entity_task_vectors"].extend([list(t_info) for t_info in state.tasks_infos])

    # 每个时间步：统计 unmatched tasks 下的全部 subtasks（无论是否 matched）。
    for task in env.unmatched_tasks:
        for subtask in task.subtasks:
            episode_stats["entity_subtask_vectors"].append(list(subtask.info(env.now_time)[0]))

    # 每个时间步：统计 observations 与 state。
    episode_stats["ob_custom_vectors"].extend([list(ob.customs) for ob in observations])
    episode_stats["ob_subtask_vectors"].extend([list(ob.subtask_info[0]) for ob in observations])
    episode_stats["ob_task_vectors"].extend([list(ob.task_info) for ob in observations])


def _print_vector_stats(name: str, vectors: list[list[float]]) -> None:
    if not vectors:
        print(f"{name}: count=0, dims=0")
        return

    dims = min(len(v) for v in vectors)
    print(f"{name}: count={len(vectors)}, dims={dims}")
    for dim_idx in range(dims):
        values = [float(v[dim_idx]) for v in vectors]
        n = len(values)
        mean = sum(values) / n
        var = sum((v - mean) ** 2 for v in values) / n
        std = var ** 0.5
        print(
            f"  d{dim_idx}: mean={mean:.4f}, std={std:.4f}, "
            f"min={min(values):.4f}, max={max(values):.4f}"
        )



def _print_run_boundary(label: str, phase: str) -> None:
    if not label:
        return
    marker = f"========== META_RUN_{phase} label={label} =========="
    print(marker)


def _save_config_source_once(log_dir: str) -> str | None:
    global _CONFIG_SOURCE_SAVED
    if _CONFIG_SOURCE_SAVED:
        return None

    os.makedirs(log_dir, exist_ok=True)
    config_source_path = os.path.join(log_dir, "config_source.py")
    with open(inspect.getfile(BaseConfig), "r", encoding="utf-8") as src:
        config_source = src.read()
    with open(config_source_path, "w", encoding="utf-8") as dst:
        dst.write(config_source)

    _CONFIG_SOURCE_SAVED = True
    return config_source_path


def run_training(options: TrainOptions) -> TrainingResult:
    """
    main training loop, returns aggregated results after all episodes.
    """

    # ===== initialization before training =====
    seed = time.time_ns() % (2**32 - 1)
    _set_global_seed(seed)

    ModelConfig.enable_llm_augmentation = options.enable_llm_augmentation
    ModelConfig.enable_credit_net = options.enable_credit_net

    log_folder = _build_log_folder(options.log_prefix, options.log_folder)
    if options.dataset_paths is None:
        raise ValueError("TrainOptions.dataset_paths must be provided explicitly")
    dataset_paths = options.dataset_paths
    csv_output_dir = options.csv_output_dir or log_folder
    os.makedirs(csv_output_dir, exist_ok=True)

    env = CrowdsourcingEnv(SystemConfig.time_window_size, dataset_paths=dataset_paths)
    agent = Agent()

    episode_revenues: list[float] = []
    episode_success_tasks: list[int] = []

    try:
        _print_run_boundary(options.run_label, "START")

        # ===== print static configurations of this run =====
        print(f"[Run] seed: {seed}")
        print(f"[Run] dataset.subtask_csv: {dataset_paths['subtask_csv']}")
        print(f"[Run] dataset.task_csv: {dataset_paths['task_csv']}")
        print(f"[Run] dataset.worker_csv: {dataset_paths['worker_csv']}")

        # reset the environment
        env.reset()
        print(
            f"tasks:{len(env.all_tasks)}, "
            f"subtasks:{sum([len(t.subtasks) for t in env.all_tasks])}, "
            f"workers:{len(env.all_workers)}"
        )

        # ===== run training for episodes =====
        for i in range(options.episodes):

            sum_revenue = 0.0
            env.reset()
            step = 0

            should_collect_feature_stats = _should_collect_feature_stats(i)
            episode_stats = _init_episode_feature_stats() if should_collect_feature_stats else None

            csv_file_path = os.path.join(csv_output_dir, f"{i}.csv")
            csvfile = open(csv_file_path, "w", newline="", encoding="utf-8")
            writer = csv.writer(csvfile)
            writer.writerow(
                ["step", "revenue"]
                + [
                    "unmatched_subtasks",
                    "unmatched_tasks",
                ]
            )

            # in each step: get state -> select action -> env.step -> record transition。
            while True:
                if env.is_done():
                    break
                if SystemConfig.scale_time_span and step > SystemConfig.time_span / SystemConfig.time_window_size:
                    break

                do_print = True
                step_result = env.step(
                    agent=agent,
                    function_set=options.function_set,
                    enable_llm_augmentation=options.enable_llm_augmentation,
                    enable_credit_net=options.enable_credit_net,
                    do_print=do_print,
                )
                if episode_stats is not None:
                    _accumulate_episode_feature_stats(
                        episode_stats,
                        step_result.state,
                        step_result.observations,
                        env,
                    )

                sum_revenue += step_result.revenue
                step += 1

                assert len(step_result.observations) == len(step_result.actions) == len(step_result.subtasks)

                t = Transition(
                    subtasks=step_result.subtasks,
                    observations=step_result.observations,
                    actions=step_result.actions,
                    log_probs=step_result.log_probs,
                    state=step_result.state,
                    next_state=step_result.next_state,
                    done=step_result.done,
                    is_matched=step_result.is_matched,
                )
                t.contrib_features = step_result.contrib_features

                agent.buffer.store(t)

                if do_print:
                    print("step:", step, "revenue:", step_result.revenue)

                writer.writerow([step, step_result.revenue, step_result.infos[0], step_result.infos[5]] )
                csvfile.flush()

            # ===== delayed reward backpropagation =====
            delayed_reward_sum = sum([task.reward for task in env.all_tasks if task.is_fully_matched()])
            assert sum_revenue - 1e-6 <= delayed_reward_sum <= sum_revenue + 1e-6


            for t in agent.buffer.transitions:
                for j, sub in enumerate(t.subtasks):
                    if t.actions[j] == 1:
                        if t.is_matched[j]:
                            if sub.task.is_fully_matched():
                                r = sub.estimated_reward()
                            else:
                                r = -sub.estimated_reward() * ModelConfig.penalty_matched_failed
                        else:
                            r = -sub.estimated_reward() * ModelConfig.penalty_unmatched
                    else:
                        r = 0.0

                    t.reward += r
                    t.reward_list.append(r)

            # ===== Episode Summary and learning =====
            sum_reward = sum(t.reward for t in agent.buffer.transitions)

            success_tasks = len([task for task in env.all_tasks if task.is_fully_matched()])
            success_subtasks = len([sub for t in env.all_tasks for sub in t.subtasks if sub.is_matched()])
            success_task_subtasks = sum([len(t.subtasks) for t in env.all_tasks if t.is_fully_matched()])

            episode_revenues.append(float(sum_revenue))
            episode_success_tasks.append(success_tasks)

            csvfile.close()
            print(f"[Episode {i}]:", end="  ")
            print("success matched tasks:", success_tasks, end="  ")
            print("success task_subtasks:", success_task_subtasks, end="  ")
            agent.learn()
            agent.buffer.clear()
            print(f"[Episode {i}]:", "complete learning", end="  ")

        if options.save_model_id is not None:
            agent.save_model(options.save_model_id)

    finally:
        _print_run_boundary(options.run_label, "END")

    # ===== aggregate training results =====
    avg_revenue = float(np.mean(episode_revenues)) if episode_revenues else 0.0
    avg_success_tasks = float(np.mean(episode_success_tasks)) if episode_success_tasks else 0.0

    if len(episode_revenues) >= 2:
        last = np.array(episode_revenues[-min(10, len(episode_revenues)):], dtype=np.float32)
        denom = max(abs(float(last.mean())), 1e-6)
        cv = float(last.std() / denom)
        stability = 1.0 / (1.0 + cv)
    else:
        stability = 1.0

    return TrainingResult(
        log_folder=csv_output_dir,
        avg_revenue=avg_revenue,
        avg_success_tasks=avg_success_tasks,
        stability=stability,
        episode_revenues=episode_revenues,
        episode_success_tasks=episode_success_tasks,
    )
