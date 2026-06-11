from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.config.config import ModelConfig
from src.env.crowdsourcing_env import CrowdsourcingEnv
from src.env.entity import ContributionContext, MatchDetailSnapshot, SubTask, Worker
from src.env.distance_util import distance_util


def _query_schema_description() -> dict[str, Any]:
    return {
        "function_definition": "Real callable function signature without the query. prefix.",
        "inputs": "Input parameter schema. Use {} when there are no parameters.",
        "outputs": {
            "type": "Return type annotation.",
            "structure": "Return shape or nested field schema; do not use example values.",
            "meaning": "Semantic meaning of the returned result.",
        },
        "meaning": "Semantic purpose of the query method.",
    }




class StateQuerySet:

    def __init__(self, env: CrowdsourcingEnv):
        self.env = env

    @property
    def now_time(self) -> int:

        return self.env.now_time

    def num_unmatched_tasks(self) -> int:

        return len(self.env.unmatched_tasks)

    def num_matched_tasks(self) -> int:

        return sum(1 for task in self.env.all_tasks if task.is_fully_matched())

    def task_reward_distribution(self) -> dict[str, float]:

        tasks = self.env.unmatched_tasks
        if not tasks:
            return {"mean": 0.0, "max": 0.0, "min": 0.0, "pending_total": 0.0}
        rewards = np.array([task.reward for task in tasks], dtype=float)
        return {
            "mean": float(np.mean(rewards)),
            "max": float(np.max(rewards)),
            "min": float(np.min(rewards)),
            "pending_total": float(np.sum(rewards)),
        }

    def task_deadline_urgency(self) -> list[float]:

        urgencies: list[float] = []
        for task in self.env.unmatched_tasks:
            urgency = 1.0 - (task.deadline - self.now_time) / (task.deadline - task.published_time)
            urgencies.append(urgency)
        return urgencies

    def num_unmatched_subtasks(self) -> int:

        return len(self.env.unmatched_subtasks)

    def subtask_dependency_depth_stats(self) -> dict[str, Any]:
        subtasks = self.env.unmatched_subtasks
        if not subtasks:
            return {"mean_depth": 0.0, "max_depth": 0, "depth_distribution": {}}

        depths = [int(sub.depth or 0) for sub in subtasks]
        depth_distribution: dict[str, int] = {}
        for depth in depths:
            key = str(depth)
            depth_distribution[key] = depth_distribution.get(key, 0) + 1
        return {
            "mean_depth": float(np.mean(depths)),
            "max_depth": int(max(depths)),
            "depth_distribution": depth_distribution,
        }

    def subtask_blocking_ratio(self) -> float:

        subtasks = self.env.unmatched_subtasks
        if not subtasks:
            return 0.0
        blocked_num = sum(
            1 for sub in subtasks if any(not pre.is_matched() for pre in sub.pre_subtasks)
        )
        return blocked_num / len(subtasks)

    def subtask_skill_demand_stats(self) -> dict[int, int]:

        demand: dict[int, int] = {}
        for sub in self.env.unmatched_subtasks:
            for skill in sub.needed_skills:
                demand[skill] = demand.get(skill, 0) + 1
        return demand

    def num_idle_workers(self) -> int:

        return len(self.env.idle_workers)

    def worker_skill_coverage_stats(self) -> dict[int, float]:

        workers = self.env.idle_workers
        coverage: dict[int, int] = {skill: 0 for skill in range(ModelConfig.nl_skill_nums())}
        for worker in workers:
            for skill in worker.skills:
                if skill in coverage:
                    coverage[skill] += 1

        if not workers:
            return {skill: 0.0 for skill in coverage}
        return {skill: count / len(workers) for skill, count in coverage.items()}

    def worker_service_range_stats(self) -> dict[str, Any]:
        workers = self.env.idle_workers
        if not workers:
            return {"mean_range": 0.0, "range_distribution": {}}
        ranges = np.array([worker.max_distance for worker in workers], dtype=float)
        return {
            "mean_range": float(np.mean(ranges)),
            "range_distribution": {
                "std": float(np.std(ranges)),
                "min": float(np.min(ranges)),
                "max": float(np.max(ranges)),
            },
        }

    def skill_supply_demand_ratio(self) -> dict[int, float]:

        
        supply: dict[int, int] = {skill: 0 for skill in range(ModelConfig.nl_skill_nums())}
        for worker in self.env.idle_workers:
            for skill in worker.skills:
                if skill in supply:
                    supply[skill] += 1

        demand = self.subtask_skill_demand_stats()
        ratios: dict[int, float] = {}
        for skill in range(ModelConfig.nl_skill_nums()):
            demand_count = demand.get(skill, 0)
            if demand_count == 0:
                
                ratios[skill] = 0.0
            else:
                ratios[skill] = supply.get(skill, 0) / demand_count
        return ratios

    def scarce_skills(self, threshold: float = 0.5) -> list[int]:
        ratios = self.skill_supply_demand_ratio()
        return [skill for skill, ratio in ratios.items() if 0 < ratio < threshold]

    def subtask_worker_distance_stats(self) -> dict[str, Any]:
        distances: list[float] = []
        for sub in self.env.unmatched_subtasks:
            for worker in self.env.idle_workers:
                if worker.constraints_ok(sub, self.now_time):
                    distances.append(distance_util.shortest_distance(sub.location, worker.location))

        if not distances:
            return {"mean_distance": 0.0, "distance_percentiles": {}}

        arr = np.array(distances, dtype=float)
        return {
            "mean_distance": float(np.mean(arr)),
            "distance_percentiles": {
                "p25": float(np.percentile(arr, 25)),
                "p50": float(np.percentile(arr, 50)),
                "p75": float(np.percentile(arr, 75)),
                "p90": float(np.percentile(arr, 90)),
            },
        }

    def spatial_dispersion(self) -> float:
        subtasks = self.env.unmatched_subtasks
        if len(subtasks) < 2:
            return 0.0

        lngs = np.array([sub.location[0] for sub in subtasks], dtype=float)
        lats = np.array([sub.location[1] for sub in subtasks], dtype=float)
        lng_std = np.std(lngs) / 180.0
        lat_std = np.std(lats) / 90.0
        return float(np.sqrt(lng_std**2 + lat_std**2))

    def time_until_next_deadline(self) -> int:
        if not self.env.unmatched_tasks:
            return 0
        nearest_deadline = min(task.deadline for task in self.env.unmatched_tasks)
        return max(0, nearest_deadline - self.now_time)

    def avg_time_pressure(self) -> float:

        subtasks = self.env.unmatched_subtasks
        if not subtasks:
            return 0.0

        reference_time = max(1, ModelConfig.nl_time())
        pressures = [
            max(0.0, min(1.0, 1.0 - (sub.sub_deadline - self.now_time) / reference_time))
            for sub in subtasks
            if sub.sub_deadline is not None
        ]
        if not pressures:
            return 0.0
        return float(np.mean(pressures))

    @staticmethod
    def prompt_example() -> dict[str, Any]:
        return {
            "_schema_description": _query_schema_description(),
            "methods": {
                "num_unmatched_tasks": {
                    "function_definition": "num_unmatched_tasks() -> int",
                    "inputs": {},
                    "outputs": {"type": "int", "structure": "scalar count", "meaning": "Number of currently unfinished tasks."},
                    "meaning": "Measure the current global pending task load.",
                },
                "num_matched_tasks": {
                    "function_definition": "num_matched_tasks() -> int",
                    "inputs": {},
                    "outputs": {"type": "int", "structure": "scalar count", "meaning": "Number of tasks whose subtasks are all matched."},
                    "meaning": "Measure global task completion progress.",
                },
                "task_reward_distribution": {
                    "function_definition": "task_reward_distribution() -> dict[str, float]",
                    "inputs": {},
                    "outputs": {
                        "type": "dict[str, float]",
                        "structure": {"mean": "float", "max": "float", "min": "float", "pending_total": "float"},
                        "meaning": "Reward distribution over unfinished tasks.",
                    },
                    "meaning": "Computation: compute mean/min/max/sum of rewards over all unfinished tasks; returns floats where mean/max/min share the same units as individual task rewards, and pending_total is the sum of all unfinished task rewards. Meaning: measures the scale and concentration of pending-task rewards; larger values indicate higher potential gains.",
                },
                "task_deadline_urgency": {
                    "function_definition": "task_deadline_urgency() -> list[float]",
                    "inputs": {},
                    "outputs": {"type": "list[float]", "structure": "one urgency value per unmatched task", "meaning": "Normalized deadline urgency values."},
                    "meaning": "Computation: for each unfinished task compute urgency normalized to [0,1] as 1 - (deadline - now)/(deadline - published_time); returns a list of floats in [0,1]. Meaning: values closer to 1 indicate tasks nearer to their deadline and of higher priority.",
                },
                "num_unmatched_subtasks": {
                    "function_definition": "num_unmatched_subtasks() -> int",
                    "inputs": {},
                    "outputs": {"type": "int", "structure": "scalar count", "meaning": "Number of currently unmatched subtasks."},
                    "meaning": "Computation: returns the count of unmatched subtasks; return range: non-negative integer. Meaning: measures the pending workload at the subtask level.",
                },
                "subtask_dependency_depth_stats": {
                    "function_definition": "subtask_dependency_depth_stats() -> dict[str, Any]",
                    "inputs": {},
                    "outputs": {
                        "type": "dict[str, Any]",
                        "structure": {"mean_depth": "float", "max_depth": "int", "depth_distribution": "dict[str, int]"},
                        "meaning": "Depth distribution of unmatched subtasks in dependency DAGs.",
                    },
                    "meaning": "Computation: collect depths of unmatched subtasks and compute mean depth, maximum depth, and a frequency distribution mapping depth->count; return ranges: mean_depth is float, max_depth is a non-negative integer, depth_distribution is a dict. Meaning: evaluates hierarchical pressure in the dependency DAG; higher mean or max implies deeper chains and potentially stronger dependency delays.",
                },
                "subtask_blocking_ratio": {
                    "function_definition": "subtask_blocking_ratio() -> float",
                    "inputs": {},
                    "outputs": {"type": "float", "structure": "ratio in [0, 1]", "meaning": "Share of unmatched subtasks blocked by unmatched predecessors."},
                    "meaning": "Computation: proportion of unmatched subtasks that are blocked by unfinished predecessors (blocked_num/total); returns a float in [0,1]. Meaning: values closer to 1 indicate widespread dependency blocking that hinders parallel scheduling.",
                },
                "subtask_skill_demand_stats": {
                    "function_definition": "subtask_skill_demand_stats() -> dict[int, int]",
                    "inputs": {},
                    "outputs": {"type": "dict[int, int]", "structure": "key=skill_id, value=demand_count", "meaning": "Demand count for each required skill."},
                    "meaning": "Computation: tally required skills across unmatched subtasks into a dictionary {skill_id: count}; return values are non-negative integers. Meaning: reflects instantaneous demand intensity per skill; higher counts indicate more tasks requiring that skill.",
                },
                "num_idle_workers": {
                    "function_definition": "num_idle_workers() -> int",
                    "inputs": {},
                    "outputs": {"type": "int", "structure": "scalar count", "meaning": "Number of currently idle workers."},
                    "meaning": "Computation: return the length of the current idle_workers list; return range: non-negative integer. Meaning: represents the number of workers immediately available for assignment and indicates supply capacity.",
                },
                "worker_skill_coverage_stats": {
                    "function_definition": "worker_skill_coverage_stats() -> dict[int, float]",
                    "inputs": {},
                    "outputs": {"type": "dict[int, float]", "structure": "key=skill_id, value=idle_worker_coverage_ratio", "meaning": "Coverage ratio of each skill among idle workers."},
                    "meaning": "Computation: for each skill compute the ratio of idle workers who possess that skill to the total number of idle workers; return values are in [0,1]. Meaning: indicates coverage of each skill among idle labor; values near 1 mean the skill is widespread.",
                },
                "worker_service_range_stats": {
                    "function_definition": "worker_service_range_stats() -> dict[str, Any]",
                    "inputs": {},
                    "outputs": {
                        "type": "dict[str, Any]",
                        "structure": {"mean_range": "float", "range_distribution": {"std": "float", "min": "float", "max": "float"}},
                        "meaning": "Distribution of idle workers' service ranges.",
                    },
                    "meaning": "Computation: compute the mean of idle workers' max service distances and summarize distribution statistics (std/min/max); returns floats in distance units. Meaning: measures spatial reach of idle workers; a larger mean indicates a broader overall coverage.",
                },
                "skill_supply_demand_ratio": {
                    "function_definition": "skill_supply_demand_ratio() -> dict[int, float]",
                    "inputs": {},
                    "outputs": {"type": "dict[int, float]", "structure": "key=skill_id, value=supply_divided_by_demand", "meaning": "Finite supply-demand ratio for each skill."},
                    "meaning": "Computation: compute supply/demand ratio per skill; when demand=0 return 0.0 to ensure finite values; return range: non-negative floats (0.0 indicates no demand or no supply). Meaning: ratio <1 indicates shortage, ~1 indicates balance, larger values indicate supply exceeds demand.",
                },
                "scarce_skills": {
                    "function_definition": "scarce_skills(threshold: float = 0.5) -> list[int]",
                    "inputs": {"threshold": {"type": "float", "default": 0.5, "meaning": "Skills with 0 < supply-demand ratio < threshold are treated as scarce."}},
                    "outputs": {"type": "list[int]", "structure": "skill_id list", "meaning": "Skill ids considered scarce under the threshold."},
                    "meaning": "Computation: select skill ids whose supply/demand ratio lies in (0, threshold); return: list of skill ids. Meaning: identifies globally scarce skills for prioritization or pricing strategies.",
                },
                "subtask_worker_distance_stats": {
                    "function_definition": "subtask_worker_distance_stats() -> dict[str, Any]",
                    "inputs": {},
                    "outputs": {
                        "type": "dict[str, Any]",
                        "structure": {"mean_distance": "float", "distance_percentiles": {"p25": "float", "p50": "float", "p75": "float", "p90": "float"}},
                        "meaning": "Distance distribution over feasible subtask-worker pairs.",
                    },
                    "meaning": "Computation: for all feasible subtask-worker pairs (constraints_ok) compute shortest-path distances and derive mean and percentile statistics; returns floats in distance units and a percentiles dictionary. Meaning: reflects spatial proximity between tasks and workers; high mean or high percentiles indicate spatial matching difficulty.",
                },
                "spatial_dispersion": {
                    "function_definition": "spatial_dispersion() -> float",
                    "inputs": {},
                    "outputs": {"type": "float", "structure": "normalized scalar", "meaning": "Spatial dispersion score of unmatched subtasks."},
                    "meaning": "Computation: compute standard deviations of longitudes and latitudes of unmatched subtasks, normalize by geographic ranges (180/90) and combine into a scalar; return 0.0 when fewer than 2 subtasks. Return range: non-negative float. Meaning: larger values indicate more geographically dispersed tasks, affecting centralized dispatch strategies.",
                },
                "time_until_next_deadline": {
                    "function_definition": "time_until_next_deadline() -> int",
                    "inputs": {},
                    "outputs": {"type": "int", "structure": "non-negative seconds", "meaning": "Remaining time until the nearest task deadline."},
                    "meaning": "Computation: find the earliest deadline among unfinished tasks and return max(0, deadline - now) in seconds; return range: non-negative integer seconds. Meaning: smaller values indicate imminent deadlines requiring immediate scheduling.",
                },
                "avg_time_pressure": {
                    "function_definition": "avg_time_pressure() -> float",
                    "inputs": {},
                    "outputs": {"type": "float", "structure": "ratio in [0, 1]", "meaning": "Average normalized time pressure over unmatched subtasks."},
                    "meaning": "Computation: for each unmatched subtask compute time pressure as 1 - (sub_deadline - now)/reference_time clamped to [0,1], then take the mean; return range: float in [0,1]. Meaning: values closer to 1 indicate overall subtasks are nearer to deadlines and overall urgency is higher.",
                },
            },
        }


class ObservationQuerySet:
    

    def __init__(self, env: CrowdsourcingEnv, subtask: SubTask):

        self.env = env
        self.subtask = subtask

    @property
    def now_time(self) -> int:
        
        return self.env.now_time

    def subtask_attributes(self) -> dict[str, Any]:
        
        return {
            "id": self.subtask.id,
            "needed_skills": sorted(list(self.subtask.needed_skills)),
            "num_needed_skills": len(self.subtask.needed_skills),
            "duration": self.subtask.duration,
            "depth": int(self.subtask.depth or 0),
            "is_root": len(self.subtask.pre_subtasks) == 0,
            "is_leaf": len(self.subtask.post_subtasks) == 0,
        }

    def subtask_location(self) -> tuple[float, float]:
        
        return self.subtask.location

    def dependency_context(self) -> dict[str, Any]:
        
        task = self.subtask.task
        unmatched_pre = [pre for pre in self.subtask.pre_subtasks if not pre.is_matched()]
        unmatched_post = [post for post in self.subtask.post_subtasks if not post.is_matched()]

        is_on_critical_path = False
        if task is not None:
            ancestors = task.subtask_ancestors.get(self.subtask, set())
            descendants = task.subtask_descendants.get(self.subtask, set())
            is_on_critical_path = len(ancestors) > 0 and len(descendants) > 0

        return {
            "num_predecessors": len(self.subtask.pre_subtasks),
            "num_successors": len(self.subtask.post_subtasks),
            "num_unmatched_predecessors": len(unmatched_pre),
            "num_unmatched_successors": len(unmatched_post),
            "predecessor_depths": [int(pre.depth or 0) for pre in self.subtask.pre_subtasks],
            "successor_depths": [int(post.depth or 0) for post in self.subtask.post_subtasks],
            "is_on_critical_path": is_on_critical_path,
        }

    def dependency_chain_length(self) -> dict[str, int]:
        
        task = self.subtask.task
        if task is None:
            return {
                "longest_chain_from_root": 0,
                "longest_chain_to_leaf": 0,
                "total_chain_length": 0,
            }

        descendants = task.subtask_descendants.get(self.subtask, set())
        chain_from_root = int(self.subtask.depth or 0)
        max_depth_in_descendants = max([int(self.subtask.depth or 0)] + [int(ds.depth or 0) for ds in descendants])
        chain_to_leaf = max_depth_in_descendants - chain_from_root

        return {
            "longest_chain_from_root": chain_from_root,
            "longest_chain_to_leaf": chain_to_leaf,
            "total_chain_length": chain_from_root + chain_to_leaf,
        }

    def _available_workers(self) -> list[Worker]:
        
        return [
            worker
            for worker in self.env.idle_workers
            if worker.constraints_ok(self.subtask, self.now_time)
        ]

    def available_workers_summary(self) -> dict[str, Any]:
        
        available_workers = self._available_workers()
        if not available_workers:
            return {
                "num_available_workers": 0,
                "worker_skill_match_ratios": [],
                "worker_distance_ratios": [],
                "worker_velocity_stats": {"mean": 0.0, "std": 0.0},
            }

        skill_match_ratios: list[float] = []
        distance_ratios: list[float] = []
        velocities: list[float] = []

        for worker in available_workers:
            denominator = max(1, len(worker.skills))
            skill_match_ratios.append(len(self.subtask.needed_skills) / denominator)

            distance = distance_util.shortest_distance(self.subtask.location, worker.location)
            max_distance = max(worker.max_distance, 1e-8)
            distance_ratios.append(distance / max_distance)
            velocities.append(worker.velocity)

        return {
            "num_available_workers": len(available_workers),
            "worker_skill_match_ratios": skill_match_ratios,
            "worker_distance_ratios": distance_ratios,
            "worker_velocity_stats": {
                "mean": float(np.mean(velocities)),
                "std": float(np.std(velocities)),
            },
        }

    def worker_competition_level(self) -> float:
        
        competing_subtasks = [
            other
            for other in self.env.unmatched_subtasks
            if other != self.subtask and bool(other.needed_skills & self.subtask.needed_skills)
        ]
        if not competing_subtasks:
            return 0.0

        available_workers = self._available_workers()
        return len(competing_subtasks) / (len(competing_subtasks) + len(available_workers) + 1e-8)

    def skill_feasibility(self) -> dict[str, Any]:
        
        available_workers = self._available_workers()
        workers_with_full_skill = sum(
            1 for worker in available_workers if self.subtask.needed_skills.issubset(worker.skills)
        )
        total_workers = len(self.env.idle_workers)
        coverage_ratio = workers_with_full_skill / total_workers if total_workers > 0 else 0.0

        state_query = StateQuerySet(self.env)
        ratios = state_query.skill_supply_demand_ratio()
        is_skill_scarce = any(ratios.get(skill, float("inf")) < 0.5 for skill in self.subtask.needed_skills)

        skill_gap_analysis: dict[int, dict[str, Any]] = {}
        for skill in self.subtask.needed_skills:
            supply = sum(1 for worker in available_workers if skill in worker.skills)
            skill_gap_analysis[skill] = {
                "supply": supply,
                "is_scarce": supply < 2,
            }

        return {
            "skill_coverage_ratio": coverage_ratio,
            "is_skill_scarce": is_skill_scarce,
            "skill_gap_analysis": skill_gap_analysis,
        }

    def temporal_feasibility(self) -> dict[str, Any]:
        
        time_until_deadline = max(0, (self.subtask.sub_deadline or self.now_time) - self.now_time)
        reference_time = max(1, ModelConfig.nl_time())

        if self.subtask.pre_complete_time is not None and self.subtask.sub_deadline is not None:
            predecessor_completion_margin = (
                self.subtask.sub_deadline - self.subtask.pre_complete_time - self.subtask.duration
            )
        else:
            predecessor_completion_margin = time_until_deadline - self.subtask.duration

        is_time_critical = time_until_deadline < reference_time * 0.2
        is_tight_schedule = time_until_deadline < self.subtask.duration * 2

        return {
            "time_until_deadline": time_until_deadline,
            "time_until_sub_deadline": time_until_deadline,
            "is_time_critical": is_time_critical,
            "predecessor_completion_margin": predecessor_completion_margin,
            "is_tight_schedule": is_tight_schedule,
        }

    @staticmethod
    def prompt_example() -> dict[str, Any]:
        
        return {
            "_schema_description": _query_schema_description(),
            "methods": {
                "subtask_attributes": {
                    "function_definition": "subtask_attributes() -> dict[str, Any]",
                    "inputs": {},
                    "outputs": {
                        "type": "dict[str, Any]",
                        "structure": {
                            "id": "str",
                            "needed_skills": "list[int]",
                            "num_needed_skills": "int",
                            "duration": "int",
                            "depth": "int",
                            "is_root": "bool",
                            "is_leaf": "bool",
                        },
                        "meaning": "Static and structural attributes of the current subtask.",
                    },
                    "meaning": "Describe the target subtask's basic demand and DAG position.",
                },
                "subtask_location": {
                    "function_definition": "subtask_location() -> tuple[float, float]",
                    "inputs": {},
                    "outputs": {"type": "tuple[float, float]", "structure": "(lng, lat)", "meaning": "Geographic coordinates of the current subtask."},
                    
                    "meaning": "Computation: directly return the subtask's geographic coordinates as (lng, lat); return range: a pair of floats. Meaning: used for spatial reasoning, matching and routing; these coordinates indicate the task location.",
                },
                "dependency_context": {
                    "function_definition": "dependency_context() -> dict[str, Any]",
                    "inputs": {},
                    "outputs": {
                        "type": "dict[str, Any]",
                        "structure": {
                            "num_predecessors": "int",
                            "num_successors": "int",
                            "num_unmatched_predecessors": "int",
                            "num_unmatched_successors": "int",
                            "predecessor_depths": "list[int]",
                            "successor_depths": "list[int]",
                            "is_on_critical_path": "bool",
                        },
                        "meaning": "Immediate dependency context around the current subtask.",
                    },
                    
                    "meaning": "Computation: count direct predecessors/successors and unmatched predecessors/successors, list predecessor and successor depths, and determine if the node lies on a critical path (based on presence of both ancestors and descendants); returns integers, booleans and depth lists. Meaning: assesses the task's local dependency position and importance—critical path nodes often warrant higher priority.",
                },
                "dependency_chain_length": {
                    "function_definition": "dependency_chain_length() -> dict[str, int]",
                    "inputs": {},
                    "outputs": {
                        "type": "dict[str, int]",
                        "structure": {"longest_chain_from_root": "int", "longest_chain_to_leaf": "int", "total_chain_length": "int"},
                        "meaning": "Estimated chain length information around the current subtask.",
                    },
                    
                    "meaning": "Computation: using the task's depth data compute the longest chain length from the root to the current node, the longest chain from the current node to the deepest descendant, and the total chain length; returns non-negative integers. Meaning: indicates how deep or central the subtask is within its dependency chain.",
                },
                "available_workers_summary": {
                    "function_definition": "available_workers_summary() -> dict[str, Any]",
                    "inputs": {},
                    "outputs": {
                        "type": "dict[str, Any]",
                        "structure": {
                            "num_available_workers": "int",
                            "worker_skill_match_ratios": "list[float]",
                            "worker_distance_ratios": "list[float]",
                            "worker_velocity_stats": {"mean": "float", "std": "float"},
                        },
                        "meaning": "Statistics for idle workers currently feasible for this subtask.",
                    },
                    
                    "meaning": "Computation: filter idle workers that meet constraints and compute their count, skill_match_ratios normalized by worker skill counts, distance ratios (distance/max_service_range), and velocity stats (mean/std); returns integers and float arrays. Meaning: evaluates the availability and quality of candidate workers—higher num_available_workers means more choices; lower distance_ratios mean closer workers.",
                },
                "worker_competition_level": {
                    "function_definition": "worker_competition_level() -> float",
                    "inputs": {},
                    "outputs": {"type": "float", "structure": "ratio near [0, 1]", "meaning": "Competition pressure for workers relevant to this subtask."},
                    
                    "meaning": "Computation: count other unmatched subtasks that share skills with the current subtask as competitors and compute competition intensity as competing_subtasks/(competing_subtasks + available_workers); returns a float approximately in [0,1]. Meaning: higher values indicate stronger contention for resources and greater difficulty in matching.",
                },
                "skill_feasibility": {
                    "function_definition": "skill_feasibility() -> dict[str, Any]",
                    "inputs": {},
                    "outputs": {
                        "type": "dict[str, Any]",
                        "structure": {
                            "skill_coverage_ratio": "float",
                            "is_skill_scarce": "bool",
                            "skill_gap_analysis": "dict[int, dict[str, Any]] with supply:int and is_scarce:bool",
                        },
                        "meaning": "Skill availability and scarcity information for this subtask.",
                    },
                    
                    "meaning": "Computation: compute coverage_ratio as the number of available workers who fully satisfy the subtask's skills divided by total idle workers; use global supply-demand ratios to detect scarce skills; provide per-skill supply and an is_scarce flag. Returns floats, booleans and a dict. Meaning: assesses whether skill supply meets demand—higher coverage_ratio implies easier matching to fully qualified workers.",
                },
                "temporal_feasibility": {
                    "function_definition": "temporal_feasibility() -> dict[str, Any]",
                    "inputs": {},
                    "outputs": {
                        "type": "dict[str, Any]",
                        "structure": {
                            "time_until_deadline": "int",
                            "time_until_sub_deadline": "int",
                            "is_time_critical": "bool",
                            "predecessor_completion_margin": "int",
                            "is_tight_schedule": "bool",
                        },
                        "meaning": "Deadline and schedule feasibility for this subtask.",
                    },
                    
                    "meaning": "Computation: compute remaining seconds until the subtask deadline, compare with a reference time to determine time criticality; compute predecessor_completion_margin (time margin after predecessors complete); determine is_tight_schedule based on duration. Returns integers and booleans. Meaning: evaluates whether the subtask requires priority scheduling (true values indicate higher urgency).",
                },
            },
        }



class ContributionQuerySet:
    

    def __init__(self, context: ContributionContext, subtask: SubTask):
        
        self._context = context
        self.subtask = subtask

        
        self._step_subtasks: list[SubTask] = context.subtasks
        self._idle_workers: list[Worker] = context.idle_workers
        self._actions: dict[str, int] = context.actions_by_subtask_id
        self._activated_ids: set[str] = context.activated_subtask_ids
        self._formal_ids: set[str] = context.formal_subtask_ids
        self._candidate_ids: set[str] = context.candidate_subtask_ids
        self._match_details: dict[str, MatchDetailSnapshot] = context.match_details
        self._completed_task_ids: set[str] = context.completed_task_ids
        self._task_matched_before: dict[str, int] = context.task_matched_before
        self._step_revenue: float = context.revenue

        
        self._num_formal = len(self._formal_ids)
        self._num_activated = len(self._activated_ids)
        self._num_subtasks = len(self._step_subtasks)

        
        self._total_utility = sum(d.utility for d in self._match_details.values())
        
        self._total_travel_cost = sum(d.travel_cost for d in self._match_details.values())
        
        self._total_matched_est_reward = sum(
            d.subtask.estimated_reward() for d in self._match_details.values()
        )
        
        self._consumed_worker_ids: set[str] = {
            d.worker.id for d in self._match_details.values()
        }
        self._worker_skill_supply = self._worker_skill_supply_from_workers(self._idle_workers)
        self._subtask_skill_demand = self._subtask_skill_demand_from_subtasks(self._step_subtasks)
        self._skill_supply_demand_ratio = self._skill_supply_demand_ratio_from_counts(
            self._worker_skill_supply,
            self._subtask_skill_demand,
        )

    @property
    def now_time(self) -> int:
        return self._context.now_time

    @staticmethod
    def _worker_skill_supply_from_workers(workers: list[Worker]) -> dict[int, int]:
        supply: dict[int, int] = {}
        for worker in workers:
            for skill in worker.skills:
                supply[skill] = supply.get(skill, 0) + 1
        return supply

    @staticmethod
    def _subtask_skill_demand_from_subtasks(subtasks: list[SubTask]) -> dict[int, int]:
        demand: dict[int, int] = {}
        for subtask in subtasks:
            for skill in subtask.needed_skills:
                demand[skill] = demand.get(skill, 0) + 1
        return demand

    @staticmethod
    def _skill_supply_demand_ratio_from_counts(
        supply: dict[int, int],
        demand: dict[int, int],
    ) -> dict[int, float]:
        ratios: dict[int, float] = {}
        all_skills = set(supply.keys()) | set(demand.keys())
        for skill in all_skills:
            s = supply.get(skill, 0)
            d = demand.get(skill, 0)
            ratios[skill] = 0.0 if d == 0 else s / d
        return ratios

    
    
    
    def step_allocation_summary(self) -> dict[str, Any]:
        
        return {
            "num_subtasks": self._num_subtasks,
            "num_activated": self._num_activated,
            "num_candidate": len(self._candidate_ids),
            "num_formal": self._num_formal,
            "activation_rate": self._num_activated / max(1, self._num_subtasks),
            "match_success_rate": (
                self._num_formal / max(1, self._num_activated)
            ),
            "step_revenue": self._step_revenue,
            "num_completed_tasks": len(self._completed_task_ids),
            "total_utility": self._total_utility,
            "total_travel_cost": self._total_travel_cost,
        }

    
    
    
    def immediate_reward_info(self) -> dict[str, Any]:
        
        estimated_reward = self.subtask.estimated_reward()
        action = self._actions.get(self.subtask.id, 0)
        is_matched = self.subtask.id in self._formal_ids

        if action == 0:
            return {
                "is_activated": False,
                "is_matched": False,
                "immediate_reward": 0.0,
                "estimated_reward": estimated_reward,
                "reward_ratio": 0.0,
            }

        if is_matched:
            return {
                "is_activated": True,
                "is_matched": True,
                "immediate_reward": estimated_reward,
                "estimated_reward": estimated_reward,
                "reward_ratio": 1.0,
            }

        penalty = estimated_reward * ModelConfig.penalty_unmatched
        return {
            "is_activated": True,
            "is_matched": False,
            "immediate_reward": -penalty,
            "estimated_reward": estimated_reward,
            "reward_ratio": (-penalty / estimated_reward) if estimated_reward != 0 else 0.0,
        }

    
    
    
    def relative_contribution(self) -> dict[str, float]:
        
        detail = self._match_details.get(self.subtask.id)
        task = self.subtask.task

        if detail is None:
            return {
                "utility_share": 0.0,
                "reward_share": 0.0,
                "travel_cost_share": 0.0,
                "task_progress_delta": 0.0,
                "is_task_completing": 0.0,
            }

        utility_share = detail.utility / self._total_utility if self._total_utility > 0 else 0.0
        reward_share = (
            self.subtask.estimated_reward() / self._total_matched_est_reward
            if self._total_matched_est_reward > 0 else 0.0
        )
        travel_cost_share = (
            detail.travel_cost / self._total_travel_cost
            if self._total_travel_cost > 0 else 0.0
        )

        total_subs = max(1, len(task.subtasks))
        before = self._task_matched_before.get(task.id, 0)
        after = sum(1 for s in task.subtasks if s.is_matched())
        task_progress_delta = (after - before) / total_subs

        is_task_completing = task.id in self._completed_task_ids

        return {
            "utility_share": utility_share,
            "reward_share": reward_share,
            "travel_cost_share": travel_cost_share,
            "task_progress_delta": task_progress_delta,
            "is_task_completing": float(is_task_completing),
        }

    
    
    
    def revenue_impact(self) -> dict[str, float]:
        
        task = self.subtask.task
        action = self._actions.get(self.subtask.id, 0)
        is_matched = self.subtask.id in self._formal_ids

        if task is None:
            return {"potential_reward": 0.0, "reward_risk": 1.0, "expected_value": 0.0,
                    "revenue_contribution_ratio": 0.0}

        estimated_reward = self.subtask.estimated_reward()
        total_subtasks = max(1, len(task.subtasks))
        matched_subtasks = sum(1 for s in task.subtasks if s.is_matched())
        remaining_ratio = (total_subtasks - matched_subtasks) / total_subtasks

        if is_matched:
            completion_prob = 0.5 + 0.3 * (1 - remaining_ratio)
        else:
            completion_prob = 0.2 * (1 - remaining_ratio)

        potential_reward = estimated_reward if action == 1 else 0.0
        expected_value = estimated_reward * completion_prob if action == 1 else 0.0

        if self._step_revenue > 0 and task.id in self._completed_task_ids:
            revenue_contribution_ratio = (task.reward / total_subtasks) / self._step_revenue
        else:
            revenue_contribution_ratio = 0.0

        return {
            "potential_reward": potential_reward,
            "reward_risk": 1.0 - completion_prob,
            "expected_value": expected_value,
            "revenue_contribution_ratio": revenue_contribution_ratio,
        }

    
    
    
    def dependency_impact(self) -> dict[str, Any]:
        
        task = self.subtask.task
        is_matched = self.subtask.id in self._formal_ids
        step_matched_ids = self._formal_ids

        successors_unlocked = []
        if is_matched:
            for succ in self.subtask.post_subtasks:
                if all(
                    pre.is_matched() or pre.id in step_matched_ids
                    for pre in succ.pre_subtasks
                ):
                    successors_unlocked.append(succ)

        is_critical_node = False
        if task is not None:
            ancestors = task.subtask_ancestors.get(self.subtask, set())
            descendants = task.subtask_descendants.get(self.subtask, set())
            is_critical_node = len(ancestors) > 0 and len(descendants) > 0

        total_successors = len(self.subtask.post_subtasks)
        dependency_relief_score = (
            len(successors_unlocked) / total_successors if total_successors > 0 else 0.0
        )

        step_dependency_chain_progress = 0.0
        if task is not None:
            step_depths = [
                int(s.depth or 0)
                for s in task.subtasks
                if s.id in step_matched_ids
            ]
            if step_depths:
                step_dependency_chain_progress = float(max(step_depths) - min(step_depths))

        blocks_others = False
        detail = self._match_details.get(self.subtask.id)
        if is_matched and detail is not None:
            matched_worker = detail.worker
            for other_sub in self._step_subtasks:
                if other_sub.id == self.subtask.id or other_sub.id not in self._activated_ids:
                    continue
                if not (other_sub.needed_skills & self.subtask.needed_skills):
                    continue
                alternatives = [
                    w for w in self._idle_workers
                    if w.id != matched_worker.id
                    and w.id not in self._consumed_worker_ids
                    and w.constraints_ok(other_sub, self.now_time)
                ]
                if not alternatives:
                    blocks_others = True
                    break

        return {
            "unlocks_successors": len(successors_unlocked) > 0,
            "num_successors_unlocked": len(successors_unlocked),
            "is_critical_node": is_critical_node,
            "dependency_relief_score": dependency_relief_score,
            "blocks_others": blocks_others,
            "step_dependency_chain_progress": step_dependency_chain_progress,
        }

    
    
    
    def step_resource_competition(self) -> dict[str, Any]:
        
        num_idle = max(1, len(self._idle_workers))
        worker_contention_ratio = len(self._consumed_worker_ids) / num_idle
        subtask_per_worker = self._num_activated / num_idle

        other_skills: set[int] = set()
        for sid, detail in self._match_details.items():
            if sid != self.subtask.id:
                other_skills |= detail.subtask.needed_skills
        if other_skills or self.subtask.needed_skills:
            intersection = len(self.subtask.needed_skills & other_skills)
            union = len(self.subtask.needed_skills | other_skills)
            skill_overlap = intersection / max(1, union)
        else:
            skill_overlap = 0.0

        is_worker_scarce = self._num_formal > num_idle * 0.8

        return {
            "worker_contention_ratio": worker_contention_ratio,
            "subtask_per_worker": subtask_per_worker,
            "skill_overlap_with_step": skill_overlap,
            "is_worker_scarce_in_step": float(is_worker_scarce),
        }

    
    
    
    def resource_utilization(self) -> dict[str, float]:
        
        detail = self._match_details.get(self.subtask.id)
        if detail is None:
            return {
                "matched_worker_skill_surplus": 0.0,
                "matched_worker_distance_efficiency": 0.0,
                "travel_cost": 0.0,
                "utilization_score": 0.0,
                "relative_efficiency": 0.0,
            }

        worker = detail.worker
        skill_surplus = float(len(worker.skills) - len(self.subtask.needed_skills))
        distance = distance_util.shortest_distance(self.subtask.location, worker.location)
        distance_efficiency = max(0.0, 1.0 - distance / max(worker.max_distance, 1e-8))
        travel_cost = float(detail.travel_cost)

        utilization_score = (
            0.5 * distance_efficiency
            + 0.3 * (1.0 - skill_surplus / (len(worker.skills) + 1e-8))
            + 0.2 * (1.0 - min(1.0, travel_cost / 10.0))
        )

        if self._num_formal > 0:
            all_scores = []
            for sid, d in self._match_details.items():
                w = d.worker
                s = d.subtask
                dist = distance_util.shortest_distance(s.location, w.location)
                de = max(0.0, 1.0 - dist / max(w.max_distance, 1e-8))
                ss = float(len(w.skills) - len(s.needed_skills))
                tc = float(d.travel_cost)
                sc = 0.5 * de + 0.3 * (1.0 - ss / (len(w.skills) + 1e-8)) + 0.2 * (1.0 - min(1.0, tc / 10.0))
                all_scores.append(sc)
            avg_score = sum(all_scores) / len(all_scores)
            relative_efficiency = utilization_score / max(avg_score, 1e-8)
        else:
            relative_efficiency = 0.0

        return {
            "matched_worker_skill_surplus": skill_surplus,
            "matched_worker_distance_efficiency": distance_efficiency,
            "travel_cost": travel_cost,
            "utilization_score": float(utilization_score),
            "relative_efficiency": float(relative_efficiency),
        }

    
    
    
    def skill_resource_impact(self) -> dict[str, Any]:
        
        detail = self._match_details.get(self.subtask.id)
        if detail is None:
            return {
                "skill_depletion": {},
                "is_scarce_skill_used": False,
                "resource_substitutability": 0.0,
                "step_skill_pressure": 0.0,
            }

        worker = detail.worker
        ratios = self._skill_supply_demand_ratio
        skill_depletion: dict[int, float] = {}
        for skill in self.subtask.needed_skills:
            ratio = ratios.get(skill, float("inf"))
            depletion = 1.0 / (ratio + 1e-8) if ratio < float("inf") else 0.0
            skill_depletion[skill] = min(1.0, float(depletion))

        is_scarce_skill_used = any(
            ratios.get(skill, float("inf")) < 0.5 for skill in self.subtask.needed_skills
        )

        substitutes = sum(
            1 for other in self._idle_workers
            if other.id != worker.id
            and other.id not in self._consumed_worker_ids
            and other.constraints_ok(self.subtask, self.now_time)
        )
        resource_substitutability = substitutes / max(1, len(self._idle_workers))

        all_consumed_skills: list[int] = []
        for d in self._match_details.values():
            all_consumed_skills.extend(d.subtask.needed_skills)
        if all_consumed_skills:
            scarce_count = sum(1 for sk in all_consumed_skills if ratios.get(sk, float("inf")) < 1.0)
            step_skill_pressure = scarce_count / len(all_consumed_skills)
        else:
            step_skill_pressure = 0.0

        return {
            "skill_depletion": skill_depletion,
            "is_scarce_skill_used": is_scarce_skill_used,
            "resource_substitutability": resource_substitutability,
            "step_skill_pressure": step_skill_pressure,
        }

    @staticmethod
    def prompt_example() -> dict[str, Any]:
        
        return {
            "_schema_description": _query_schema_description(),
            "methods": {
                "step_allocation_summary": {
                    "function_definition": "step_allocation_summary() -> dict[str, Any]",
                    "inputs": {},
                    "outputs": {
                        "type": "dict[str, Any]",
                        "structure": {
                            "num_subtasks": "int",
                            "num_activated": "int",
                            "num_candidate": "int",
                            "num_formal": "int",
                            "activation_rate": "float",
                            "match_success_rate": "float",
                            "step_revenue": "float",
                            "num_completed_tasks": "int",
                            "total_utility": "float",
                            "total_travel_cost": "float",
                        },
                        "meaning": "Overall summary of the current step allocation result.",
                    },
                    
                    "meaning": "Computation: based on the current step context, aggregate counts of activated/candidate/formal matched subtasks, compute activation rate (activated/total), match success rate (formal/activated), step revenue and number of completed tasks, and sum total utility and total travel cost; returns integers and floats. Meaning: provides an overall assessment of allocation efficiency and quality for this time step, useful to judge whether revenue and resource objectives were met.",
                },
                "immediate_reward_info": {
                    "function_definition": "immediate_reward_info() -> dict[str, Any]",
                    "inputs": {},
                    "outputs": {
                        "type": "dict[str, Any]",
                        "structure": {
                            "is_activated": "bool",
                            "is_matched": "bool",
                            "immediate_reward": "float",
                            "estimated_reward": "float",
                            "reward_ratio": "float",
                        },
                        "meaning": "Immediate reward outcome for the current subtask decision.",
                    },
                    
                    "meaning": "Computation: determine immediate reward based on the subtask's action and whether it was formally matched this step: return 0 if not activated, return -penalty if activated but unmatched, return estimated_reward if matched; also return estimated_reward and reward_ratio (immediate/estimated). Returns booleans and floats. Meaning: quantifies the direct immediate reward impact and associated risk of the subtask decision (negative values indicate penalties).",
                },
                "relative_contribution": {
                    "function_definition": "relative_contribution() -> dict[str, float]",
                    "inputs": {},
                    "outputs": {
                        "type": "dict[str, float]",
                        "structure": {
                            "utility_share": "float",
                            "reward_share": "float",
                            "travel_cost_share": "float",
                            "task_progress_delta": "float",
                            "is_task_completing": "float",
                        },
                        "meaning": "Relative contribution of the subtask to the step outcome.",
                    },
                    
                    "meaning": "Computation: if the subtask was matched this step, compute its utility share (detail.utility/total_utility), reward share (estimated_reward/total_matched_est_reward), and travel cost share (detail.travel_cost/total_travel_cost); compute task_progress_delta as the change in matched subtasks fraction for its task and is_task_completing if the task finished this step. Return ranges: floats typically in [0,1] though relative ratios may exceed 1. Meaning: quantifies the subtask's relative contribution to step-level utility, reward, and task progress to aid prioritization.",
                },
                "revenue_impact": {
                    "function_definition": "revenue_impact() -> dict[str, float]",
                    "inputs": {},
                    "outputs": {
                        "type": "dict[str, float]",
                        "structure": {
                            "potential_reward": "float",
                            "reward_risk": "float",
                            "expected_value": "float",
                            "revenue_contribution_ratio": "float",
                        },
                        "meaning": "Revenue-related impact of the current subtask.",
                    },
                    
                    "meaning": "Computation: using heuristic completion probability based on whether the subtask is matched/activated and the fraction of matched subtasks in its task, compute potential_reward, expected_value = estimated_reward * completion_prob, and reward_risk = 1 - completion_prob; if the task completed this step, compute the subtask's revenue contribution ratio. Return: floats. Meaning: assesses the subtask's expected revenue contribution and failure risk to inform resource allocation decisions.",
                },
                "dependency_impact": {
                    "function_definition": "dependency_impact() -> dict[str, Any]",
                    "inputs": {},
                    "outputs": {
                        "type": "dict[str, Any]",
                        "structure": {
                            "unlocks_successors": "bool",
                            "num_successors_unlocked": "int",
                            "is_critical_node": "bool",
                            "dependency_relief_score": "float",
                            "blocks_others": "bool",
                            "step_dependency_chain_progress": "float",
                        },
                        "meaning": "Dependency-chain impact of this subtask within the current step.",
                    },
                    
                    "meaning": "Computation: if matched, identify successors that become unlocked because all their predecessors are matched or matched this step and count them; compute dependency_relief_score as unlocked_successors/total_successors; determine is_critical_node by ancestors/descendants presence; compute step_dependency_chain_progress from depths of step-matched subtasks; detect blocks_others by checking if alternatives exist for other activated subtasks. Returns booleans, integers and floats. Meaning: evaluates downstream unlocking benefits and potential blocking risks caused by this match.",
                },
                "step_resource_competition": {
                    "function_definition": "step_resource_competition() -> dict[str, Any]",
                    "inputs": {},
                    "outputs": {
                        "type": "dict[str, Any]",
                        "structure": {
                            "worker_contention_ratio": "float",
                            "subtask_per_worker": "float",
                            "skill_overlap_with_step": "float",
                            "is_worker_scarce_in_step": "float",
                        },
                        "meaning": "Resource competition pressure in the current step.",
                    },
                    
                    "meaning": "Computation: compute worker_contention_ratio as consumed_worker_count / num_idle, subtask_per_worker as num_activated / num_idle, compute skill_overlap as intersection/union between this subtask's skills and other step skills, and flag is_worker_scarce if num_formal > 0.8 * num_idle. Returns floats and booleans. Meaning: measures resource competition level in this step and whether allocation should consider substitutes or mitigation.",
                },
                "resource_utilization": {
                    "function_definition": "resource_utilization() -> dict[str, float]",
                    "inputs": {},
                    "outputs": {
                        "type": "dict[str, float]",
                        "structure": {
                            "matched_worker_skill_surplus": "float",
                            "matched_worker_distance_efficiency": "float",
                            "travel_cost": "float",
                            "utilization_score": "float",
                            "relative_efficiency": "float",
                        },
                        "meaning": "Resource utilization quality for the matched worker.",
                    },
                    
                    "meaning": "Computation: if matched, compute matched worker's skill surplus (worker_skills - needed_skills), distance efficiency = 1 - distance/max_distance, travel_cost, and compose a utilization_score by weighted combination of distance efficiency, skill utilization and travel cost; compute relative_efficiency as utilization_score divided by the average score across matched subtasks. Returns floats (may exceed 1). Meaning: evaluates how efficiently the matched worker was used; higher relative_efficiency means this match is more efficient than the step average.",
                },
                "skill_resource_impact": {
                    "function_definition": "skill_resource_impact() -> dict[str, Any]",
                    "inputs": {},
                    "outputs": {
                        "type": "dict[str, Any]",
                        "structure": {
                            "skill_depletion": "dict[int, float]",
                            "is_scarce_skill_used": "bool",
                            "resource_substitutability": "float",
                            "step_skill_pressure": "float",
                        },
                        "meaning": "Skill resource pressure caused by the current step.",
                    },
                    
                    "meaning": "Computation: using step-level and global supply-demand ratios compute skill_depletion per required skill as approx. 1/(ratio+eps) clipped to 1; detect if any required skill is scarce (ratio < 0.5); compute resource_substitutability as the count of substitute idle workers divided by total idle workers; compute step_skill_pressure as fraction of consumed skills considered scarce. Returns dicts, booleans and floats. Meaning: assesses how the step's matches deplete the skill pool and whether high-value or scarce resources were used, guiding substitution or protection strategies.",
                },
            },
        }
