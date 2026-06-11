from __future__ import annotations

import math
import uuid
from dataclasses import dataclass

from src.config.config import SystemConfig, ModelConfig
from src.env.distance_util import distance_util


@dataclass
class StepResult:
    

    subtasks: list["SubTask"]
    state: "State"
    observations: list["Observation"]
    actions: list[int]
    log_probs: list[float]
    next_state: "State"
    done: bool
    is_matched: list[bool]
    contrib_features: list[list[float]]
    revenue: float
    infos: list[float]


@dataclass(frozen=True)
class MatchWorkerSnapshot:
    

    id: str
    location: tuple[float, float]
    skills: frozenset[int]
    max_distance: float
    idle_time: int


@dataclass(frozen=True)
class MatchDetailSnapshot:
    

    subtask: "SubTask"
    worker: MatchWorkerSnapshot
    utility: float
    travel_time: int
    travel_cost: float
    complete_time: int
    pre_complete_time: int


@dataclass
class ContributionContext:
    

    step_index: int
    now_time: int
    subtasks: list["SubTask"]
    idle_workers: list["Worker"]
    actions_by_subtask_id: dict[str, int]
    activated_subtask_ids: set[str]
    candidate_matching: list[tuple["SubTask", "Worker"]]
    candidate_subtask_ids: set[str]
    formal_matching: list[tuple["SubTask", "Worker"]]
    formal_subtask_ids: set[str]
    match_details: dict[str, MatchDetailSnapshot]
    completed_task_ids: set[str]
    task_matched_before: dict[str, int]
    revenue: float


@dataclass
class StepAllocationResult:
    

    revenue: float
    is_matched: list[bool]
    infos: list[float]
    contribution_context: ContributionContext


class Transition:
    
    def __init__(
            self,
            subtasks: list[SubTask],
            observations: list[Observation],
            actions: list[int],
            log_probs: list[float],
            state: State,
            next_state: State,
            done: bool,
            is_matched: list[bool],
    ):
        self.id = uuid.uuid4()
        self.subtasks = subtasks
        self.observations = observations
        self.state = state
        self.actions = actions
        self.log_probs = log_probs
        self.reward = 0.
        self.reward_list = []
        self.next_state = next_state
        self.return_: float | None = None  
        self.done = done
        self.is_matched = is_matched
        self.action_adv: list[float] = []
        self.contrib_features: list[list[float]] = []

        assert all([a == 1 or a == 0 for a in actions])

    def __eq__(self, other):
        
        if not isinstance(other, Transition):
            return False
        return self.id == other.id

    def __hash__(self):
        
        return hash(self.id)


class State:
    
    def __init__(
            self,
            customs: list[float],
            workers_infos: list[tuple[list[float], set[int]]],
            tasks_infos: list[list[float]],
            subtasks_infos: list[tuple[list[float], set[int]]],
    ):
        self.customs = customs
        self.workers_infos = workers_infos
        self.tasks_infos = tasks_infos
        self.subtasks_infos = subtasks_infos


class Observation:
    
    def __init__(
            self,
            customs: list[float],
            subtask_info: tuple[list[float], set[int]],
            task_info: list[float],
            available_workers_infos: list[tuple[list[float], set[int]]],
            graph_matrix: list[list[int]],
            graph_subtasks_infos: list[tuple[list[float], set[int]]],
            graph_index: int,
    ):
        self.customs = customs
        self.subtask_info = subtask_info
        self.task_info = task_info
        self.available_workers_infos = available_workers_infos
        self.graph_matrix = graph_matrix
        self.graph_subtasks_infos = graph_subtasks_infos
        self.graph_index = graph_index

class Worker:
    
    def __init__(
            self,
            id: str,
            location: tuple[float, float],
            skills: set[int],
            velocity: float,
            travel_cost_per_km: float,
            max_distance: float,
            appear_time: float,
    ):
        
        self.id = id
        self.location = location
        self.skills = skills
        self.velocity = velocity
        self.travel_cost_per_km = travel_cost_per_km
        self.max_distance = max_distance
        
        self.idle_time: int = 0
        
        self.assigned_subtask: SubTask | None = None

        self.appear_time = appear_time

    def info(self, now_time: int) -> tuple[list[float], set[int]]:
        
        features = [len(self.skills) / ModelConfig.nl_skill_nums()]
        return features, self.skills

    def travel_time(self, location: tuple[float, float]) -> int:
        
        
        return math.ceil(distance_util.shortest_distance(location, self.location) / self.velocity * 3600)

    def travel_cost(self, location: tuple[float, float]) -> float:
        
        return distance_util.shortest_distance(location, self.location) * self.travel_cost_per_km

    def constraints_ok(self, subtask: SubTask, now_time: int) -> bool:
        
        return distance_util.shortest_distance(subtask.location, self.location) <= self.max_distance and \
            self.skills.issuperset(subtask.needed_skills) and \
            now_time + self.travel_time(subtask.location) + subtask.duration <= subtask.sub_deadline

    def matching_utility(self, subtask: SubTask, now_time: int) -> float:
        
        task = subtask.task
        return task.reward / len(task.subtasks) \
            * math.exp(-distance_util.shortest_distance(subtask.location, self.location) / self.max_distance)

    def __eq__(self, other):
        
        if not isinstance(other, Worker):
            return False
        return self.id == other.id

    def __hash__(self):
        
        return hash(self.id)

class Task:
    
    def __init__(
            self,
            id: str,
            reward: float,
            published_time: int,
            deadline: int,
            dependency_matrix: list[list[int]],
            subtasks: list[SubTask],
    ):
        
        self.id = id
        self.reward = reward
        self.published_time = published_time
        self.deadline = deadline
        self.dependency_matrix = dependency_matrix

        self.complete_time: int | None = None

        self.subtasks: list[SubTask] = subtasks

        self.subtask_ancestors: dict[SubTask, set[SubTask]] = dict()
        self.subtask_descendants: dict[SubTask, set[SubTask]] = dict()

        self.init_subtasks()



    def info(self, now_time: int) -> list[float]:
        
        return [
            (self.deadline - now_time) / ModelConfig.nl_time() , self.reward / ModelConfig.nl_reward(),
            len([subtask for subtask in self.subtasks if subtask.is_matched()]) / ModelConfig.nl_subtasks(),
            len([subtask for subtask in self.subtasks if not subtask.is_matched()]) / ModelConfig.nl_subtasks(),
        ]


    def is_fully_matched(self) -> bool:
        
        return all(subtask.is_matched() for subtask in self.subtasks)

    def build_ancestor_descendant_maps(self):
        
        ancestors = {sub: set() for sub in self.subtasks}
        descendants = {sub: set() for sub in self.subtasks}

        def dfs_desc(sub: SubTask):
            if descendants[sub]:
                return descendants[sub]
            for post in sub.post_subtasks:
                descendants[sub].add(post)
                descendants[sub] |= dfs_desc(post)
            return descendants[sub]

        def dfs_anc(sub: SubTask):
            if ancestors[sub]:
                return ancestors[sub]
            for pre in sub.pre_subtasks:
                ancestors[sub].add(pre)
                ancestors[sub] |= dfs_anc(pre)
            return ancestors[sub]

        for sub in self.subtasks:
            dfs_desc(sub)
            dfs_anc(sub)

        self.subtask_ancestors = ancestors
        self.subtask_descendants = descendants

    def init_subtasks(self):
        
        for subtask in self.subtasks:
            subtask.task = self

        for i in range(len(self.subtasks)):
            for j in range(len(self.subtasks)):
                if self.dependency_matrix[i][j] == 1:
                    self.subtasks[i].post_subtasks.append(self.subtasks[j])
                    self.subtasks[j].pre_subtasks.append(self.subtasks[i])


        def do_init_depth(subtask: SubTask):
            if len(subtask.pre_subtasks) == 0:
                subtask.depth = 0
                return
            for pre in subtask.pre_subtasks:
                if pre.depth is None:
                    do_init_depth(pre)
            subtask.depth = max(pre.depth for pre in subtask.pre_subtasks) + 1

        def do_init_deadline(subtask: SubTask):
            if len(subtask.post_subtasks) == 0:
                subtask.sub_deadline = self.deadline
                return
            for post in subtask.post_subtasks:
                if post.sub_deadline is None:
                    do_init_deadline(post)
            subtask.sub_deadline = min(post.sub_deadline - post.duration for post in subtask.post_subtasks)

        for subtask in self.subtasks:
            do_init_deadline(subtask)
            do_init_depth(subtask)

        self.build_ancestor_descendant_maps()

    def print_completion(self):
        
        print(self.complete_time, self.deadline, [(subtask.pre_complete_time, subtask.complete_time, subtask.sub_deadline) for subtask in self.subtasks])



    def __eq__(self, other):
        
        if not isinstance(other, Task):
            return False
        return self.id == other.id

    def __hash__(self):
        
        return hash(self.id)



class SubTask:
    
    def __init__(
            self,
            id: str,
            location: tuple[float, float],
            needed_skills: set,
            duration: int,
    ):
        
        self.id = id
        self.location = location
        self.needed_skills = needed_skills
        self.duration = duration 

        
        self.pre_subtasks: list[SubTask] = []
        self.post_subtasks: list[SubTask] = []

        self.task: Task | None = None

        self.pre_complete_time: int | None = None

        self.complete_time: int | None = None

        self.travel_cost: float | None = None

        self.matched_worker: Worker | None = None

        self.depth: int | None = None

        self.sub_deadline: int | None = None

        
        self.t = 0
        self.is_activated: bool = False

    def num_post_unmatched(self, sub: SubTask) -> int:
        
        if len(sub.post_subtasks) == 0:
            return 0

        return sum(
            self.num_post_unmatched(post) + 1
            for post in sub.post_subtasks if not post.is_matched()
        )


    def info(self, now_time: int) -> tuple[list[float], set[int]]:
        

        num_ancestors = len([act for act in self.task.subtask_ancestors[self] if not act.is_matched()])
        num_descendants = len([dsd for dsd in self.task.subtask_descendants[self] if not dsd.is_matched()])

        features = [
            num_ancestors / ModelConfig.nl_subtasks(),
            num_descendants / ModelConfig.nl_subtasks(),
            int(self.is_matched()),
            self.duration / ModelConfig.nl_time() * ModelConfig.nl_subtasks(),
            (self.sub_deadline - now_time) / ModelConfig.nl_time(),
        ]
        return features, self.needed_skills


    def is_matched(self) -> bool:
        
        return self.complete_time is not None


    def init_dependency(self):
        
        self.pre_subtasks: list[SubTask] = []
        self.post_subtasks: list[SubTask] = []
        self.depth = 0
        pass

    def estimated_reward(self) -> float:
        
        return self.task.reward / len(self.task.subtasks)

    def __eq__(self, other):
        
        if not isinstance(other, SubTask):
            return False
        return self.id == other.id

    def __hash__(self):
        
        return hash(self.id)
