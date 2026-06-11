from copy import deepcopy

import pandas as pd

from src.config.config import ModelConfig, BaseConfig, SystemConfig
from src.env.entity import (
    Worker,
    Task,
    SubTask,
    Observation,
    State,
    StepResult,
    ContributionContext,
    MatchDetailSnapshot,
    MatchWorkerSnapshot,
    StepAllocationResult,
)
from src.env.match import KM
from src.utils.timer import Timer


class CrowdsourcingEnv:
    

    def __init__(self, time_window_size: int, dataset_paths: dict[str, str] | None = None):
        # initialize environment
        self.time_window_size = time_window_size
        self.dataset_paths = dict(dataset_paths) if dataset_paths is not None else None

        self.all_workers: list[Worker] = []
        self.all_tasks: list[Task] = []
        self.all_subtasks: list[SubTask] = []
        self.load()

        self.latest_publish_time = max(task.published_time for task in self.all_tasks)
        
        self.now_time = min(task.published_time for task in self.all_tasks) - self.time_window_size

        self.unmatched_tasks: list[Task] = []
        self.unmatched_subtasks: list[SubTask] = []
        self.idle_workers: list[Worker] = []

        self.step_index: int = 0


    def load(self):
        # load dataset and initialize tasks, subtasks, workers

        subtasks = []
        subtask_dict = {}  

        subtask_csv = (
            self.dataset_paths.get("subtask_csv")
            if self.dataset_paths is not None
            else BaseConfig.subtask_csv
        )
        task_csv = (
            self.dataset_paths.get("task_csv")
            if self.dataset_paths is not None
            else BaseConfig.task_csv
        )
        worker_csv = (
            self.dataset_paths.get("worker_csv")
            if self.dataset_paths is not None
            else BaseConfig.worker_csv
        )

        df_subtask = pd.read_csv(subtask_csv)
        for _, row in df_subtask.iterrows():
            subtask = SubTask(
                id=str(row['id']),
                location=(float(row['lng']), float(row['lat'])),
                needed_skills=set(map(int, row['skills'].split(','))),
                duration=int(row['workload'])
            )
            subtasks.append(subtask)
            subtask_dict[subtask.id] = subtask

        
        tasks = []
        df_task = pd.read_csv(task_csv)
        for _, row in df_task.iterrows():
            
            matrix = [
                list(map(int, line.split(',')))
                for line in row['dependency_matrix'].split('_')
            ]

            task = Task(
                id=str(row['id']),
                reward=float(row['max_reward']),
                published_time=int(row['published_time']),
                deadline=int(row['deadline']),
                subtasks=[subtask_dict[subtask_id] for subtask_id in row['subtasks'].split(',')],
                dependency_matrix=matrix
            )
            tasks.append(task)

        
        workers = []
        df_worker = pd.read_csv(worker_csv)
        for _, row in df_worker.iterrows():
            worker = Worker(
                id=str(row['id']),
                location=(float(row['lng']), float(row['lat'])),
                skills=set(map(int, row['skills'].split(','))),
                velocity=float(row['velocity']),
                travel_cost_per_km=float(row['travel_cost_per_km']),
                max_distance=10.,
                appear_time=float(row['appear_time'])
            )
            workers.append(worker)
            if not SystemConfig.dynamic_worker_appearance:
                worker.appear_time = -1

        self.all_workers = workers
        self.all_tasks = tasks
        self.all_subtasks = subtasks

    def reset(self):
        
        self.__init__(self.time_window_size, self.dataset_paths)
        self.to_next_batch()

    def get_state(self) -> State:
        
        i_unmatched_tasks = [t.info(self.now_time) for t in self.unmatched_tasks]
        i_unmatched_subtasks = [
            sub.info(self.now_time)
            for sub in self.unmatched_subtasks
        ]

        i_idle_workers = [
            wk.info(self.now_time)
            for wk in self.idle_workers
        ]

        num_matched_subtasks = sum([len(t.subtasks) for t in self.unmatched_tasks]) - len(i_unmatched_subtasks)

        customs = [
            len(i_unmatched_tasks) / ModelConfig.nl_global_tasks(),
            len(i_unmatched_subtasks) / ModelConfig.nl_global_subtasks(),
            len(i_idle_workers) / ModelConfig.nl_global_workers(),
            num_matched_subtasks / ModelConfig.nl_global_subtasks()
        ]

        return State(
            customs=customs,
            workers_infos=i_idle_workers,
            tasks_infos=i_unmatched_tasks,
            subtasks_infos=i_unmatched_subtasks,
        )

    def get_observation(self, subtask: SubTask) -> Observation:
        
        task = subtask.task
        graph_martix = deepcopy(task.dependency_matrix)
        subtask_index = task.subtasks.index(subtask)
        i_subtasks = [s.info(self.now_time) for s in task.subtasks]

        i_available_workers = [
            wk.info(self.now_time) for wk in self.idle_workers
            if wk.constraints_ok(subtask, self.now_time)
        ]


        activated_nums = sum([sub.is_activated for sub in self.unmatched_subtasks])

        customs = [
            len(self.unmatched_subtasks) / ModelConfig.nl_global_subtasks(),
            len(self.idle_workers) / ModelConfig.nl_global_workers(),
            len(i_available_workers) / ModelConfig.nl_workers(),
            activated_nums / ModelConfig.nl_global_subtasks()
        ]

        return Observation(
            customs=customs, 
            subtask_info=subtask.info(self.now_time),
            task_info=task.info(self.now_time),
            available_workers_infos=i_available_workers,
            graph_matrix=graph_martix,
            graph_index=subtask_index,
            graph_subtasks_infos=i_subtasks,
        )


    def is_done(self) -> bool:
        
        return len(self.unmatched_tasks) == 0 and self.now_time >= self.latest_publish_time

    def _attach_state_augmentation(self, state: State, function_set) -> None:
        
        from src.utils.query_functions import StateQuerySet

        state_query = StateQuerySet(self)
        state.llm_state_aug = function_set.build_state_aug(state_query)

    def _build_observation_with_aug(self, subtask: SubTask, function_set) -> Observation:
        
        from src.utils.query_functions import ObservationQuerySet

        observation = self.get_observation(subtask)
        ob_query = ObservationQuerySet(self, subtask)
        observation.llm_observation_aug = function_set.build_observation_aug(ob_query)
        return observation

    def _build_contrib_features(
        self,
        context: ContributionContext,
        function_set,
    ) -> list[list[float]]:
        
        from src.utils.query_functions import ContributionQuerySet

        return [
            function_set.build_contribution_repr(ContributionQuerySet(context, subtask))
            for subtask in context.subtasks
        ]

    def _select_actions_with_aug(
        self,
        agent,
        subtasks: list[SubTask],
        enable_llm_augmentation: bool,
        function_set,
    ) -> tuple[list[int], list[Observation], list[float]]:
        
        if not enable_llm_augmentation or function_set is None:
            return agent.select_actions(subtasks=subtasks, get_observation=self.get_observation)

        return agent.select_actions(
            subtasks=subtasks,
            get_observation=lambda subtask: self._build_observation_with_aug(subtask, function_set),
        )

    def step(
        self,
        agent,
        function_set=None,
        enable_llm_augmentation: bool = False,
        enable_credit_net: bool = False,
        do_print: bool = False,
    ) -> StepResult:
        # get current state
        state = self.get_state()
        if enable_llm_augmentation and function_set is not None:
            self._attach_state_augmentation(state, function_set)

        step_subtasks = self.unmatched_subtasks.copy()
        actions, observations, log_probs = self._select_actions_with_aug(
            agent=agent,
            subtasks=step_subtasks,
            enable_llm_augmentation=enable_llm_augmentation,
            function_set=function_set,
        )
        # perform action and get allocation result
        allocation = self._step_actions(actions, do_print)

        contrib_features: list[list[float]] = []
        if enable_credit_net and function_set is not None:
            contrib_features = self._build_contrib_features(
                allocation.contribution_context,
                function_set,
            )
        # update environment state to the next step
        self.to_next_batch()
        # get next state
        next_state = self.get_state()
        if enable_llm_augmentation and function_set is not None:
            self._attach_state_augmentation(next_state, function_set)

        return StepResult(
            subtasks=step_subtasks,
            state=state,
            observations=observations,
            actions=actions,
            log_probs=log_probs,
            next_state=next_state,
            done=self.is_done(),
            is_matched=allocation.is_matched,
            contrib_features=contrib_features,
            revenue=allocation.revenue,
            infos=allocation.infos,
        )

    @Timer()
    def _step_actions(self, actions: list[int], do_print = False) -> StepAllocationResult:
        
        
        step_subtasks = self.unmatched_subtasks.copy()
        step_workers = self.idle_workers.copy()
        task_matched_before = {
            task.id: sum(1 for sub in task.subtasks if sub.is_matched())
            for task in self.unmatched_tasks
        }

        activated_subtasks = [subtask for subtask, action in zip(step_subtasks, actions) if action == 1]
        candidate_matching = self.match_algorithm(activated_subtasks, step_workers)
        formal_matching = self.filter_match(candidate_matching)
        if do_print:
            print(f"--tasks:{len(self.unmatched_tasks)}, subtasks:{len(self.unmatched_subtasks)}")

        revenue, match_details, completed_task_ids = self.update_by_matching(formal_matching)

        is_matched = [sub.is_matched() for sub in step_subtasks]

        infos = [
            len(self.unmatched_subtasks), len(activated_subtasks), len(self.idle_workers),
            len(candidate_matching), len(formal_matching), len(self.unmatched_tasks)
        ]

        contribution_context = ContributionContext(
            step_index=self.step_index,
            now_time=self.now_time,
            subtasks=step_subtasks,
            idle_workers=step_workers,
            actions_by_subtask_id={sub.id: act for sub, act in zip(step_subtasks, actions)},
            activated_subtask_ids={sub.id for sub in activated_subtasks},
            candidate_matching=candidate_matching,
            candidate_subtask_ids={sub.id for sub, _ in candidate_matching},
            formal_matching=formal_matching,
            formal_subtask_ids={sub.id for sub, _ in formal_matching},
            match_details=match_details,
            completed_task_ids=completed_task_ids,
            task_matched_before=task_matched_before,
            revenue=revenue,
        )
        self.step_index += 1

        return StepAllocationResult(
            revenue=revenue,
            is_matched=is_matched,
            infos=infos,
            contribution_context=contribution_context,
        )

    def to_next_batch(self):
        
        
        for subtask in self.unmatched_subtasks:
            subtask.is_activated = False

        self.now_time += self.time_window_size

        
        self.unmatched_tasks = [
            task for task in self.all_tasks
            if task.published_time <= self.now_time and not task.is_fully_matched()
               and all(subtask.sub_deadline - subtask.duration >= self.now_time for subtask in task.subtasks)
        ]

        self.unmatched_subtasks = [
            subtask
            for task in self.unmatched_tasks
            for subtask in task.subtasks
            if not subtask.is_matched()
        ]

        self.idle_workers = [worker for worker in self.all_workers if
                             worker.appear_time <= self.now_time and worker.idle_time <= self.now_time]




    def filter_match(self, candidate_matching: list[tuple[SubTask, Worker]]) -> list[tuple[SubTask, Worker]]:
        
        candidate_matching = candidate_matching.copy()
        
        candidate_matching.sort(key=lambda mt: mt[0].depth)
        
        valid_subtask = set()
        formal_matching = []

        for subtask, worker in candidate_matching:
            if len(subtask.pre_subtasks) == 0 or \
                    all(pre.is_matched() or pre in valid_subtask for pre in subtask.pre_subtasks):
                valid_subtask.add(subtask)
                formal_matching.append((subtask, worker))

        return formal_matching

    @Timer()
    def match_algorithm(
            self,
            subtasks: list[SubTask],
            workers: list[Worker]
    ) -> list[tuple[SubTask, Worker]]:
        

        matches, sum_utility = KM(subtasks, workers, self.now_time)
        
        
        

        return matches

    def update_by_matching(
        self,
        matching: list[tuple[SubTask, Worker]],
    ) -> tuple[float, dict[str, MatchDetailSnapshot], set[str]]:
        
        matching = matching.copy()
        
        matching.sort(key=lambda mt: mt[0].depth)
        reward = 0
        match_details: dict[str, MatchDetailSnapshot] = {}
        completed_task_ids: set[str] = set()
        for subtask, worker in matching:
            utility = worker.matching_utility(subtask, self.now_time)
            travel_t = worker.travel_time(subtask.location)
            travel_c = worker.travel_cost(subtask.location)
            worker_snapshot = MatchWorkerSnapshot(
                id=worker.id,
                location=worker.location,
                skills=frozenset(worker.skills),
                max_distance=worker.max_distance,
                idle_time=worker.idle_time,
            )

            
            subtask.matched_worker = worker
            if len(subtask.pre_subtasks) == 0:
                subtask.pre_complete_time = 0
            else:
                subtask.pre_complete_time = max(pre.complete_time for pre in subtask.pre_subtasks)
            subtask.complete_time = max(subtask.pre_complete_time, travel_t + self.now_time) \
                                    + subtask.duration
            subtask.travel_cost = travel_c

            match_details[subtask.id] = MatchDetailSnapshot(
                subtask=subtask,
                worker=worker_snapshot,
                utility=utility,
                travel_time=travel_t,
                travel_cost=travel_c,
                complete_time=subtask.complete_time,
                pre_complete_time=subtask.pre_complete_time,
            )

            
            if subtask.complete_time > subtask.sub_deadline:
                raise Exception(f"error complete time {subtask.complete_time}, {subtask.sub_deadline}")

            
            worker.idle_time = subtask.complete_time
            worker.location = subtask.location

            
            task = subtask.task
            if task.is_fully_matched():
                
                task.complete_time = max(subtask.complete_time for subtask in task.subtasks)
                reward += task.reward
                completed_task_ids.add(task.id)


        return reward, match_details, completed_task_ids
