import os
import random
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from src.config.config import ModelConfig, BaseConfig, SystemConfig
from src.env.entity import SubTask, Transition, State, Observation, Task
from src.net.network import PolicyNet, ValueNet, CreditNet, encode_skill_set
from src.utils.common import adj_matrix_to_edge_index



class Buffer:
    

    def __init__(self):
        self.transitions: list[Transition] = []

    def store(self, transition: Transition):
        
        self.transitions.append(transition)

    def clear(self):
        
        self.transitions.clear()


class Agent:
    

    def __init__(
        self,
        gamma: float = ModelConfig.gamma,
        gae_lambda: float = ModelConfig.gae_lambda,
        clip_epsilon: float = ModelConfig.clip_epsilon,
        policy_lr = ModelConfig.policy_lr,
        value_lr = ModelConfig.value_lr,
        value_coef: float = ModelConfig.value_coef,
        entropy_coef: float = ModelConfig.entropy_coef, 
        max_grad_norm: float = ModelConfig.max_grad_norm,
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.buffer = Buffer()
        self.policy = PolicyNet(
            ModelConfig.policy_output_dim,
            ModelConfig.policy_hidden_dim
        ).to(self.device)

        self.value = ValueNet(
            ModelConfig.value_input_dim,
            ModelConfig.value_hidden_dim
        ).to(self.device)

        
        self.policy_optimizer = Adam(
            list(self.policy.parameters()),
            lr=policy_lr
        )

        self.value_optimizer = Adam(
            list(self.value.parameters()),
            lr=value_lr
        )

        self.enable_credit_net = ModelConfig.enable_credit_net
        self.credit_alpha = ModelConfig.credit_alpha
        self.credit = None
        self.credit_optimizer = None
        if self.enable_credit_net:
            self.credit = CreditNet(
                input_dim=ModelConfig.llm_contribution_dim,
                hidden_dim=ModelConfig.credit_hidden_dim,
            ).to(self.device)
            self.credit_optimizer = Adam(
                self.credit.parameters(),
                lr=ModelConfig.credit_lr,
            )

        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self._policy_update_steps = 0
        self._value_update_steps = 0

    def _fixed_aug_tensor(self, aug: list[float] | None, target_dim: int) -> torch.Tensor:
        
        if aug is None:
            return torch.zeros((target_dim,), dtype=torch.float32, device=self.device)

        aug_tensor = torch.tensor(aug, dtype=torch.float32, device=self.device)
        if aug_tensor.numel() == target_dim:
            return aug_tensor

        
        fixed = torch.zeros((target_dim,), dtype=torch.float32, device=self.device)
        copy_len = min(target_dim, int(aug_tensor.numel()))
        if copy_len > 0:
            fixed[:copy_len] = aug_tensor[:copy_len]
        return fixed

    def _state_aug_tensor(self, state: State) -> torch.Tensor:
        
        aug = getattr(state, "llm_state_aug", None)
        if not ModelConfig.enable_llm_augmentation:
            aug = None
        return self._fixed_aug_tensor(aug, ModelConfig.llm_state_aug_dim)

    def _observation_aug_tensor(self, observation: Observation) -> torch.Tensor:
        
        aug = getattr(observation, "llm_observation_aug", None)
        if not ModelConfig.enable_llm_augmentation:
            aug = None
        return self._fixed_aug_tensor(aug, ModelConfig.llm_observation_aug_dim)

    def _contrib_tensor(self, contrib_features: list[list[float]]) -> torch.Tensor:
        
        if len(contrib_features) == 0:
            return torch.empty((0, ModelConfig.llm_contribution_dim), dtype=torch.float32, device=self.device)
        return torch.tensor(contrib_features, dtype=torch.float32, device=self.device)

    def states_to_tensor(self, states: list[State]):
        
        
        
        customs_list: list[torch.Tensor] = []
        for state in states:
            base = torch.tensor(state.customs, dtype=torch.float32, device=self.device)
            aug = self._state_aug_tensor(state)
            customs_list.append(torch.cat([base, aug], dim=0))
        customs = torch.stack(customs_list, dim=0)

        workers_enc_list = []
        tasks_enc_list = []
        subtasks_enc_list = []

        for state in states:
            
            if state.workers_infos:
                worker_encs = []
                for features, skill_set in state.workers_infos:
                    skill_enc = encode_skill_set(skill_set, device=self.device)
                    enc = torch.cat([torch.tensor(features, dtype=torch.float32, device=self.device), skill_enc])
                    worker_encs.append(enc)
                worker_encs = torch.stack(worker_encs, dim=0)
                workers_enc_list.append(torch.cat([
                    worker_encs.mean(dim=0),
                    worker_encs.max(dim=0).values
                ]))
            else:
                workers_enc_list.append(torch.zeros(ModelConfig.worker_dim * 2, device=self.device))

            if state.tasks_infos:
                tasks_tensor = torch.tensor(state.tasks_infos, dtype=torch.float32, device=self.device)
                tasks_enc_list.append(torch.cat([
                    tasks_tensor.mean(dim=0),
                    tasks_tensor.max(dim=0).values
                ]))
            else:
                tasks_enc_list.append(torch.zeros(ModelConfig.task_dim * 2, device=self.device))

            if state.subtasks_infos:
                subtask_encs = []
                for features, skill_set in state.subtasks_infos:
                    skill_enc = encode_skill_set(skill_set, device=self.device)
                    enc = torch.cat([torch.tensor(features, dtype=torch.float32, device=self.device), skill_enc])
                    subtask_encs.append(enc)
                subtask_encs = torch.stack(subtask_encs, dim=0)
                subtasks_enc_list.append(torch.cat([
                    subtask_encs.mean(dim=0),
                    subtask_encs.max(dim=0).values
                ]))
            else:
                subtasks_enc_list.append(torch.zeros(ModelConfig.subtask_dim * 2, device=self.device))

        workers_enc = torch.stack(workers_enc_list, dim=0)
        tasks_enc = torch.stack(tasks_enc_list, dim=0)
        subtasks_enc = torch.stack(subtasks_enc_list, dim=0)

        return customs, workers_enc, tasks_enc, subtasks_enc

    def observation_to_tensor(self, observation: Observation):
        
        device = self.device

        
        ob_aug = self._observation_aug_tensor(observation)
        ob_features = torch.cat([
            torch.tensor(observation.customs, dtype=torch.float32, device=device),
            ob_aug,
            torch.tensor(observation.task_info, dtype=torch.float32, device=device),
        ])

        
        workers_enc = self._pre_encode_workers(observation.available_workers_infos)

        
        graph_nodes = self._pre_encode_subtasks(observation.graph_subtasks_infos)

        return (
            ob_features,
            adj_matrix_to_edge_index(observation.graph_matrix, device=device),
            graph_nodes,
            torch.tensor(observation.graph_index, dtype=torch.long, device=device),
            workers_enc,
        )

    def _pre_encode_workers(self, workers_info: list) -> torch.Tensor:
        
        if not workers_info:
            return torch.zeros(ModelConfig.worker_dim * 2, device=self.device)

        
        worker_encs = []
        for features, skill_set in workers_info:
            skill_enc = encode_skill_set(skill_set, device=self.device)
            enc = torch.cat([torch.tensor(features, dtype=torch.float32, device=self.device), skill_enc])
            worker_encs.append(enc)

        worker_encs = torch.stack(worker_encs, dim=0)  

        
        mean_enc = worker_encs.mean(dim=0)
        max_enc = worker_encs.max(dim=0).values

        return torch.cat([mean_enc, max_enc])

    def _pre_encode_subtasks(self, subtasks_infos: list) -> torch.Tensor:
        
        if not subtasks_infos:
            return torch.empty((0, ModelConfig.subtask_dim), dtype=torch.float32, device=self.device)

        subtask_encs = []
        for features, skill_set in subtasks_infos:
            skill_enc = encode_skill_set(skill_set, device=self.device)
            enc = torch.cat([torch.tensor(features, dtype=torch.float32, device=self.device), skill_enc])
            subtask_encs.append(enc)

        return torch.stack(subtask_encs, dim=0)  

    def observations_to_tensor(self, observations: list[Observation]):
        
        ob_features_list = []
        workers_enc_list = []
        graph_nodes_list = []
        max_nodes = 0
        max_edges = 0

        for ob in observations:
            ob_aug = self._observation_aug_tensor(ob)
            ob_features_list.append(
                torch.cat([
                    torch.tensor(ob.customs, dtype=torch.float32, device=self.device),
                    ob_aug,
                    torch.tensor(ob.task_info, dtype=torch.float32, device=self.device),
                ])
            )
            
            workers_enc_list.append(self._pre_encode_workers(ob.available_workers_infos))
            
            graph_nodes_list.append(self._pre_encode_subtasks(ob.graph_subtasks_infos))
            max_nodes = max(max_nodes, len(ob.graph_subtasks_infos))
            edge_count = sum(1 for row in ob.graph_matrix for v in row if v != 0)
            max_edges = max(max_edges, edge_count)

        batch_size = len(observations)
        graph_nodes = torch.full(
            (batch_size, max_nodes, ModelConfig.subtask_dim),
            float("nan"),
            dtype=torch.float32,
            device=self.device,
        )
        edge_index = torch.full(
            (batch_size, 2, max_edges),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        graph_index = torch.empty((batch_size,), dtype=torch.long, device=self.device)

        for i, ob in enumerate(observations):
            
            edge_index_i = adj_matrix_to_edge_index(ob.graph_matrix, device=self.device)
            if edge_index_i.numel() > 0:
                edge_index[i, :, :edge_index_i.size(1)] = edge_index_i
            graph_index[i] = ob.graph_index

        ob_features = torch.stack(ob_features_list, dim=0)
        workers_enc = torch.stack(workers_enc_list, dim=0)  
        return ob_features, edge_index, graph_nodes_list, graph_index, workers_enc

    def entropy_decay(self):
        
        entropy_decay = 0.995
        entropy_min = 0.0
        self.entropy_coef = max(entropy_min, self.entropy_coef * entropy_decay)

    def _policy_parameters(self) -> list[torch.nn.Parameter]:
        
        return [p for p in self.policy.parameters() if p.requires_grad]

    @staticmethod
    def _set_module_requires_grad(module: torch.nn.Module | None, flag: bool):
        
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad_(flag)

    def _flatten_policy_tensors(
            self,
            tensors: tuple[torch.Tensor | None, ...] | list[torch.Tensor | None],
    ) -> torch.Tensor:
        
        flat_tensors: list[torch.Tensor] = []
        params = self._policy_parameters()
        assert len(tensors) == len(params)

        for tensor, param in zip(tensors, params):
            if tensor is None:
                flat_tensors.append(torch.zeros(param.numel(), dtype=param.dtype, device=param.device))
            else:
                flat_tensors.append(tensor.reshape(-1))

        if len(flat_tensors) == 0:
            return torch.empty((0,), dtype=torch.float32, device=self.device)
        return torch.cat(flat_tensors, dim=0)

    def _forward_policy_batch(
            self,
            observations: list[Observation],
            actions: list[int],
            old_log_probs: list[float],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        ob_features, edge_index, graph_nodes, graph_index, workers_enc = self.observations_to_tensor(observations)
        logits = self.policy(ob_features, edge_index, graph_nodes, graph_index, workers_enc)
        dist = torch.distributions.Categorical(logits=logits)

        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        old_log_probs_tensor = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        log_probs = dist.log_prob(actions_tensor)
        ratio = torch.exp(log_probs - old_log_probs_tensor)

        return ratio, dist.entropy(), log_probs, old_log_probs_tensor

    def _credit_adjust_advantages_tensor(
            self,
            base_advantages: torch.Tensor,
            contrib_features: list[list[float]],
    ) -> tuple[torch.Tensor, bool]:
        
        if (
                not self.enable_credit_net
                or self.credit is None
                or len(contrib_features) == 0
                or len(contrib_features) != int(base_advantages.numel())
        ):
            return base_advantages, False

        c_tensor = self._contrib_tensor(contrib_features)
        if c_tensor.shape[0] == 0:
            return base_advantages, False

        beta = torch.tanh(self.credit(c_tensor)).squeeze(-1)
        beta_centered = beta - beta.mean()
        global_adv_abs = torch.abs(base_advantages.sum())
        scale = self.credit_alpha * global_adv_abs / max(1, int(base_advantages.numel()))
        credited = base_advantages + scale * beta_centered
        return credited, True

    def _scale_action_advantages(self, action_advantages: list[list[float]]) -> list[list[float]]:
        
        scaled: list[list[float]] = []
        for adv in action_advantages:
            adv_np = np.array(adv, dtype=np.float32)
            adv_np = adv_np * ModelConfig.advantage_scale
            scaled.append(adv_np.tolist())
        return scaled

    def _log_action_advantages(self, action_advantages: list[list[float]]):
        
        flat_scaled = [v for adv in action_advantages for v in adv]
        all_scaled_adv = np.array(flat_scaled, dtype=np.float32) if len(flat_scaled) > 0 else np.array([0.0], dtype=np.float32)
        clipped = np.clip(all_scaled_adv, -5.0, 5.0)

    def _build_policy_batches(
            self,
            transitions: list[Transition],
            credited_action_advantages: list[list[float]],
            base_action_advantages: list[list[float]],
    ) -> list[dict[str, object]]:
        
        transition_items: list[dict[str, object]] = []
        for idx, transition in enumerate(transitions):
            if len(transition.actions) == 0:
                continue
            if idx >= len(credited_action_advantages) or idx >= len(base_action_advantages):
                continue

            credited_adv = credited_action_advantages[idx]
            base_adv = base_action_advantages[idx]
            if len(credited_adv) != len(transition.actions):
                continue
            if len(base_adv) != len(transition.actions):
                continue

            transition_items.append({
                "observations": transition.observations,
                "actions": transition.actions,
                "log_probs": transition.log_probs,
                "credited_adv": credited_adv,
                "base_adv": base_adv,
                "contrib_features": transition.contrib_features,
            })

        if len(transition_items) == 0:
            return []

        perm = torch.randperm(len(transition_items)).tolist()
        shuffled_items = [transition_items[i] for i in perm]

        batches: list[dict[str, object]] = []
        current_items: list[dict[str, object]] = []
        current_tokens = 0

        def flush_current():
            nonlocal current_items, current_tokens
            if len(current_items) == 0:
                return

            observations: list[Observation] = []
            actions: list[int] = []
            log_probs: list[float] = []
            credited_adv_groups: list[list[float]] = []
            base_adv_groups: list[list[float]] = []
            contrib_groups: list[list[list[float]]] = []
            lengths: list[int] = []

            for item in current_items:
                item_observations = item["observations"]
                item_actions = item["actions"]
                item_log_probs = item["log_probs"]
                item_credited_adv = item["credited_adv"]
                item_base_adv = item["base_adv"]
                item_contrib = item["contrib_features"]

                observations.extend(item_observations)
                actions.extend(item_actions)
                log_probs.extend(item_log_probs)
                credited_adv_groups.append(item_credited_adv)
                base_adv_groups.append(item_base_adv)
                contrib_groups.append(item_contrib)
                lengths.append(len(item_actions))

            batches.append({
                "observations": observations,
                "actions": actions,
                "log_probs": log_probs,
                "credited_adv_groups": credited_adv_groups,
                "base_adv_groups": base_adv_groups,
                "contrib_groups": contrib_groups,
                "transition_lengths": lengths,
            })
            current_items = []
            current_tokens = 0

        for item in shuffled_items:
            item_tokens = len(item["actions"])
            if current_items and current_tokens + item_tokens > ModelConfig.policy_batch:
                flush_current()

            current_items.append(item)
            current_tokens += item_tokens

            if current_tokens >= ModelConfig.policy_batch:
                flush_current()

        flush_current()
        return batches

    def save_model(self, model_id: str | int, save_dir: str = BaseConfig.models_dir):
        
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"mappo_agent_{model_id}.pt")

        checkpoint = {

            "policy": self.policy.state_dict(),
            
            "value": self.value.state_dict(),
            
            "params": {},
        }

        torch.save(checkpoint, save_path)
        print(f"[INFO] Model saved to: {save_path}")

    def load_model(self, model_id: str | int, save_dir: str = BaseConfig.models_dir):
        
        load_path = os.path.join(save_dir, f"mappo_agent_{model_id}.pt")
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.value.load_state_dict(checkpoint["value"])

        print(f"Model loaded from: {load_path}")

    def select_action(
            self,
            observation: Observation,
    ):
        
        
        with torch.no_grad():
            ob_features, edge_index, graph_nodes, graph_index, workers_enc = self.observation_to_tensor(observation)
            logits = self.policy(ob_features, edge_index, graph_nodes, graph_index, workers_enc)
            dist = torch.distributions.Categorical(logits=logits)
            action_tensor = dist.sample()
            log_prob = dist.log_prob(action_tensor).item()
            action = action_tensor.item()

        return action, log_prob

    def heuristic_actions(self, subtasks: list[SubTask]):
        

        tasks = list(set(sub.task for sub in subtasks))
        activated_tasks: list[Task] = random.sample(tasks, min(SystemConfig.heuristic_num, len(tasks)))

        actions = [int(sub.task in activated_tasks) for sub in subtasks]

        return actions

    def select_actions(
            self,
            subtasks: list[SubTask],
            get_observation: Callable[[SubTask], Observation],
    ) -> tuple[list[int], list[Observation], list[float]]:
        
        tasks = set(subtask.task for subtask in subtasks)
        ha = self.heuristic_actions(subtasks)
        if SystemConfig.mode != "rl":
            actions = []
            obs = []
            log_probs = []
            for i in range(len(subtasks)):
                obs.append(get_observation(subtasks[i]))
                if SystemConfig.mode == "random":
                    a = random.randint(0, 1)
                elif SystemConfig.mode == "all":
                    a = 1
                elif SystemConfig.mode == "heuristic":
                    a = ha[i]
                else:
                    raise NotImplementedError

                actions.append(a)
                subtasks[i].is_activated = bool(a)
                log_probs.append(1)

            return actions, obs, log_probs


        observations_map: dict[SubTask, Observation] = dict()
        actions_map: dict[SubTask, int] = dict()
        log_probs_map: dict[SubTask, float] = dict()

        for task in tasks:
            unmatched_subtasks = [subtask for subtask in task.subtasks if not subtask.is_matched()]
            unmatched_subtasks = sorted(unmatched_subtasks, key=lambda subtask: subtask.depth)
            for subtask in unmatched_subtasks:
                
                observation = get_observation(subtask)
                observations_map[subtask] = observation
                
                action, log_prob = self.select_action(observation)
                
                subtask.is_activated = bool(action)
                actions_map[subtask] = action
                log_probs_map[subtask] = log_prob

        assert len(observations_map) == len(subtasks) == len(log_probs_map)
        return [actions_map[subtask] for subtask in subtasks], \
            [observations_map[subtask] for subtask in subtasks], \
            [log_probs_map[subtask] for subtask in subtasks]

    def compute_gae(
        self,
        rewards: list[float],
        values: list[float],
        next_values: list[float],
        dones: list[bool],
    ) -> tuple[list[float], list[float]]:
        
        advantages = []
        returns = []

        gae = 0.0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                delta = rewards[t] + self.gamma * next_values[t] - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        return advantages, returns

    def compute_actions_gae(
            self,
            action_rewards: list[list[float]],  
            values: list[float],  
            next_values: list[float],  
            dones: list[bool],  
    ) -> tuple[list[list[float]], list[float]]:
        

        T = len(action_rewards)
        local_reward_deviation_coef = ModelConfig.heuristic_credit_lambda

        
        global_advantages = [0.0] * T
        returns = [0.0] * T

        gae = 0.0
        for t in reversed(range(T)):
            step_reward = sum(action_rewards[t])
            if dones[t]:
                delta = step_reward - values[t]
                gae = delta
            else:
                delta = step_reward + self.gamma * next_values[t] - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae

            global_advantages[t] = gae
            returns[t] = gae + values[t]

        
        action_advantages = []

        for t in range(T):
            rewards_t = action_rewards[t]
            n_t = len(rewards_t)

            if n_t == 0:
                action_advantages.append([])
                continue

            step_reward = sum(rewards_t)
            step_reward_mean = step_reward / n_t
            global_adv_mean = global_advantages[t] / n_t

            advantages_t = [
                global_adv_mean + local_reward_deviation_coef * (r_i - step_reward_mean)
                for r_i in rewards_t
            ]

            action_advantages.append(advantages_t)

        
        
        
        ga, gr = self.compute_gae(
            rewards=[sum(r) for r in action_rewards],
            values=values,
            next_values=next_values,
            dones=dones,
        )
        for t in range(len(action_rewards)):
            if len(action_rewards[t]) == 0:
                
                continue
            assert abs(sum(action_advantages[t]) - ga[t]) < 1e-5
            assert abs(returns[t] - gr[t]) < 1e-5


        return action_advantages, returns


    def learn(self):
        
        if SystemConfig.mode != "rl":
            return

        if len(self.buffer.transitions) == 0:
            return

        # ===== stage1: get training data from buffer =====
        
        transitions = self.buffer.transitions
        
        
        rewards = [t.reward for t in transitions]
        states = [t.state for t in transitions]
        next_states = [t.next_state for t in transitions]

        
        dones = [t.done for t in transitions]

        action_rewards = [t.reward_list.copy() for t in transitions]

        # ===== stage2: estimate state and next_state values with critic =====
        
        with torch.no_grad():
            customs, workers, tasks, subtasks = self.states_to_tensor(states)
            state_values = self.value(customs, workers, tasks, subtasks).squeeze(dim=-1).tolist()

            customs, workers, tasks, subtasks = self.states_to_tensor(next_states)
            next_state_values = self.value(customs, workers, tasks, subtasks).squeeze(dim=-1).tolist()


        # ===== stage3: compute action-level advantages =====
        action_adv, returns = self.compute_actions_gae(action_rewards, state_values, next_state_values, dones)

        base_action_adv = [adv.copy() for adv in action_adv]

        # ===== stage4: write value targets back to transitions =====
        
        for i, transition in enumerate(transitions):
            transition.return_ = returns[i]

        transitions = self.buffer.transitions.copy()

        grad_ratio_list = []

        # ===== stage5: fit critic to current rollout's return target =====
        for epoch in range(ModelConfig.value_epochs):
            # ===== value update =====
            print(f"Value update epoch {epoch}")
            explain_vars = self._update_value(states, returns)
            if epoch > ModelConfig.value_epochs * (1 - np.mean(explain_vars).item()):
                print(f"Value update break in epoch {epoch}")
                break

        # ===== stage6: perform policy update and credit update =====
        for epoch in range(ModelConfig.policy_epochs):
            
            epoch_action_adv = [adv.copy() for adv in base_action_adv]
            if self.enable_credit_net:
                epoch_action_adv = self._apply_credit_adjustment(transitions, epoch_action_adv)

            scaled_action_adv = self._scale_action_advantages(epoch_action_adv)
            assert len(scaled_action_adv) == len(transitions)
            self._log_action_advantages(scaled_action_adv)

            for i, transition in enumerate(transitions):
                transition.action_adv = scaled_action_adv[i]

            policy_batches = self._build_policy_batches(transitions, scaled_action_adv, base_action_adv)

            # ===== policy update =====
            print(f"Policy update epoch {epoch}, batch count: {len(policy_batches)}")
            grad_ratio, epoch_credit_batches = self._update_policy(policy_batches)
            grad_ratio_list.append(grad_ratio)

            # ===== credit update =====
            if self.enable_credit_net:
                print(f"Credit update epoch {epoch}")
                self._update_credit(epoch_credit_batches)

        
        if grad_ratio_list[0] < ModelConfig.grad_ratio_lower_bound:
            self.entropy_coef *= 1 + ModelConfig.entropy_coef_fit
            print("entropy_coef rise:", self.entropy_coef, grad_ratio_list)
        elif grad_ratio_list[0] > ModelConfig.grad_ratio_upper_bound:
            self.entropy_coef *= 1 - ModelConfig.entropy_coef_fit
            print("entropy_coef fall:", self.entropy_coef, grad_ratio_list)
        else:
            print("entropy_coef:", self.entropy_coef, grad_ratio_list)


    def _apply_credit_adjustment(
            self,
            transitions: list[Transition],
            action_advantages: list[list[float]],
    ) -> list[list[float]]:
        
        if not self.enable_credit_net or self.credit is None:
            return action_advantages

        adjusted: list[list[float]] = []
        with torch.no_grad():
            for t_idx, adv_t in enumerate(action_advantages):
                if len(adv_t) == 0:
                    adjusted.append([])
                    continue

                contrib = transitions[t_idx].contrib_features if t_idx < len(transitions) else []
                if len(contrib) != len(adv_t):
                    adjusted.append(adv_t.copy())
                    continue

                adv_tensor = torch.tensor(adv_t, dtype=torch.float32, device=self.device)
                credited, _ = self._credit_adjust_advantages_tensor(adv_tensor, contrib)
                adjusted.append(credited.detach().cpu().tolist())

        return adjusted

    def _update_credit(self, epoch_batches: list[dict[str, object]]):
        
        if not self.enable_credit_net or self.credit is None or self.credit_optimizer is None:
            return

        if len(epoch_batches) == 0:
            return

        policy_params = self._policy_parameters()
        credit_params = [p for p in self.credit.parameters() if p.requires_grad]
        if len(policy_params) == 0 or len(credit_params) == 0:
            return

        meta_terms: list[torch.Tensor] = []
        valid_batch_count = 0

        
        
        self.policy_optimizer.zero_grad(set_to_none=True)
        self.credit_optimizer.zero_grad(set_to_none=True)

        for batch in epoch_batches:
            observations = batch["observations"]
            actions = batch["actions"]
            old_log_probs = batch["log_probs"]
            base_adv_groups = batch["base_adv_groups"]
            contrib_groups = batch["contrib_groups"]
            policy_signal = batch["policy_signal"]

            if len(actions) == 0:
                continue

            differentiable_adv_groups: list[torch.Tensor] = []
            batch_depends_on_credit = False
            for base_adv, contrib in zip(base_adv_groups, contrib_groups):
                base_adv_tensor = torch.tensor(base_adv, dtype=torch.float32, device=self.device)
                credited_adv_tensor, depends_on_credit = self._credit_adjust_advantages_tensor(base_adv_tensor, contrib)
                differentiable_adv_groups.append(credited_adv_tensor * ModelConfig.advantage_scale)
                batch_depends_on_credit = batch_depends_on_credit or depends_on_credit

            if not batch_depends_on_credit:
                continue

            
            ratio, _, _, _ = self._forward_policy_batch(observations, actions, old_log_probs)
            differentiable_adv = torch.cat(differentiable_adv_groups, dim=0)
            surr1 = ratio * differentiable_adv
            surr2 = torch.clamp(
                ratio,
                1.0 - self.clip_epsilon,
                1.0 + self.clip_epsilon
            ) * differentiable_adv
            policy_objective = torch.min(surr1, surr2).sum()

            policy_update_tensors = torch.autograd.grad(
                policy_objective,
                policy_params,
                retain_graph=False,
                create_graph=True,
                allow_unused=True,
            )
            policy_update_vector = self._flatten_policy_tensors(policy_update_tensors)
            meta_terms.append(torch.dot(policy_signal, policy_update_vector))
            valid_batch_count += 1

        if len(meta_terms) == 0:
            return

        meta_objective = torch.stack(meta_terms).mean()
        credit_loss = -meta_objective
        credit_grads = torch.autograd.grad(
            credit_loss,
            credit_params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        for param, grad in zip(credit_params, credit_grads):
            param.grad = None if grad is None else grad.detach()

        torch.nn.utils.clip_grad_norm_(self.credit.parameters(), self.max_grad_norm)
        self.credit_optimizer.step()
        self.policy_optimizer.zero_grad(set_to_none=True)




    def _update_policy(self, policy_batches: list[dict[str, object]]):
        
        if len(policy_batches) == 0:
            return 0.0, []

        batch_nums = []
        batch_grad_ratios = []
        credit_batches: list[dict[str, object]] = []
        policy_params = self._policy_parameters()

        
        
        self._set_module_requires_grad(self.credit, False)

        try:
            for batch in policy_batches:
                obs = batch["observations"]
                acts = batch["actions"]
                old_logp = batch["log_probs"]
                adv_groups = batch["credited_adv_groups"]

                flat_adv = [v for adv in adv_groups for v in adv]
                if len(flat_adv) == 0:
                    continue

                batch_nums.append(len(flat_adv))

                ratio, entropies, log_probs, old_log_probs = self._forward_policy_batch(obs, acts, old_logp)
                action_advantages = torch.tensor(flat_adv, dtype=torch.float32, device=self.device)

                assert len(action_advantages.tolist()) == len(acts) == len(obs) == len(old_logp) == len(flat_adv)

                surr1 = ratio * action_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon
                ) * action_advantages

                policy_objective = torch.min(surr1, surr2).sum()
                sum_policy_loss = -policy_objective
                sum_entropy = entropies.sum()
                loss = sum_policy_loss - self.entropy_coef * sum_entropy

                
                
                policy_grads = torch.autograd.grad(
                    policy_objective,
                    policy_params,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True,
                )
                entropy_grads = torch.autograd.grad(
                    self.entropy_coef * sum_entropy,
                    policy_params,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True,
                )

                policy_signal = self._flatten_policy_tensors(policy_grads).detach()
                entropy_signal = self._flatten_policy_tensors(entropy_grads).detach()
                policy_grad_norm = policy_signal.norm(p=2).item()
                entropy_grad_norm = entropy_signal.norm(p=2).item()
                grad_ratio = entropy_grad_norm / (policy_grad_norm + 1e-8)

                batch_grad_ratios.append(grad_ratio)
          

                self.policy_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.max_grad_norm
                )

                self._policy_update_steps += 1
                self.policy_optimizer.step()



                if self.enable_credit_net:
                    credit_batches.append({
                        "observations": obs,
                        "actions": acts,
                        "log_probs": old_logp,
                        "base_adv_groups": batch["base_adv_groups"],
                        "contrib_groups": batch["contrib_groups"],
                        "policy_signal": policy_signal,
                    })
        finally:
            self._set_module_requires_grad(self.credit, True)

        if len(batch_nums) == 0:
            return 0.0, credit_batches

        avg_grad_ratio = np.sum(
            np.array(batch_nums, dtype=np.float32) * np.array(batch_grad_ratios, dtype=np.float32)
        ).item() / np.sum(batch_nums).item()
        return avg_grad_ratio, credit_batches




    def _update_value(self, states: list[State], returns: list[float]):
        
        perm = torch.randperm(len(states))

        states = [states[i] for i in perm]
        returns = [returns[i] for i in perm]

        batch_explained_vars = []

        
        for start in range(0, len(states), ModelConfig.value_batch):

            end = min(start + ModelConfig.value_batch, len(states))
            b_states = states[start:end]
            b_returns = returns[start:end]

            customs, workers, tasks, subtasks = self.states_to_tensor(b_states)
            b_values = self.value(customs, workers, tasks, subtasks).squeeze(dim=-1)

            b_returns = torch.tensor(b_returns, dtype=torch.float32, device=self.device)

            
            
            value_loss = F.mse_loss(b_values, b_returns)
           
            self._value_update_steps += 1

            with torch.no_grad():
                advantages = b_returns - b_values
                adv_mean = advantages.mean().item()
                adv_std = advantages.std(unbiased=False).item()

                
                var_returns = torch.var(b_returns, unbiased=False)
                explained_var = (1.0 - torch.var(b_returns - b_values, unbiased=False) / var_returns).item()


            self.value_optimizer.zero_grad(set_to_none=True)
            value_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                list(self.value.parameters()),
                self.max_grad_norm
            )
            self.value_optimizer.step()

            batch_explained_vars.append(explained_var)

        return batch_explained_vars
