from __future__ import annotations
import os
from pathlib import Path

import torch





def _load_project_env() -> None:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _get_required_env(name: str) -> str:
    _load_project_env()
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}. Please set it in .env")
    return value


class BaseConfig:

    base_path = _get_required_env("LLM_CTA_BASE_PATH")
    worker_csv = os.path.join(base_path, "data/worker.csv")
    task_csv = os.path.join(base_path, "data/task.csv")
    subtask_csv = os.path.join(base_path, "data/subtask.csv")
    models_dir = os.path.join(base_path, "models")
    logs_dir = os.path.join(base_path, "logs")

    active_scaling_profile = "w300_s2000"


class LLMConfig:


    OPENAI_API_KEY = _get_required_env("OPENAI_API_KEY")
    OPENAI_BASE_URL = _get_required_env("OPENAI_BASE_URL")
    OPENAI_MODEL = _get_required_env("OPENAI_MODEL")
    MAX_NEW_TOKENS = 4096  
    ENABLE_THINKING = False
    ENABLE_JSON_MODE = True
    DISABLE_THINKING = True
    REQUIRE_FINAL_ONLY_OUTPUT = True
    NO_THINK_DIRECTIVE = "/no_think"
    STRICT_REASONING_TEXT_VALIDATION = True



    


class SystemConfig:

    time_window_size = 600
    scale_time_span = False 
    time_span = 3600 * 8

    mode = "rl" 
    heuristic_num = 2
    dynamic_worker_appearance = True

    episodes = 500

    skill_nums = 12


class DataRuntimeConfig:
    seed = 42


class ModelConfig:
    skill_encoder_output_dim = 2
    skill_projection_matrix = torch.randn(SystemConfig.skill_nums, skill_encoder_output_dim)
    skill_projection_matrix = skill_projection_matrix / skill_projection_matrix.norm(dim=0)

    task_dim = 4
    worker_base_dim = 1
    subtask_base_dim = 5
    worker_dim = worker_base_dim + skill_encoder_output_dim  # non-skill(worker_base_dim) + skill encoded
    subtask_dim = subtask_base_dim + skill_encoder_output_dim  # non-skill(subtask_base_dim) + skill encoded

    subtask_gat_hidden_dim = 16
    subtask_gat_output_dim = subtask_dim
    subtask_gat_heads = 2

    enable_llm_augmentation = False
    enable_credit_net = False

    base_observation_custom_dim = 4
    base_state_custom_dim = 4

    
    llm_state_aug_dim = 8           
    llm_observation_aug_dim = 6     
    llm_contribution_dim = 8        

    # CreditNet
    credit_hidden_dim = 32
    credit_lr = 1e-4 
    credit_alpha = 0.2 


    observation_custom_dim = base_observation_custom_dim + llm_observation_aug_dim
    state_custom_dim = base_state_custom_dim + llm_state_aug_dim

    state_dim = state_custom_dim + task_dim * 2 + worker_dim * 2 + subtask_dim * 2
    policy_input_dim = (
        observation_custom_dim
        + task_dim
        + subtask_dim
        + worker_dim * 2
        + subtask_gat_output_dim
    )
    policy_output_dim = 2
    policy_hidden_dim = 64

    value_input_dim = state_dim
    value_hidden_dim = 64

    policy_epochs = 2
    value_epochs = 100

    value_batch = 6400   
    policy_batch = 512


    penalty_matched_failed = 0.05 
    penalty_unmatched = 0.05 



   
    profiles = {
        "w300_s2000": {
            "scales": {
                "nl_time": 5000,
                "nl_skill_nums": 10,
                "nl_global_workers": 50,
                "nl_global_tasks": 10,
                "nl_global_subtasks": 50,
                "nl_reward": 1000,
                "nl_subtasks": 10,
                "nl_workers": 25,
            },
        },

    }
    @staticmethod
    def nl_time() -> float:
        return ModelConfig.profiles[BaseConfig.active_scaling_profile]["scales"]["nl_time"]

    @staticmethod
    def nl_skill_nums() -> float:
        return ModelConfig.profiles[BaseConfig.active_scaling_profile]["scales"]["nl_skill_nums"]

    @staticmethod
    def nl_global_workers() -> float:
        return ModelConfig.profiles[BaseConfig.active_scaling_profile]["scales"]["nl_global_workers"]

    @staticmethod
    def nl_global_tasks() -> float:
        return ModelConfig.profiles[BaseConfig.active_scaling_profile]["scales"]["nl_global_tasks"]

    @staticmethod
    def nl_global_subtasks() -> float:
        return ModelConfig.profiles[BaseConfig.active_scaling_profile]["scales"]["nl_global_subtasks"]

    @staticmethod
    def nl_reward() -> float:
        return ModelConfig.profiles[BaseConfig.active_scaling_profile]["scales"]["nl_reward"]

    @staticmethod
    def nl_subtasks() -> float:
        return ModelConfig.profiles[BaseConfig.active_scaling_profile]["scales"]["nl_subtasks"]

    @staticmethod
    def nl_workers() -> float:
        return ModelConfig.profiles[BaseConfig.active_scaling_profile]["scales"]["nl_workers"]

    advantage_scale = 0.03

    gamma = 0.99 
    gae_lambda = 0.95 
  
    heuristic_credit_lambda = 1.0 
    clip_epsilon = 0.1 
    policy_lr = 2e-4 
    value_lr = 5e-4 
    value_coef = 0.5
    entropy_coef = 0.01  
    max_grad_norm = 0.5
    entropy_coef_fit = 0.05    
    grad_ratio_lower_bound = 0.03  
    grad_ratio_upper_bound = 0.1  

class MetaConfig:
    """外层LLM迭代优化配置。"""
    generator_retry = 2
    generator_temperature = 0.75  #------------------大模型输出温度

    topk_history = 3 #------------------ llm生成函数时,采样历史函数个数topk
    init_candidates = topk_history
    iterations = 500 # -------------------- llm最大迭代生成次数$I_{max}$

    short_episodes = 50
    final_episodes = SystemConfig.episodes

    weight_revenue = 0.8
    weight_success_tasks = 0.15
    weight_stability = 0.05

    


class RunConfig:

    enable_feature_stats_print = True
    feature_stats_episode_interval = 0
