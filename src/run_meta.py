from __future__ import annotations

import os

from src.config.config import BaseConfig, DataRuntimeConfig
from src.meta.outer_loop import OuterLoopRunner
from src.training.engine import _save_config_source_once
from src.utils.output_redirect import redirect_output


"""Meta-learning + LLM outer-loop entry."""


def main() -> None:
    work_dir = OuterLoopRunner._init_work_dir()
    dataset_paths = {
        "subtask_csv": BaseConfig.subtask_csv,
        "task_csv": BaseConfig.task_csv,
        "worker_csv": BaseConfig.worker_csv,
    }
    meta_out = os.path.join(work_dir, "out.txt")

    with redirect_output(meta_out):
        config_source_path = _save_config_source_once(work_dir)
        if config_source_path is not None:
            print(f"[Meta] config_source: {config_source_path}")
        runner = OuterLoopRunner(seed=DataRuntimeConfig.seed, work_dir=work_dir, dataset_paths=dataset_paths)
        print(f"[Meta] work_dir: {runner.work_dir}")
        print(f"[Meta] dataset.subtask_csv: {runner.shared_dataset_paths['subtask_csv']}")
        print(f"[Meta] dataset.task_csv: {runner.shared_dataset_paths['task_csv']}")
        print(f"[Meta] dataset.worker_csv: {runner.shared_dataset_paths['worker_csv']}")

        summary = runner.run()

        print(f"[Meta] work_dir: {summary['meta_work_dir']}")
        print(f"[Meta] best_spec_path: {summary['best_spec_path']}")
        print(f"[Meta] code_sets_dir: {summary['code_sets_dir']}")
        print(f"[Meta] best_code_set_path: {summary['best_code_set_path']}")
        print(f"[Meta] final_avg_revenue: {summary['final_avg_revenue']}")
        print(f"[Meta] final_avg_success_tasks: {summary['final_avg_success_tasks']}")
        print(f"[Meta] final_stability: {summary['final_stability']}")
        print(f"[Meta] final_log_folder: {summary['final_log_folder']}")


if __name__ == "__main__":
    main()
