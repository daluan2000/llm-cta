from __future__ import annotations

from dataclasses import dataclass
import json
import os
import random
from datetime import datetime

from src.config.config import BaseConfig, MetaConfig
from src.llm.client import LLMClient
from src.llm.function_generator import FunctionGenerator, HistoryItem
from src.llm.function_templates import TemplateFunctionSet
from src.training.engine import TrainOptions, TrainingResult, run_training


@dataclass
class CandidateEval:
    spec: TemplateFunctionSet
    result: TrainingResult | None
    score: float
    error: str | None = None


class OuterLoopRunner:


    def __init__(self, seed: int = 0, work_dir: str | None = None, dataset_paths: dict[str, str] | None = None):
        self.seed = seed
        self.rng = random.Random(seed)
        self.generator = FunctionGenerator(seed=seed)
        self.work_dir = work_dir or self._init_work_dir()
        self.code_sets_dir = os.path.join(self.work_dir, "meta_code_sets")
        self.out_path = os.path.join(self.work_dir, "out.txt")
        if dataset_paths is None:
            raise ValueError("OuterLoopRunner requires explicit dataset_paths")
        self.shared_dataset_paths = dataset_paths
        os.makedirs(self.code_sets_dir, exist_ok=True)

    @staticmethod
    def _init_work_dir() -> str:
        ts = datetime.now().strftime("%m%d_%H%M%S")
        folder = os.path.join(BaseConfig.logs_dir, f"meta_{ts}")
        os.makedirs(folder, exist_ok=True)
        return folder

    @staticmethod
    def _run_label(spec: TemplateFunctionSet, short_run: bool) -> str:
        suffix = "short" if short_run else "final"
        return f"{spec.kind}_{spec.round_idx}_{suffix}_{spec.name}"

    @staticmethod
    def _candidate_dir_name(spec: TemplateFunctionSet, short_run: bool) -> str:
        suffix = "short" if short_run else "final"
        return f"run_{spec.kind}_{spec.round_idx}_{suffix}"

    @staticmethod
    def _code_set_filename(spec: TemplateFunctionSet) -> str:
        return f"code_set_{spec.kind}_{spec.round_idx}.json"

    @staticmethod
    def _eval_filename(spec: TemplateFunctionSet) -> str:
        return f"code_set_{spec.kind}_{spec.round_idx}_eval.json"

    @staticmethod
    def _score(result: TrainingResult) -> float:
        return (
            MetaConfig.weight_revenue * result.avg_revenue
            + MetaConfig.weight_success_tasks * result.avg_success_tasks * 100
            + MetaConfig.weight_stability * result.stability * 10000
        )

    def _eval_payload(self, spec: TemplateFunctionSet, item: CandidateEval) -> dict[str, object]:
        return {
            "name": spec.name,
            "kind": spec.kind,
            "round_idx": spec.round_idx,
            "score": item.score,
            "avg_revenue": item.result.avg_revenue if item.result is not None else None,
            "avg_success_tasks": item.result.avg_success_tasks if item.result is not None else None,
            "stability": item.result.stability if item.result is not None else None,
            "log_folder": item.result.log_folder if item.result is not None else None,
            "error": item.error,
            "spec_path": os.path.join(self.code_sets_dir, self._code_set_filename(spec)),
        }

    def _write_code_set_file(self, spec: TemplateFunctionSet) -> str:
        path = os.path.join(self.code_sets_dir, self._code_set_filename(spec))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(spec.to_dict(), f, ensure_ascii=False, indent=2)
        return path

    def _write_eval_file(self, spec: TemplateFunctionSet, item: CandidateEval) -> str:
        path = os.path.join(self.code_sets_dir, self._eval_filename(spec))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._eval_payload(spec, item), f, ensure_ascii=False, indent=2)
        return path

    @staticmethod
    def _to_history(items: list[CandidateEval]) -> list[HistoryItem]:
        out: list[HistoryItem] = []
        for item in items:
            if item.result is None:
                continue
            out.append(
                HistoryItem(
                    name=item.spec.name,
                    score=item.score,
                    avg_revenue=item.result.avg_revenue,
                    avg_success_tasks=item.result.avg_success_tasks,
                    stability=item.result.stability,
                    spec=item.spec.to_dict(),
                )
            )
        return out

    def _evaluate_candidate(self, spec: TemplateFunctionSet, short_run: bool) -> CandidateEval:
        episodes = MetaConfig.short_episodes if short_run else MetaConfig.final_episodes
        candidate_dir = os.path.join(self.work_dir, self._candidate_dir_name(spec, short_run))
        try:
            result = run_training(
                TrainOptions(
                    episodes=episodes,
                    seed=self.seed,
                    enable_llm_augmentation=True,
                    enable_credit_net=True,
                    function_set=spec,
                    log_prefix=f"cand_{spec.name}",
                    log_folder=self.work_dir,
                    dataset_paths=self.shared_dataset_paths,
                    out_file_path=self.out_path,
                    append_out_file=True,
                    csv_output_dir=candidate_dir,
                    run_label=self._run_label(spec, short_run),
                    save_model_id=None,
                )
            )
            score = self._score(result)
            evaluated = CandidateEval(spec=spec, result=result, score=score)
        except Exception as exc:
            phase = "short_run" if short_run else "final_run"
            print(
                f"[Meta Candidate Error] candidate={spec.name} kind={spec.kind} "
                f"round_idx={spec.round_idx} phase={phase} error={type(exc).__name__}: {exc}"
            )
            evaluated = CandidateEval(spec=spec, result=None, score=float("-inf"), error=f"{type(exc).__name__}: {exc}")
        return evaluated

    def run(self) -> dict[str, object]:

        all_items: list[CandidateEval] = []

        with LLMClient.raw_io_log_root(self.work_dir):
            # ===== stage1: generate and evaluate initial function candidates =====
            for idx in range(MetaConfig.init_candidates):
                print(f"[Meta][Initial {idx}] generation start")
                spec = self.generator.generate_initial_candidate(idx)
                print(f"[Meta][Initial {idx}] generation success")
                code_set_path = self._write_code_set_file(spec)
                print(f"[Meta][Initial {idx}] evaluation start")
                item = self._evaluate_candidate(spec, short_run=True)
                self._write_eval_file(spec, item)
                print(f"[Meta][Initial {idx}] evaluation complete")
                all_items.append(item)

            # ===== stage2: iterate and generate new optimized candidates based on history =====
            for round_idx in range(MetaConfig.iterations):
                print(f"[Meta][Iter {round_idx}] generation start")
                ranked = sorted(all_items, key=lambda x: x.score, reverse=True)
                history = self._to_history(ranked[: MetaConfig.topk_history])
                new_spec = self.generator.generate_next_candidate(history, round_idx=round_idx)
                print(f"[Meta][Iter {round_idx}] generation success: name={new_spec.name}")
                code_set_path = self._write_code_set_file(new_spec)
                print(f"[Meta][Iter {round_idx}] code set saved: {code_set_path}")

                print(f"[Meta][Iter {round_idx}] evaluation start")
                item = self._evaluate_candidate(new_spec, short_run=True)
                self._write_eval_file(new_spec, item)
                print(f"[Meta][Iter {round_idx}] evaluation complete: score={item.score}, log_folder={item.result.log_folder if item.result is not None else None}")
                all_items.append(item)

            # ===== stage3: select the best candidate and perform final run =====
            ranked = sorted(all_items, key=lambda x: x.score, reverse=True)
            best_short = ranked[0]
            if best_short.result is None:
                raise RuntimeError(f"all candidate evaluations failed, top error: {best_short.error}")

            final_eval = self._evaluate_candidate(best_short.spec, short_run=False)
            if final_eval.result is None:
                raise RuntimeError(f"final evaluation failed for best candidate: {final_eval.error}")
            self._write_eval_file(best_short.spec, final_eval)
            final_result = final_eval.result

            # ===== stage4: save the best and summary information =====
            best_spec_path = os.path.join(self.work_dir, "best_function_set.json")
            with open(best_spec_path, "w", encoding="utf-8") as f:
                json.dump(best_short.spec.to_dict(), f, ensure_ascii=False, indent=2)

            best_code_set_path = os.path.join(self.code_sets_dir, self._code_set_filename(best_short.spec))

            summary = {
                "meta_work_dir": self.work_dir,
                "code_sets_dir": self.code_sets_dir,
                "best_spec_path": best_spec_path,
                "best_code_set_path": best_code_set_path,
                "best_short_score": best_short.score,
                "best_short_spec": best_short.spec.to_dict(),
                "final_avg_revenue": final_result.avg_revenue,
                "final_avg_success_tasks": final_result.avg_success_tasks,
                "final_stability": final_result.stability,
                "final_log_folder": final_result.log_folder,
            }

            leaderboard = [
                self._eval_payload(item.spec, item)
                for item in sorted(all_items, key=lambda x: x.score, reverse=True)
            ]

            summary_path = os.path.join(self.work_dir, "summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            leaderboard_path = os.path.join(self.work_dir, "leaderboard.json")
            with open(leaderboard_path, "w", encoding="utf-8") as f:
                json.dump(leaderboard, f, ensure_ascii=False, indent=2)

            return summary
