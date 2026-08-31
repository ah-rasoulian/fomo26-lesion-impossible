from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import optuna
from sklearn.metrics import log_loss, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--checkpoint-run-id", required=True)
    parser.add_argument("--n-trials", type=int, default=15)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--final-samples-per-epoch",
        type=int,
        default=84,
    )
    parser.add_argument("--work-dir", default="optuna_runs")
    parser.add_argument("--skip-refit", action="store_true")
    parser.add_argument("--minimum-refit-epochs", type=int, default=10)
    parser.add_argument("--refit-epoch-multiplier", type=float, default=1.5)
    parser.add_argument("--maximum-refit-epochs", type=int, default=40)
    parser.add_argument("override", nargs="*")
    return parser.parse_args()


def run_job(
    script: Path,
    config_name: str,
    overrides: list[str],
) -> None:
    command = [
        sys.executable,
        str(script),
        "--config-name",
        config_name,
        *overrides,
    ]
    print("\nRunning command:")
    print(" ".join(command))
    subprocess.run(command, check=True)


def validate_fold_result(
    result: dict,
    fold: int,
) -> tuple[list[int], list[float], int, float]:
    required = {
        "labels",
        "probabilities",
        "best_epoch",
        "best_val_loss",
    }
    missing = required.difference(result)
    if missing:
        raise RuntimeError(
            f"Fold {fold} result is missing {sorted(missing)}."
        )

    labels = [int(value) for value in result["labels"]]
    probabilities = [
        float(value) for value in result["probabilities"]
    ]
    best_epoch = int(result["best_epoch"])
    best_val_loss = float(result["best_val_loss"])

    if not labels or len(labels) != len(probabilities):
        raise RuntimeError(
            f"Fold {fold} has inconsistent labels/probabilities."
        )
    if any(label not in (0, 1) for label in labels):
        raise RuntimeError(
            f"Fold {fold} contains non-binary labels: {labels}."
        )
    if len(set(labels)) == 1:
        print(
            f"Warning: fold {fold} contains only class {labels[0]}. "
            "Its AUROC is undefined and will be omitted from fold-level "
            "summaries, but its predictions will remain in the pooled "
            "objective."
        )
    if any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in probabilities
    ):
        raise RuntimeError(
            f"Fold {fold} produced invalid probabilities."
        )
    if not math.isfinite(best_val_loss) or best_epoch < 1:
        raise RuntimeError(
            f"Fold {fold} produced invalid checkpoint metadata."
        )

    return labels, probabilities, best_epoch, best_val_loss


def main() -> None:
    args = parse_args()
    if args.minimum_refit_epochs < 1:
        raise ValueError("--minimum-refit-epochs must be positive.")
    if args.refit_epoch_multiplier < 1.0:
        raise ValueError("--refit-epoch-multiplier must be at least 1.")
    if args.maximum_refit_epochs < args.minimum_refit_epochs:
        raise ValueError(
            "--maximum-refit-epochs cannot be smaller than "
            "--minimum-refit-epochs."
        )
    script = Path(__file__).with_name("finetune_cls.py").resolve()

    # A versioned directory prevents mixing the old mean-fold-AUROC study
    # with this pooled out-of-fold AUROC objective.
    root = (
        Path(args.work_dir).resolve()
        / args.config_name
        / "pooled_oof_80_20_v3"
    )
    root.mkdir(parents=True, exist_ok=True)

    storage = f"sqlite:///{(root / 'study.sqlite3').as_posix()}"
    study = optuna.create_study(
        study_name=f"{args.config_name}_vit_s_pooled_oof_80_20_v3",
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=args.seed,
            n_startup_trials=5,
        ),
    )

    def objective(trial: optuna.Trial) -> float:
        blocks = trial.suggest_categorical(
            "trainable_backbone_blocks",
            [0, 1, 2],
        )
        head_lr = trial.suggest_float(
            "finetune_lr",
            3e-5,
            1e-3,
            log=True,
        )
        weight_decay = trial.suggest_float(
            "finetune_weight_decay",
            1e-5,
            1e-2,
            log=True,
        )
        backbone_multiplier = (
            trial.suggest_categorical(
                "backbone_lr_multiplier",
                [0.003, 0.01, 0.03],
            )
            if blocks > 0
            else 0.0
        )

        all_labels: list[int] = []
        all_probabilities: list[float] = []
        fold_aurocs: list[float | None] = []
        fold_losses: list[float] = []
        fold_epochs: list[int] = []
        fold_sizes: list[int] = []

        trial_dir = root / f"trial_{trial.number:03d}"
        for fold in range(args.n_folds):
            fold_dir = trial_dir / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            result_path = fold_dir / "result.json"

            run_job(
                script,
                args.config_name,
                [
                    f"checkpoint_run_id={args.checkpoint_run_id}",
                    f"data.fold={fold}",
                    "optimization.enabled=true",
                    f"optimization.result_path={result_path.as_posix()}",
                    f"model.trainable_backbone_blocks={blocks}",
                    f"model.finetune_lr={head_lr}",
                    f"model.finetune_weight_decay={weight_decay}",
                    f"model.backbone_lr_multiplier={backbone_multiplier}",
                    f"training.seed={args.seed}",
                    "logger.wandb_logging=false",
                    "logger.mlflow_logging=false",
                    "logger.log_images_every_n_epoch=0",
                    f"hydra.run.dir={fold_dir.as_posix()}",
                    *args.override,
                ],
            )

            if not result_path.is_file():
                raise RuntimeError(
                    f"Fold {fold} did not create {result_path}."
                )
            result = json.loads(
                result_path.read_text(encoding="utf-8")
            )
            labels, probabilities, epoch, loss = (
                validate_fold_result(result, fold)
            )
            fold_auroc = (
                float(roc_auc_score(labels, probabilities))
                if len(set(labels)) == 2
                else None
            )

            all_labels.extend(labels)
            all_probabilities.extend(probabilities)
            fold_aurocs.append(fold_auroc)
            fold_losses.append(loss)
            fold_epochs.append(epoch)
            fold_sizes.append(len(labels))

            trial.set_user_attr(f"fold_{fold}_auroc", fold_auroc)
            trial.set_user_attr(f"fold_{fold}_best_val_loss", loss)
            trial.set_user_attr(f"fold_{fold}_best_epoch", epoch)

        if len(set(all_labels)) == 2:
            pooled_auroc = float(
                roc_auc_score(all_labels, all_probabilities)
            )
        else:
            pooled_auroc = 0.5
            print(
                "Warning: pooled validation labels contain only one class. "
                "Using the non-informative AUROC value 0.5 for this trial."
            )
        pooled_log_loss = float(
            log_loss(
                all_labels,
                all_probabilities,
                labels=[0, 1],
            )
        )
        median_best_epoch = float(statistics.median(fold_epochs))
        expanded_refit_epochs = math.ceil(
            median_best_epoch * args.refit_epoch_multiplier
        )
        refit_epochs = min(
            args.maximum_refit_epochs,
            max(args.minimum_refit_epochs, expanded_refit_epochs),
        )

        valid_fold_aurocs = [
            value for value in fold_aurocs if value is not None
        ]
        mean_fold_auroc = (
            float(statistics.fmean(valid_fold_aurocs))
            if valid_fold_aurocs
            else None
        )
        std_fold_auroc = (
            float(statistics.pstdev(valid_fold_aurocs))
            if len(valid_fold_aurocs) > 1
            else (0.0 if len(valid_fold_aurocs) == 1 else None)
        )

        trial.set_user_attr("pooled_oof_auroc", pooled_auroc)
        trial.set_user_attr("pooled_oof_log_loss", pooled_log_loss)
        trial.set_user_attr("oof_labels", all_labels)
        trial.set_user_attr("oof_probabilities", all_probabilities)
        trial.set_user_attr("fold_aurocs", fold_aurocs)
        trial.set_user_attr(
            "mean_fold_auroc",
            mean_fold_auroc,
        )
        trial.set_user_attr(
            "std_fold_auroc",
            std_fold_auroc,
        )
        trial.set_user_attr("fold_best_val_losses", fold_losses)
        trial.set_user_attr("fold_best_epochs", fold_epochs)
        trial.set_user_attr("fold_sizes", fold_sizes)
        trial.set_user_attr("median_fold_best_epoch", median_best_epoch)
        trial.set_user_attr("refit_epochs", refit_epochs)

        print(
            json.dumps(
                {
                    "trial": trial.number,
                    "pooled_oof_auroc": pooled_auroc,
                    "pooled_oof_log_loss": pooled_log_loss,
                    "fold_aurocs": fold_aurocs,
                    "fold_best_epochs": fold_epochs,
                    "refit_epochs": refit_epochs,
                },
                indent=2,
            )
        )
        return pooled_auroc

    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=1,
    )

    best = study.best_trial
    summary = {
        "objective": "pooled_oof_auroc",
        "best_trial": best.number,
        "pooled_oof_auroc": best.value,
        "pooled_oof_log_loss": best.user_attrs[
            "pooled_oof_log_loss"
        ],
        "mean_fold_auroc": best.user_attrs["mean_fold_auroc"],
        "std_fold_auroc": best.user_attrs["std_fold_auroc"],
        "fold_aurocs": best.user_attrs["fold_aurocs"],
        "fold_best_val_losses": best.user_attrs[
            "fold_best_val_losses"
        ],
        "fold_best_epochs": best.user_attrs["fold_best_epochs"],
        "fold_sizes": best.user_attrs["fold_sizes"],
        "median_fold_best_epoch": best.user_attrs[
            "median_fold_best_epoch"
        ],
        "oof_labels": best.user_attrs["oof_labels"],
        "oof_probabilities": best.user_attrs[
            "oof_probabilities"
        ],
        "refit_epochs": best.user_attrs["refit_epochs"],
        "parameters": best.params,
    }
    (root / "best_trial.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print("\nBest trial:")
    print(json.dumps(summary, indent=2))

    if args.skip_refit:
        return

    params = dict(best.params)
    if int(params["trainable_backbone_blocks"]) == 0:
        params["backbone_lr_multiplier"] = 0.0

    refit_dir = root / "final_refit"
    refit_dir.mkdir(parents=True, exist_ok=True)
    refit_result = refit_dir / "result.json"
    run_job(
        script,
        args.config_name,
        [
            f"checkpoint_run_id={args.checkpoint_run_id}",
            "data.fold=0",
            "training.refit_full_data=true",
            f"training.epochs={best.user_attrs['refit_epochs']}",
            f"training.samples_per_epoch={args.final_samples_per_epoch}",
            f"training.steps_per_epoch={args.final_samples_per_epoch}",
            f"model.trainable_backbone_blocks={params['trainable_backbone_blocks']}",
            f"model.finetune_lr={params['finetune_lr']}",
            f"model.finetune_weight_decay={params['finetune_weight_decay']}",
            f"model.backbone_lr_multiplier={params['backbone_lr_multiplier']}",
            "optimization.enabled=false",
            f"optimization.result_path={refit_result.as_posix()}",
            f"hydra.run.dir={refit_dir.as_posix()}",
            *args.override,
        ],
    )
    if not refit_result.is_file():
        raise RuntimeError(
            "Final refit did not produce its result file."
        )
    print("\nFinal refit:")
    print(refit_result.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
