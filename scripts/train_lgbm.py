from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.lgbm_pipeline import predict_lgbm, train_lgbm


def _parse_params(param_items: list[str]) -> dict:
    params: dict = {}
    for item in param_items:
        if "=" not in item:
            raise ValueError(f"Invalid param format: {item}. Use key=value.")
        key, value = item.split("=", 1)
        value = value.strip()
        if value.lower() in {"true", "false"}:
            parsed: object = value.lower() == "true"
        else:
            try:
                parsed = int(value)
            except ValueError:
                try:
                    parsed = float(value)
                except ValueError:
                    parsed = value
        params[key.strip()] = parsed
    return params


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM for CTR prediction")
    parser.add_argument("--train-path", type=Path, default=Path("data/train/train.parquet"))
    parser.add_argument("--test-path", type=Path, default=Path("data/test/test.parquet"))
    parser.add_argument(
        "--submission-template",
        type=Path,
        default=Path("data/submission/sample_submission.csv"),
    )
    parser.add_argument("--output-model", type=Path, default=Path("models/lgbm_model.txt"))
    parser.add_argument(
        "--output-submission", type=Path, default=Path("reports/submission_lgbm.csv")
    )
    parser.add_argument("--output-metrics", type=Path, default=Path("reports/metrics_lgbm.json"))
    parser.add_argument("--target-col", type=str, default="clicked")
    parser.add_argument("--id-col", type=str, default="ID")
    parser.add_argument("--valid-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-boost-round", type=int, default=3000)
    parser.add_argument("--early-stopping-rounds", type=int, default=200)
    parser.add_argument(
        "--categorical-cols",
        type=str,
        default="",
        help="Comma-separated categorical columns. Leave empty to use defaults.",
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="Override LightGBM params with key=value. Can be used multiple times.",
    )
    args = parser.parse_args()

    categorical_cols = None
    if args.categorical_cols:
        categorical_cols = [c.strip() for c in args.categorical_cols.split(",") if c.strip()]

    param_overrides = _parse_params(args.param)

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.output_submission.parent.mkdir(parents=True, exist_ok=True)
    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)

    print("[1/3] Training model...")
    artifacts, valid_info = train_lgbm(
        train_path=args.train_path,
        target_col=args.target_col,
        id_col=args.id_col,
        categorical_cols=categorical_cols,
        valid_frac=args.valid_frac,
        seed=args.seed,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        params=param_overrides or None,
    )

    artifacts.model.save_model(str(args.output_model))

    print("[2/3] Predicting test set...")
    test_ids, preds = predict_lgbm(
        model=artifacts.model,
        test_path=args.test_path,
        id_col=args.id_col,
        categorical_cols=artifacts.categorical_features,
        category_levels=artifacts.categorical_levels,
        frequency_maps=artifacts.frequency_maps,
    )

    sub = pd.read_csv(args.submission_template)
    if args.id_col not in sub.columns:
        raise ValueError(f"Missing id column in submission template: {args.id_col}")
    if args.target_col in sub.columns:
        sub = sub.drop(columns=[args.target_col])

    pred_df = pd.DataFrame({args.id_col: test_ids, args.target_col: preds})
    sub = sub.merge(pred_df, on=args.id_col, how="left")

    sub.to_csv(args.output_submission, index=False)

    print("[3/3] Saving metrics...")
    metrics_payload = {
        "best_iteration": artifacts.best_iteration,
        "categorical_features": artifacts.categorical_features,
        "frequency_features": [f"{c}__freq" for c in artifacts.categorical_features],
    }
    if valid_info:
        metrics_payload.update(valid_info)

    with args.output_metrics.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)


if __name__ == "__main__":
    main()
