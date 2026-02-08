from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.metrics import average_precision, competition_score, weighted_log_loss


@dataclass
class TrainArtifacts:
    model: lgb.Booster
    feature_names: List[str]
    categorical_features: List[str]
    categorical_levels: dict
    best_iteration: int


def _infer_categorical_columns(df: pd.DataFrame, candidates: Iterable[str]) -> List[str]:
    cols = []
    for col in candidates:
        if col in df.columns:
            cols.append(col)
    return cols


def _apply_categoricals(
    df: pd.DataFrame,
    categorical_cols: Iterable[str],
    category_levels: Optional[dict] = None,
) -> pd.DataFrame:
    for col in categorical_cols:
        if col not in df.columns:
            continue
        if category_levels and col in category_levels:
            df[col] = df[col].astype(pd.CategoricalDtype(categories=category_levels[col]))
        else:
            df[col] = df[col].astype("category")
    return df


def _default_params(seed: int, scale_pos_weight: float) -> dict:
    return {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "learning_rate": 0.05,
        "num_leaves": 255,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "max_depth": -1,
        "max_bin": 255,
        "lambda_l2": 1.0,
        "scale_pos_weight": scale_pos_weight,
        "seed": seed,
        "verbosity": -1,
        "num_threads": -1,
    }


def train_lgbm(
    train_path: str | Path,
    target_col: str,
    id_col: str,
    categorical_cols: Optional[List[str]] = None,
    valid_frac: float = 0.0,
    seed: int = 42,
    num_boost_round: int = 800,
    early_stopping_rounds: int = 50,
    params: Optional[dict] = None,
) -> Tuple[TrainArtifacts, Optional[dict]]:
    train_path = Path(train_path)

    train_df = pd.read_parquet(train_path)

    if target_col not in train_df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    if id_col in train_df.columns:
        train_df = train_df.drop(columns=[id_col])

    y = train_df[target_col].values
    X = train_df.drop(columns=[target_col])

    if categorical_cols is None:
        categorical_cols = _infer_categorical_columns(
            X,
            candidates=["gender", "age_group", "inventory_id", "day_of_week", "hour", "seq"],
        )

    category_levels = {}
    if categorical_cols:
        X = _apply_categoricals(X, categorical_cols)
        for col in categorical_cols:
            if col in X.columns and hasattr(X[col], "cat"):
                category_levels[col] = X[col].cat.categories

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    scale_pos_weight = float(n_neg / max(n_pos, 1))

    lgb_params = _default_params(seed=seed, scale_pos_weight=scale_pos_weight)
    if params:
        lgb_params.update(params)

    valid_info = None
    if valid_frac and valid_frac > 0:
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=valid_frac, random_state=seed, stratify=y
        )
        lgb_train = lgb.Dataset(
            X_train,
            label=y_train,
            categorical_feature=categorical_cols or "auto",
            free_raw_data=False,
        )
        lgb_valid = lgb.Dataset(
            X_valid,
            label=y_valid,
            categorical_feature=categorical_cols or "auto",
            free_raw_data=False,
        )
        model = lgb.train(
            lgb_params,
            lgb_train,
            num_boost_round=num_boost_round,
            valid_sets=[lgb_train, lgb_valid],
            valid_names=["train", "valid"],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=100,
        )
        preds = model.predict(X_valid, num_iteration=model.best_iteration)
        valid_info = {
            "average_precision": average_precision(y_valid, preds),
            "weighted_logloss": weighted_log_loss(y_valid, preds),
            "competition_score": competition_score(y_valid, preds),
        }
    else:
        lgb_train = lgb.Dataset(
            X,
            label=y,
            categorical_feature=categorical_cols or "auto",
            free_raw_data=False,
        )
        model = lgb.train(
            lgb_params,
            lgb_train,
            num_boost_round=num_boost_round,
            valid_sets=[lgb_train],
            valid_names=["train"],
            verbose_eval=200,
        )

    feature_names = list(X.columns)

    del train_df, X
    gc.collect()

    artifacts = TrainArtifacts(
        model=model,
        feature_names=feature_names,
        categorical_features=categorical_cols or [],
        categorical_levels=category_levels,
        best_iteration=int(model.best_iteration or num_boost_round),
    )
    return artifacts, valid_info


def predict_lgbm(
    model: lgb.Booster,
    test_path: str | Path,
    id_col: str,
    categorical_cols: Optional[List[str]] = None,
    category_levels: Optional[dict] = None,
) -> Tuple[pd.Series, pd.Series]:
    test_path = Path(test_path)
    test_df = pd.read_parquet(test_path)

    if id_col not in test_df.columns:
        raise ValueError(f"Missing id column in test: {id_col}")

    test_ids = test_df[id_col].copy()
    X_test = test_df.drop(columns=[id_col])

    if categorical_cols:
        X_test = _apply_categoricals(X_test, categorical_cols, category_levels=category_levels)

    preds = model.predict(X_test, num_iteration=model.best_iteration)
    return test_ids, pd.Series(preds, name="clicked")
