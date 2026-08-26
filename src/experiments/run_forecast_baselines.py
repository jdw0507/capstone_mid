from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.csv_adapter import make_model_matrices
from src.data.splitters import slice_by_split
from src.evaluation.forecast_metrics import (
    ForecastMetricResult,
    build_forecast_evaluation_frame,
    summarize_forecast_metrics,
)


@dataclass
class ForecastBaselineRunResult:
    per_prediction: pd.DataFrame
    metric_result: ForecastMetricResult
    feature_contributions: pd.DataFrame | None = None


def _extract_feature_contributions(
    model,
    model_name: str,
    split_row,
    evaluation_split: str,
) -> pd.DataFrame | None:
    metric_name = None
    values = None

    if hasattr(model, "coef_") and model.coef_ is not None:
        metric_name = "coef"
        values = model.coef_
    elif hasattr(model, "feature_importances_") and model.feature_importances_ is not None:
        metric_name = "feature_importance"
        values = model.feature_importances_
    elif hasattr(getattr(model, "model", None), "feature_importances_"):
        raw = model.model.feature_importances_
        if raw is not None:
            metric_name = "feature_importance"
            values = raw

    if values is None:
        return None

    if isinstance(values, pd.Series):
        s = pd.to_numeric(values, errors="coerce").dropna()
    else:
        arr = np.asarray(values).reshape(-1)
        cols = getattr(model, "fitted_feature_columns_", None)
        if cols is None or len(cols) != len(arr):
            cols = [f"feature_{i}" for i in range(len(arr))]
        s = pd.Series(arr, index=cols)
        s = pd.to_numeric(s, errors="coerce").dropna()

    if s.empty:
        return None

    abs_s = s.abs()
    abs_sum = float(abs_s.sum())
    share = abs_s / abs_sum if abs_sum > 0 else pd.Series(0.0, index=s.index)

    out = pd.DataFrame(
        {
            "feature": s.index.astype(str),
            "contribution_raw": s.values.astype(float),
            "contribution_abs": abs_s.values.astype(float),
            "contribution_share_abs": share.values.astype(float),
        }
    ).sort_values("contribution_abs", ascending=False)

    out["contribution_rank"] = np.arange(1, len(out) + 1, dtype=int)
    out["metric_name"] = metric_name
    out["model"] = model_name
    out["fold_id"] = int(split_row.fold_id)
    out["evaluation_split"] = evaluation_split
    out["model_train_start"] = split_row.model_train_start
    out["model_train_end"] = split_row.model_train_end
    out["view_start"] = split_row.view_start
    out["view_end"] = split_row.view_end
    out["test_start"] = split_row.test_start
    out["test_end"] = split_row.test_end

    return out.reset_index(drop=True)


def run_forecast_baselines_on_splits(
    panel_df: pd.DataFrame,
    splits: pd.DataFrame,
    forecasters: dict[str, object],
    target_col: str = "target_5d_excess",
    evaluation_split: str = "view_build",
    feature_cols: list[str] | None = None,
    exclude_target_like: bool = True,
    date_col: str = "date",
    verbose_progress: bool = True,
) -> ForecastBaselineRunResult:
    """
    nested split 위에서 forecasting baseline 모델들을 실행

    Parameters
    ----------
    panel_df : pd.DataFrame
        asset-panel 데이터
    splits : pd.DataFrame
        build_nested_walk_forward_splits() 결과
    forecasters : dict[str, BaseForecaster]
    target_col : str
        예측 타깃 컬럼명
    evaluation_split : {"view_build", "test"}
        어떤 구간에서 예측을 평가할지 선택
    verbose_progress : bool
        fold/model 진행상황 로그 출력 여부
    """
    allowed = {"view_build", "test"}
    if evaluation_split not in allowed:
        raise ValueError(f"evaluation_split은 {allowed} 중 하나여야 합니다.")

    all_eval_rows = []
    all_feature_rows = []

    n_folds = len(splits)
    n_models = len(forecasters)

    for split_idx, split_row in enumerate(splits.itertuples(index=False), start=1):
        split_dict = split_row._asdict()
        parts = slice_by_split(panel_df, pd.Series(split_dict), date_col=date_col)

        train_df = parts["model_train"]
        eval_df = parts[evaluation_split]

        X_train, y_train, meta_train = make_model_matrices(
            train_df,
            target_col=target_col,
            feature_cols=feature_cols,
            exclude_target_like=exclude_target_like,
        )
        X_eval, y_eval, meta_eval = make_model_matrices(
            eval_df,
            target_col=target_col,
            feature_cols=feature_cols,
            exclude_target_like=exclude_target_like,
        )

        if verbose_progress:
            print(
                f"\n=== Fold {split_idx}/{n_folds} | fold_id={split_row.fold_id} "
                f"| train: {split_row.model_train_start.date()} ~ {split_row.model_train_end.date()} "
                f"| eval({evaluation_split}): "
                f"{(split_row.view_start if evaluation_split == 'view_build' else split_row.test_start).date()} ~ "
                f"{(split_row.view_end if evaluation_split == 'view_build' else split_row.test_end).date()} ==="
            )
            print(
                f"train_rows={len(train_df)}, eval_rows={len(eval_df)}, "
                f"train_samples={len(X_train)}, eval_samples={len(X_eval)}"
            )

        for model_idx, (model_name, forecaster) in enumerate(forecasters.items(), start=1):
            if verbose_progress:
                print(f"[{split_idx}/{n_folds}] Model {model_idx}/{n_models}: {model_name}")

            model = deepcopy(forecaster)

            model.fit(X_train, y_train, meta_train)
            feat_df = _extract_feature_contributions(
                model=model,
                model_name=model_name,
                split_row=split_row,
                evaluation_split=evaluation_split,
            )
            if feat_df is not None and not feat_df.empty:
                all_feature_rows.append(feat_df)

            pred = model.predict(X_eval, meta_eval)
            unc = model.predict_uncertainty(X_eval, meta_eval)

            eval_long = build_forecast_evaluation_frame(
                y_true=y_eval,
                y_pred=pred,
                model_name=model_name,
                meta=meta_eval,
                uncertainty=unc,
            )

            eval_long["fold_id"] = split_row.fold_id
            eval_long["evaluation_split"] = evaluation_split

            eval_long["model_train_start"] = split_row.model_train_start
            eval_long["model_train_end"] = split_row.model_train_end
            eval_long["view_start"] = split_row.view_start
            eval_long["view_end"] = split_row.view_end
            eval_long["test_start"] = split_row.test_start
            eval_long["test_end"] = split_row.test_end

            all_eval_rows.append(eval_long)

    if len(all_eval_rows) == 0:
        raise ValueError("예측 평가 결과가 생성되지 않았습니다.")

    per_prediction = pd.concat(all_eval_rows, axis=0, ignore_index=True)
    metric_result = summarize_forecast_metrics(per_prediction)
    feature_contributions = (
        pd.concat(all_feature_rows, axis=0, ignore_index=True)
        if all_feature_rows
        else pd.DataFrame(
            columns=[
                "feature",
                "contribution_raw",
                "contribution_abs",
                "contribution_share_abs",
                "contribution_rank",
                "metric_name",
                "model",
                "fold_id",
                "evaluation_split",
                "model_train_start",
                "model_train_end",
                "view_start",
                "view_end",
                "test_start",
                "test_end",
            ]
        )
    )

    return ForecastBaselineRunResult(
        per_prediction=per_prediction,
        metric_result=metric_result,
        feature_contributions=feature_contributions,
    )
