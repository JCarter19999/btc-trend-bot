from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class Fold:
    name: str
    train: pd.DataFrame
    validation: pd.DataFrame

def anchored_folds(frame: pd.DataFrame, timestamp_col: str, validation_months: int = 6, minimum_train_months: int = 24, embargo_bars: int = 144) -> list[Fold]:
    data = frame.sort_values(timestamp_col).reset_index(drop=True)
    start = pd.Timestamp(data[timestamp_col].min()).tz_convert("UTC")
    end = pd.Timestamp(data[timestamp_col].max()).tz_convert("UTC")
    cursor = start + pd.DateOffset(months=minimum_train_months)
    folds: list[Fold] = []
    while cursor < end:
        validation_end = cursor + pd.DateOffset(months=validation_months)
        train = data[data[timestamp_col] < cursor].copy()
        validation = data[(data[timestamp_col] >= cursor) & (data[timestamp_col] < validation_end)].copy()
        if embargo_bars and len(train) > embargo_bars:
            train = train.iloc[:-embargo_bars]
        if not train.empty and not validation.empty:
            folds.append(Fold(f"fold_{len(folds)+1:02d}", train, validation))
        cursor = validation_end
    return folds
