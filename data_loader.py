"""
Data loading and validation module for Prompt Evaluator.
Handles loading questions.csv and custom uploaded datasets with robust validation.
"""

from typing import Tuple, List, Optional, Union, BinaryIO
import io
import os
import pandas as pd

REQUIRED_COLUMNS = ["id", "topic", "question", "answer_current", "answer_new"]
EXPECTED_TOPICS = {"orders", "refunds", "account", "shipping", "product"}


class DataValidationError(Exception):
    """Custom exception raised when dataset fails schema or data validation."""
    pass


def load_dataset(
    source: Union[str, os.PathLike, BinaryIO, io.StringIO],
) -> Tuple[pd.DataFrame, dict]:
    """
    Load and validate the questions dataset.
    
    Args:
        source: File path (str/Path) or file-like object (UploadedFile or StringIO).
        
    Returns:
        Tuple[pd.DataFrame, dict]: Validated DataFrame and a validation metadata dict.
        
    Raises:
        DataValidationError: If schema or critical data integrity requirements are violated.
    """
    try:
        if isinstance(source, (str, os.PathLike)):
            if not os.path.exists(source):
                raise FileNotFoundError(f"Dataset file not found at: {source}")
            df = pd.read_csv(source, encoding="utf-8")
        else:
            df = pd.read_csv(source, encoding="utf-8")
    except Exception as e:
        raise DataValidationError(f"Failed to read CSV dataset: {str(e)}") from e

    validation_meta = validate_dataset(df)
    
    # Clean text strings
    for col in REQUIRED_COLUMNS:
        df[col] = df[col].astype(str).str.strip()

    return df, validation_meta


def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Perform deep validation of dataset columns, nulls, and data types.
    
    Args:
        df: Pandas DataFrame to validate.
        
    Returns:
        dict: Validation report containing counts, missing flags, warnings.
    """
    columns = list(df.columns)
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing_cols:
        raise DataValidationError(
            f"Dataset is missing required columns: {missing_cols}. "
            f"Expected: {REQUIRED_COLUMNS}"
        )

    # Check row count
    row_count = len(df)
    if row_count == 0:
        raise DataValidationError("The uploaded dataset contains 0 rows.")

    # Check for null / empty values
    null_counts = {col: int(df[col].isnull().sum()) for col in REQUIRED_COLUMNS}
    empty_counts = {
        col: int((df[col].astype(str).str.strip() == "").sum()) for col in REQUIRED_COLUMNS
    }
    
    total_null_or_empty = sum(null_counts.values()) + sum(empty_counts.values())

    # Check duplicates
    duplicate_ids = int(df["id"].duplicated().sum())

    # Topic distribution
    topic_counts = df["topic"].astype(str).str.lower().value_counts().to_dict()
    unknown_topics = [t for t in topic_counts.keys() if t not in EXPECTED_TOPICS]

    warnings: List[str] = []
    if total_null_or_empty > 0:
        warnings.append(
            f"Found {total_null_or_empty} empty or null entries across fields. "
            f"Nulls: {null_counts}, Blanks: {empty_counts}"
        )
    if duplicate_ids > 0:
        warnings.append(f"Found {duplicate_ids} duplicate question IDs.")
    if unknown_topics:
        warnings.append(f"Found non-standard topics: {unknown_topics}.")

    return {
        "row_count": row_count,
        "is_valid": True,
        "missing_columns": missing_cols,
        "null_counts": null_counts,
        "empty_counts": empty_counts,
        "duplicate_ids": duplicate_ids,
        "topic_counts": topic_counts,
        "warnings": warnings,
    }


def get_default_dataset_path() -> str:
    """Resolve default questions.csv path whether running from root or prompt_evaluator/."""
    candidates = [
        "questions.csv",
        "prompt_evaluator/questions.csv",
        os.path.join(os.path.dirname(__file__), "questions.csv"),
        os.path.join(os.path.dirname(__file__), "..", "questions.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)
    return "questions.csv"
