"""
Data models for the CSV Clean Environment.

Defines the action and observation Pydantic models used by the csv_clean_env
reinforcement learning environment for cleaning CSV datasets.
"""

from typing import Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class CsvCleanAction(Action):
    """Action for the CSV Clean environment.

    Supported operations:
        - drop_nulls: Drop rows containing null values.
        - fill_nulls: Fill null values with a specified value.
        - fix_type: Fix the data type of a specified column.
        - rename_column: Rename a column (use 'column' for old name, 'value' for new name).
        - drop_column: Drop a specified column from the dataset.
        - deduplicate: Remove duplicate rows.
        - strip_whitespace: Strip leading and trailing whitespace from string columns.
        - standardize_case: Standardize the case of string values in a column.
        - done: Signal that cleaning is complete.
    """

    operation: str = Field(
        ...,
        description="Cleaning operation to perform",
    )
    column: Optional[str] = Field(
        default=None,
        description="Target column name for the operation",
    )
    value: Optional[str] = Field(
        default=None,
        description="Value to use for the operation (e.g., fill value or new column name)",
    )


class CsvCleanObservation(Observation):
    """Observation from the CSV Clean environment."""

    current_csv: str = Field(
        default="",
        description="Current state of the CSV data as a string",
    )
    num_rows: int = Field(
        default=0,
        description="Number of rows in the current dataset",
    )
    num_cols: int = Field(
        default=0,
        description="Number of columns in the current dataset",
    )
    null_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Count of null values per column",
    )
    dtypes: dict[str, str] = Field(
        default_factory=dict,
        description="Data type of each column",
    )
    last_operation_result: str = Field(
        default="",
        description="Result message from the last performed operation",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="List of error messages from failed operations",
    )
    task_name: str = Field(
        default="",
        description="Name of the current cleaning task",
    )
    task_description: str = Field(
        default="",
        description="Description of the current cleaning task",
    )
    steps_taken: int = Field(
        default=0,
        description="Number of cleaning steps taken so far",
    )
