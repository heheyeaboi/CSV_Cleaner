# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Csv Clean Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import CsvCleanAction, CsvCleanObservation


class CsvCleanEnv(
    EnvClient[CsvCleanAction, CsvCleanObservation, State]
):
    """
    Client for the Csv Clean Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with CsvCleanEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.task_name)
        ...
        ...     result = client.step(CsvCleanAction(operation="drop_nulls", column="price"))
        ...     print(result.observation.last_operation_result)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = CsvCleanEnv.from_docker_image("csv_clean_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(CsvCleanAction(operation="drop_nulls", column="price"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: CsvCleanAction) -> Dict:
        """
        Convert CsvCleanAction to JSON payload for step message.

        Args:
            action: CsvCleanAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "operation": action.operation,
            "column": action.column,
            "value": action.value,
        }

    def _parse_result(self, payload: Dict) -> StepResult[CsvCleanObservation]:
        """
        Parse server response into StepResult[CsvCleanObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with CsvCleanObservation
        """
        obs_data = payload.get("observation", {})
        observation = CsvCleanObservation(
            current_csv=obs_data.get("current_csv", ""),
            num_rows=obs_data.get("num_rows", 0),
            num_cols=obs_data.get("num_cols", 0),
            null_counts=obs_data.get("null_counts", {}),
            dtypes=obs_data.get("dtypes", {}),
            last_operation_result=obs_data.get("last_operation_result", ""),
            errors=obs_data.get("errors", []),
            task_name=obs_data.get("task_name", ""),
            task_description=obs_data.get("task_description", ""),
            steps_taken=obs_data.get("steps_taken", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
