"""
client.py
---------
Client for the SQL Analytics Environment.

SQLAnalyticsEnv wraps EnvClient with the three typed models specific to
this environment (SQLAction, SQLObservation, SQLState).  All connection
and session management is inherited from EnvClient.

Example usage
-------------
# Against a local server:
    with SQLAnalyticsEnv(base_url="http://localhost:8000") as env:
        obs = env.reset(task_id="task_003")
        print(obs.task_description)

        result_obs = env.step(
            SQLAction(sql="SELECT name, SUM(total_amount) FROM ...")
        )
        print(result_obs.result, result_obs.reward)

# Against a HuggingFace Space:
    env = SQLAnalyticsEnv.from_env("your-hf-username/sql-analytics-env")
    try:
        obs = env.reset(task_id="task_001")
        ...
    finally:
        env.close()

# From a local Docker image:
    env = SQLAnalyticsEnv.from_docker_image("sql-analytics-env:latest")
    try:
        obs = env.reset(task_id="task_002")
        ...
    finally:
        env.close()
"""

from typing import Any, Dict

from openenv.core.env_client import EnvClient, StepResult

try:
    from .models import SQLAction, SQLObservation, SQLState
except ImportError:
    from models import SQLAction, SQLObservation, SQLState


class SQLAnalyticsEnv(EnvClient[SQLAction, SQLObservation, SQLState]):
    """
    Typed client for the SQL Analytics OpenEnv environment.

    Inherits the full EnvClient interface:
        reset(**kwargs) -> StepResult[SQLObservation]
        step(action: SQLAction) -> StepResult[SQLObservation]
        state() -> SQLState
        close()

    Convenience class methods (from EnvClient):
        SQLAnalyticsEnv.from_env(hf_repo_id)
        SQLAnalyticsEnv.from_docker_image(image_tag)
    """

    def _step_payload(self, action: SQLAction) -> Dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[SQLObservation]:
        obs_dict = payload.get("observation", {})
        obs = SQLObservation(**obs_dict)
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> SQLState:
        return SQLState(**payload)
