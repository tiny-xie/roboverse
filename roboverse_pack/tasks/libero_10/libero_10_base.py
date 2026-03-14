"""Base class for LIBERO-10 style tasks."""

from __future__ import annotations

import torch

from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.base import BaseTaskEnv
from metasim.utils.demo_util import get_traj
from metasim.utils.hf_util import check_and_download_single
from metasim.utils.state import TensorState


LIBERO_TABLETOP_SCENE = "libero_kitchen_tabletop"
LIBERO_FRONT_CAMERA = PinholeCameraCfg(
	name="camera0",
	data_types=["rgb", "depth"],
	width=256,
	height=256,
	pos=(1.0, 0.0, 1.22),
	look_at=(-0.02, 0.0, 0.64),
	focal_length=40.0,
)


class Libero10BaseTask(BaseTaskEnv):
	"""Base class for LIBERO-10 tasks.

	This mirrors the `Libero90BaseTask` workflow so custom LIBERO-10 tasks
	can reuse the same replay/data-loading path.
	"""

	scenario = None
	max_episode_steps = 300
	task_desc = None
	checker = None
	traj_filepath = None
	decimation = 30

	def __init__(self, scenario: ScenarioCfg, device: str | torch.device | None = None) -> None:
		if scenario.scene is None:
			scenario.update(scene=LIBERO_TABLETOP_SCENE)
		if not scenario.cameras:
			scenario.update(cameras=[LIBERO_FRONT_CAMERA])
		check_and_download_single(self.traj_filepath)
		super().__init__(scenario, device)

	def _terminated(self, states: TensorState) -> torch.Tensor:
		"""Success when checker condition is met."""
		return self.checker.check(self.handler, states)

	def reset(self, states=None, env_ids=None):
		"""Reset task and checker state."""
		states = super().reset(states, env_ids)
		self.checker.reset(self.handler, env_ids=env_ids)
		return states

	def _get_initial_states(self) -> list[dict] | None:
		"""Load initial states from trajectory file."""
		initial_states, _, _ = get_traj(self.traj_filepath, self.scenario.robots[0], self.handler)
		if len(initial_states) < self.num_envs:
			k = self.num_envs // len(initial_states)
			initial_states = initial_states * k + initial_states[: self.num_envs % len(initial_states)]
		self._initial_states = initial_states[: self.num_envs]
		return self._initial_states
