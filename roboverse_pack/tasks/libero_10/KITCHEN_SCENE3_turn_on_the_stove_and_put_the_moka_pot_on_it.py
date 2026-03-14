"""Configuration for the LIBERO-10 task:
KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.
"""

from __future__ import annotations

import torch

from metasim.constants import PhysicStateType
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.objects import ArticulationObjCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task
from metasim.types import TensorState

from .libero_10_base import Libero10BaseTask


@register_task(
    "libero_10.kitchen_scene3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "kitchen_scene3_turn_on_the_stove_and_put_the_moka_pot_on_it",
)
class Libero10KitchenScene3TurnOnStoveAndPutMokaPotTask(Libero10BaseTask):
    """Turn on the stove and place moka_pot on stove burner."""

    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="moka_pot",
                mjcf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/moka_pot/mjcf/moka_pot.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="chefmate_8_frypan",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/chefmate_8_frypan/mjcf/chefmate_8_frypan.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            ArticulationObjCfg(
                name="flat_stove",
                fix_base_link=True,
                mjcf_path="roboverse_data/assets/libero/COMMON/articulated_objects/flat_stove/mjcf/flat_stove.xml",
            ),
        ],
        robots=["franka"],
        scene="libero_kitchen_tabletop",
        cameras=[
            PinholeCameraCfg(
                name="camera0",
                data_types=["rgb", "depth"],
                width=256,
                height=256,
                # Match native LIBERO kitchen agentview more closely.
                pos=(0.6586, 0.0, 1.6103),
                look_at=(0.0, 0.0, 0.90),
                focal_length=40.0,
            )
        ],
    )

    max_episode_steps = 500
    task_desc = "Turn on the stove and put the moka pot on it (scene3)"

    workspace_name = ("kitchen_table",)
    workspace_offset = ((0, 0, 0),)
    workspace_size = ((1.0, 1.2, 0.05),)

    # TODO: update to your converted libero_10 trajectory path.
    traj_filepath = (
        "roboverse_data/trajs/libero10/"
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo_v2.pkl"
    )

    def _terminated(self, states: TensorState) -> torch.Tensor:
        """Task success checker: stove is on AND moka_pot is on stove."""
        stove_joint_state = states.objects["flat_stove"].joint_pos[:, 0]
        stove_on = stove_joint_state > 0.5

        moka_pos = states.objects["moka_pot"].root_state[:, :3]
        stove_center_single = self.handler.physics.named.data.site_xpos["flat_stove/burner"]
        n_env = moka_pos.shape[0]
        stove_center_pos = torch.tensor(stove_center_single, device=moka_pos.device).reshape(1, 3).repeat(n_env, 1)

        xy_distance = torch.norm(moka_pos[:, :2] - stove_center_pos[:, :2], dim=-1)
        height_diff = moka_pos[:, 2] - stove_center_pos[:, 2]
        moka_on_stove = (xy_distance < 0.06) & (height_diff > 0) & (height_diff < 0.05)

        return stove_on & moka_on_stove

    def reset(self, states=None, env_ids=None):
        """Skip checker reset."""
        states = super(Libero10BaseTask, self).reset(states, env_ids)
        return states
