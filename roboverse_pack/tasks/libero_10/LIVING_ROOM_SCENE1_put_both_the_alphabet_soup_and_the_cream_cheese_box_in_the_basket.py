"""Configuration for the LIBERO-10 task:
LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket.
"""

from __future__ import annotations

import torch

from metasim.constants import PhysicStateType
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task
from metasim.types import TensorState

from .libero_10_base import Libero10BaseTask


LIBERO_OBJ_FILE_TYPE_FOR_GENESIS = {
    "isaaclab": "usd",
    "isaacsim": "usd",
    "pybullet": "urdf",
    "sapien2": "urdf",
    "sapien3": "urdf",
    "genesis": "mjcf",
    "isaacgym": "urdf",
    "mujoco": "mjcf",
    "mjx": "mjx_mjcf",
    "newton": "urdf",
}


@register_task(
    "libero_10.living_room_scene1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
    "living_room_scene1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
)
class Libero10LivingRoomScene1PutBothAlphabetSoupAndCreamCheeseInTheBasketTask(Libero10BaseTask):
    """Put both alphabet_soup and cream_cheese into basket/contain_region."""

    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="alphabet_soup",
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/alphabet_soup/usd/alphabet_soup.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/alphabet_soup/urdf/alphabet_soup.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/alphabet_soup/mjcf/alphabet_soup.xml",
                file_type=LIBERO_OBJ_FILE_TYPE_FOR_GENESIS,
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="cream_cheese",
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/cream_cheese/usd/cream_cheese.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/cream_cheese/urdf/cream_cheese.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/cream_cheese/mjcf/cream_cheese.xml",
                file_type=LIBERO_OBJ_FILE_TYPE_FOR_GENESIS,
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="tomato_sauce",
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/tomato_sauce/usd/tomato_sauce.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/tomato_sauce/urdf/tomato_sauce.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/tomato_sauce/mjcf/tomato_sauce.xml",
                file_type=LIBERO_OBJ_FILE_TYPE_FOR_GENESIS,
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="ketchup",
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/ketchup/usd/ketchup.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/ketchup/urdf/ketchup.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/ketchup/mjcf/ketchup.xml",
                file_type=LIBERO_OBJ_FILE_TYPE_FOR_GENESIS,
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="basket",
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/basket/usd/basket.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/basket/urdf/basket.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/basket/mjcf/basket.xml",
                file_type=LIBERO_OBJ_FILE_TYPE_FOR_GENESIS,
                physics=PhysicStateType.RIGIDBODY,
            ),
        ],
        robots=["franka"],
        scene="libero_living_room_tabletop",
        cameras=[
            PinholeCameraCfg(
                name="camera0",
                data_types=["rgb", "depth"],
                width=256,
                height=256,
               
                pos=(1.0, 0.0, 1.22),
                look_at=(-0.02, 0.0, 0.64),
                focal_length=40.0,
            )
        ],
    )

    max_episode_steps = 250
    task_desc = "Put both the alphabet_soup and the cream_cheese box in the basket (living_room_scene1)"

    workspace_name = ("living_room_table",)
    workspace_offset = ((0, 0, 0),)
    workspace_size = ((1.0, 1.2, 0.1),)

    # TODO: update to your converted libre_10 trajectory path.
    traj_filepath = (
        "roboverse_data/trajs/libero10/"
        "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_demo_v2.pkl"
    )

    @staticmethod
    def _inside_region(obj_pos: torch.Tensor, region_R: torch.Tensor, region_t: torch.Tensor, half_size: torch.Tensor):
        obj_local = torch.matmul(region_R.transpose(1, 2), (obj_pos - region_t).unsqueeze(-1)).squeeze(-1)
        return (obj_local.abs() <= (half_size + 1e-6)).all(dim=-1)

    def _terminated(self, states: TensorState) -> torch.Tensor:
        """Task succeeds only when both target objects are inside basket/contain_region."""
        soup_pos = states.objects["alphabet_soup"].root_state[:, :3]
        cheese_pos = states.objects["cream_cheese"].root_state[:, :3]
        n_env = soup_pos.shape[0]

        # MuJoCo exposes named site transforms via handler.physics. Other simulators
        # (e.g. Genesis) may not have this interface; in that case keep rollout-based
        # replay running and let episode end by timeout.
        if not hasattr(self.handler, "physics"):
            return torch.zeros(n_env, dtype=torch.bool, device=soup_pos.device)

        region_mat = self.handler.physics.named.data.site_xmat["basket/contain_region"]
        region_pos = self.handler.physics.named.data.site_xpos["basket/contain_region"]

        region_R = torch.from_numpy(region_mat).float().reshape(3, 3).unsqueeze(0).expand(n_env, 3, 3).to(soup_pos.device)
        region_t = torch.from_numpy(region_pos).float().unsqueeze(0).expand(n_env, 3).to(soup_pos.device)
        half_size = torch.tensor([0.06108, 0.06108, 0.06949], device=soup_pos.device)

        soup_inside = self._inside_region(soup_pos, region_R, region_t, half_size)
        cheese_inside = self._inside_region(cheese_pos, region_R, region_t, half_size)
        return soup_inside & cheese_inside

    def reset(self, states=None, env_ids=None):
        """Skip checker reset."""
        states = super(Libero10BaseTask, self).reset(states, env_ids)
        return states
