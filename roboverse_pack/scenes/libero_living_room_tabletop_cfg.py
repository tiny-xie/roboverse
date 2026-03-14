from __future__ import annotations

from pathlib import Path

from metasim.utils.configclass import configclass

from .base_scene_cfg import SceneCfg


@configclass
class LiberoLivingRoomTabletopCfg(SceneCfg):
    """LIBERO living room tabletop scene configuration."""

    name: str = "libero_living_room_tabletop"
    # Prefer native LIBERO living-room scene for faithful geometry and lighting.
    mjcf_path: str = (
        "LIBERO/libero/libero/assets/scenes/libero_living_room_tabletop_base_style.xml"
        if Path("LIBERO/libero/libero/assets/scenes/libero_living_room_tabletop_base_style.xml").exists()
        else "roboverse_data/assets/libero/scenes/libero_living_room_tabletop_base_style.xml"
    )
    positions: list[tuple[float, float, float]] = [
        (0.0, 0.0, 0.0),
    ]
    default_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
