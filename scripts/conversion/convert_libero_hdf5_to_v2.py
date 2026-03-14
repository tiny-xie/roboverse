from __future__ import annotations

import argparse
import os
import pickle
from typing import Any

import h5py
import numpy as np

FRANKA_DOF_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
    "panda_finger_joint1",
    "panda_finger_joint2",
]

# Scene-3 kitchen anchor poses extracted from official RoboVerse LIBERO-90 trajectories.
SCENE3_STOVE_INIT = {
    "pos": [-0.1901849650391147, 0.20206509990487764, 0.0050000000000000044],
    "rot": [1.0, 0.0, 0.0, 0.0],
    "dof_pos": {"button": 0.0},
}
SCENE3_MOKA_INIT = {
    "pos": [0.042056123656959016, -0.012891268372564543, 0.06610170238630331],
    "rot": [-2.0801461861340933e-10, 2.456996438972366e-07, -8.51425562983301e-06, 0.9999999999637237],
}
SCENE3_FRYPAN_INIT = {
    "pos": [-0.05528322417801624, -0.25173016445648455, -0.0010043308165892384],
    "rot": [0.7071067559272116, -6.078223204305228e-05, 6.142891735541426e-05, 0.7071068011652152],
}

LIVING_ROOM_FRANKA_INIT = {
    "pos": [-0.51, 0.0, 0.42],
    "rot": [1.0, 0.0, 0.0, 0.0],
}

LIVING_ROOM_OBJECT_Z_LIFT = {
    "alphabet_soup": 0.0,
    "cream_cheese": 0.0,
    "tomato_sauce": 0.0,
    "ketchup": 0.0,
    "basket": 0.0,
}


def _demo_sort_key(name: str):
    if "_" in name:
        tail = name.rsplit("_", 1)[-1]
        if tail.isdigit():
            return int(tail)
    return name


def _lift_pos(pos: np.ndarray, obj_name: str) -> list[float]:
    lifted = pos.astype(np.float64).copy()
    lifted[2] += LIVING_ROOM_OBJECT_Z_LIFT.get(obj_name, 0.0)
    return lifted.tolist()


def convert_hdf5_to_v2(input_hdf5: str, output_pkl: str, robot_name: str = "franka") -> tuple[int, int]:
    """Convert LIBERO HDF5 demos to RoboVerse trajectory v2 format.

    Notes:
    - The produced v2 file keeps actions from each demo.
    - `init_state` is populated for robot and task objects when available.
    - `states` is set to None.
    """
    input_name = os.path.basename(input_hdf5).lower()
    is_scene3_moka_task = "kitchen_scene3_turn_on_the_stove_and_put_the_moka_pot_on_it" in input_name
    is_living_room_scene1_both_task = (
        "living_room_scene1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket" in input_name
    )

    with h5py.File(input_hdf5, "r") as f:
        if "data" not in f:
            raise ValueError("Expected top-level group 'data' in HDF5")

        demo_names = sorted(list(f["data"].keys()), key=_demo_sort_key)
        if not demo_names:
            raise ValueError("No demos found under /data")

        demos: list[dict[str, Any]] = []
        total_steps = 0

        for demo_name in demo_names:
            demo_group = f["data"][demo_name]
            if "actions" not in demo_group:
                raise ValueError(f"Demo {demo_name} does not contain 'actions'")

            n_steps = int(np.asarray(demo_group["actions"]).shape[0])

            if "obs" in demo_group and "joint_states" in demo_group["obs"] and "gripper_states" in demo_group["obs"]:
                arm = np.asarray(demo_group["obs"]["joint_states"], dtype=np.float32)
                grip = np.asarray(demo_group["obs"]["gripper_states"], dtype=np.float32)
                # LIBERO stores the two fingers with opposite signs; RoboVerse Franka uses two positive finger joints.
                grip = np.abs(grip)
                qpos = np.concatenate([arm, grip], axis=1)
            elif "robot_states" in demo_group:
                qpos = np.asarray(demo_group["robot_states"], dtype=np.float32)
            else:
                raise ValueError(
                    f"Demo {demo_name} does not contain robot_states or (obs/joint_states + obs/gripper_states)"
                )

            if qpos.shape[0] != n_steps:
                raise ValueError(
                    f"Demo {demo_name} step mismatch: actions={n_steps}, qpos={qpos.shape[0]}"
                )
            if qpos.shape[1] != len(FRANKA_DOF_NAMES):
                raise ValueError(
                    f"Demo {demo_name} qpos dim={qpos.shape[1]} does not match Franka DOFs={len(FRANKA_DOF_NAMES)}"
                )

            actions: list[dict[str, Any]] = []
            for i in range(n_steps):
                dof_pos_target = {jn: float(qpos[i, j]) for j, jn in enumerate(FRANKA_DOF_NAMES)}
                actions.append({"dof_pos_target": dof_pos_target})

            total_steps += n_steps

            init_state: dict[str, Any] = {
                # Keep robot base aligned with existing LIBERO kitchen tasks in RoboVerse.
                "franka": {
                    "pos": [-0.66, 0.0, 0.012],
                    "rot": [1.0, 0.0, 0.0, 0.0],
                    "dof_pos": {jn: float(qpos[0, j]) for j, jn in enumerate(FRANKA_DOF_NAMES)},
                }
            }

            if is_scene3_moka_task:
                # Use authoritative scene anchor poses to avoid floating/teleporting objects.
                init_state["flat_stove"] = SCENE3_STOVE_INIT
                init_state["moka_pot"] = SCENE3_MOKA_INIT
                init_state["chefmate_8_frypan"] = SCENE3_FRYPAN_INIT

            if is_living_room_scene1_both_task and "states" in demo_group:
                states = np.asarray(demo_group["states"], dtype=np.float32)
                # Match the official living-room robot pedestal pose instead of the kitchen default.
                init_state["franka"]["pos"] = LIVING_ROOM_FRANKA_INIT["pos"]
                init_state["franka"]["rot"] = LIVING_ROOM_FRANKA_INIT["rot"]
                # Frame-0 in LIBERO demos can include transient penetration/settling,
                # especially for distractors (e.g., ketchup). Use a settled early frame.
                stable_idx = min(10, states.shape[0] - 1)
                s0 = states[stable_idx]
                # Layout inferred from LIBERO model_file + init_state qpos ordering for this task:
                # [unknown_1d, franka_9d, alphabet(7), cream(7), tomato(7), ketchup(7), basket(7), ...]
                init_state["alphabet_soup"] = {
                    "pos": _lift_pos(s0[10:13], "alphabet_soup"),
                    "rot": s0[13:17].astype(np.float64).tolist(),
                }
                init_state["cream_cheese"] = {
                    "pos": _lift_pos(s0[17:20], "cream_cheese"),
                    "rot": s0[20:24].astype(np.float64).tolist(),
                }
                init_state["tomato_sauce"] = {
                    "pos": _lift_pos(s0[24:27], "tomato_sauce"),
                    "rot": s0[27:31].astype(np.float64).tolist(),
                }
                init_state["ketchup"] = {
                    "pos": _lift_pos(s0[31:34], "ketchup"),
                    "rot": s0[34:38].astype(np.float64).tolist(),
                }
                init_state["basket"] = {
                    "pos": _lift_pos(s0[38:41], "basket"),
                    "rot": s0[41:45].astype(np.float64).tolist(),
                }

            demos.append({
                "init_state": init_state,
                "actions": actions,
                "states": None,
            })

    os.makedirs(os.path.dirname(output_pkl), exist_ok=True)
    payload = {robot_name: demos}
    with open(output_pkl, "wb") as f:
        pickle.dump(payload, f)

    return len(demos), total_steps


def main():
    parser = argparse.ArgumentParser(description="Convert LIBERO HDF5 demo to RoboVerse v2 trajectory")
    parser.add_argument("--input", required=True, help="Path to LIBERO demo HDF5")
    parser.add_argument("--output", required=True, help="Path to output *_v2.pkl")
    parser.add_argument("--robot", default="franka", help="Robot key used in v2 file")
    args = parser.parse_args()

    n_demo, n_steps = convert_hdf5_to_v2(args.input, args.output, robot_name=args.robot)
    print(f"Converted {n_demo} demos, total {n_steps} action steps")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
