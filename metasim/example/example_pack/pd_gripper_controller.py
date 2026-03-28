"""
PD-based Gripper Controller for RoboVerse
Implements closed-loop PD feedback similar to ManiSkill's PDJointPosController
"""

import numpy as np
from typing import Dict, List


class PDGripperController:
    """
    Closed-loop PD controller for gripper joints.
    Maintains continuous gripper action and applies PD forces even when obstructed.
    """

    def __init__(
        self,
        joint_names: List[str],
        kp: float = 2000.0,  # Proportional gain (stiffness)
        kd: float = 400.0,   # Derivative gain (damping)
        max_force: float = 100.0,
    ):
        """
        Args:
            joint_names: List of gripper joint names (e.g., ["panda_finger_joint1", "panda_finger_joint2"])
            kp: Proportional gain (acts like stiffness)
            kd: Derivative gain (acts like damping)
            max_force: Maximum force to apply (safety limit)
        """
        self.joint_names = joint_names
        self.kp = kp
        self.kd = kd
        self.max_force = max_force
        
        self.target_position = {name: 0.04 for name in joint_names}  # Init to open
        self.prev_error = {name: 0.0 for name in joint_names}
        self.target_vel = {name: 0.0 for name in joint_names}

    def set_target(self, gripper_command: float):
        """
        Set gripper target.
        
        Args:
            gripper_command: float in [-1, 1]
                > 0.5: open (move to 0.04)
                <= 0.5: close (move to 0.0)
        """
        target_q = 0.04 if gripper_command > 0.5 else 0.0
        for name in self.joint_names:
            self.target_position[name] = target_q

    def compute_torques(self, current_positions: Dict[str, float], 
                       current_velocities: Dict[str, float],
                       dt: float = 0.002) -> Dict[str, float]:
        """
        Compute PD-based torques for each gripper joint.
        
        Args:
            current_positions: Current joint positions {joint_name: position}
            current_velocities: Current joint velocities {joint_name: velocity}
            dt: Time step
            
        Returns:
            torques: {joint_name: torque_command}
        """
        torques = {}
        
        for name in self.joint_names:
            q_current = current_positions[name]
            q_target = self.target_position[name]
            v_current = current_velocities.get(name, 0.0)
            
            # P term: position error
            position_error = q_target - q_current
            
            # D term: velocity damping
            velocity_error = self.target_vel[name] - v_current
            
            # PD control law: τ = kp * e_pos + kd * e_vel
            torque = self.kp * position_error + self.kd * velocity_error
            
            # Clamp to safety limits
            torque = np.clip(torque, -self.max_force, self.max_force)
            
            torques[name] = torque
            self.prev_error[name] = position_error
            
        return torques

    def reset(self):
        """Reset controller state."""
        for name in self.joint_names:
            self.target_position[name] = 0.04  # Open
            self.prev_error[name] = 0.0
            self.target_vel[name] = 0.0


class GripperActionProcessor:
    """
    Integrates binary gripper commands (0/1) into smooth PD-controlled motion.
    Similar to robosuite's GripperController.format_action().
    """
    
    def __init__(self, speed: float = 0.2, dt: float = 0.002):
        """
        Args:
            speed: Gripper speed (normalized per timestep)
            dt: Simulation timestep
        """
        self.speed = speed
        self.dt = dt
        self.current_action = 0.0  # Current accumulated action [-1, 1]
    
    def update_action(self, gripper_command: float) -> float:
        """
        Accumulate gripper action like robosuite does.
        
        Args:
            gripper_command: -1 or 1 (open/close), or 0 for no change
            
        Returns:
            Updated action value in [-1, 1]
        """
        if abs(gripper_command) > 0.5:
            # Apply speed accumulation
            self.current_action = np.clip(
                self.current_action + np.sign(gripper_command) * self.speed,
                -1.0, 1.0
            )
        return self.current_action
    
    def action_to_target(self, action: float) -> float:
        """
        Convert accumulated action to gripper target position.
        
        action in [-1, 1] -> target_position in [0.0, 0.04]
        """
        # Linear mapping: -1 -> 0.0 (close), 1 -> 0.04 (open)
        return 0.02 + action * 0.02
    
    def reset(self):
        """Reset action accumulator."""
        self.current_action = 0.0
