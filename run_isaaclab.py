# run_isaaclab.py
import argparse
import os
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isaac Lab Manipulation Script")
parser.add_argument("--idx", default="46466", type=str)
parser.add_argument("--obj_scale", default=0.5, type=float)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--x_offset", default=0.75, type=float)
parser.add_argument("--y_offset", default=0.0, type=float)
parser.add_argument("--z_offset", default=0.0, type=float)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from pxr import UsdPhysics, PhysxSchema
import omni.physx
import omni.usd

PWD = os.path.dirname(os.path.abspath(__file__))

# Load grasp and configuration data
all_grasps = json.load(open(f"{PWD}/data/gapartnet/grasps.json", "r"))
all_configs = json.load(open(f"{PWD}/data/gapartnet/selected.json", "r"))
all_grasps = {k: v for k, v in all_grasps.items() if v != {}}

# Task IDs to process
TASK_IDS = [args_cli.idx]

@configclass
class ManipulationSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene assets."""
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.0)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light", 
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def create_object_cfg(obj_id: str, env_idx: int = 0):
    """Create object configuration for a specific object ID."""
    obj_config = all_configs[obj_id]
    
    # Calculate object position relative to robot
    obj_x = args_cli.x_offset + all_grasps[obj_id].get("x_offset", 0.0)
    obj_y = args_cli.y_offset + all_grasps[obj_id].get("y_offset", 0.0)
    obj_z = obj_config['height'] * args_cli.obj_scale + args_cli.z_offset

    object_cfg = ArticulationCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Object_{obj_id}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{PWD}/data/gapartnet/{obj_id}/mobility_relabel_gapartnet.usd",
            scale=(args_cli.obj_scale, args_cli.obj_scale, args_cli.obj_scale),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(obj_x, obj_y, obj_z),
            joint_pos={
                "joint_0": 0.0,
                "joint_1": 0.0, 
                "joint_2": 0.0, 
                "joint_3": 0.0
            }
        ),
        actuators={
            "joints": ImplicitActuatorCfg(
                joint_names_expr=["joint_[0-9]"],
                stiffness=0.0,
                damping=5.0,
            ),
        }
    )

    return object_cfg, obj_config

def compute_grasp_pose(obj_id: str, env_idx: int):
    """Compute the grasp pose for a given object."""
    grasp_data = all_grasps[obj_id]
    obj_config = all_configs[obj_id]
    
    # Get grasp position and rotation from data
    p = np.asarray(grasp_data['p']) * args_cli.obj_scale
    p[0] += args_cli.x_offset + grasp_data.get("x_offset", 0.0)
    p[1] += args_cli.y_offset + grasp_data.get("y_offset", 0.0)
    # p[0] += 0.046  # Offset for gripper
    p[2] += obj_config['height'] * args_cli.obj_scale + args_cli.z_offset

    r_euler = grasp_data['R']

    # Offset for gripper in local frame
    r_matrix = R.from_euler("ZYX", r_euler, degrees=False).as_matrix()
    gripper_offset_local = np.array([0.0, 0.0, -0.046])  # Always backward
    p += r_matrix @ gripper_offset_local

    # Convert rotation from euler to quaternion (w, x, y, z format)
    r = R.from_euler("ZYX", r_euler, degrees=False).as_quat()  # Returns (x, y, z, w)
    r_wxyz = np.array([r[3], r[0], r[1], r[2]])  # Convert to (w, x, y, z)
    
    return p, r_wxyz

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, task_configs: list):
    robot = scene["robot"]
    num_envs = scene.num_envs
    
    print(f"[INFO]: Running {num_envs} environments with tasks: {TASK_IDS}")

    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=num_envs, device=sim.device)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))
    waypoint_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/waypoint"))

    # Robot base frame marker
    base_marker_cfg = FRAME_MARKER_CFG.copy()
    base_marker_cfg.markers["frame"].scale = (0.15, 0.15, 0.15)
    base_marker = VisualizationMarkers(base_marker_cfg.replace(prim_path="/Visuals/robot_base"))

    # Compute ACTUAL grasp poses from object configuration
    grasp_poses = []
    for i, obj_id in enumerate(TASK_IDS[:num_envs]):
        p, r = compute_grasp_pose(obj_id, i)
        grasp_poses.append(np.concatenate([p, r]))
    
    grasp_poses_w = torch.tensor(np.array(grasp_poses), device=sim.device, dtype=torch.float32)
    print(f"[INFO]: Target grasp poses (world): {grasp_poses_w}")
    
    # Create waypoint (offset back in world X)
    waypoint_offset = 0.15  # 15cm up and back
    waypoint_poses_w = grasp_poses_w.clone()
    waypoint_poses_w[:, 0] -= waypoint_offset
    waypoint_poses_w[:, 2] += waypoint_offset
    print(f"[INFO]: Waypoint poses (world):     {waypoint_poses_w[0, 0:3]}")

    # Robot entity config
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint[1-7]"], body_names=["panda_hand"])
    robot_entity_cfg.resolve(scene)
    ee_body_idx = robot_entity_cfg.body_ids[0]
    ee_jacobi_idx = ee_body_idx if not robot.is_fixed_base else ee_body_idx - 1
    
    # Gripper commands
    gripper_open = torch.tensor([[0.04, 0.04]], device=sim.device).repeat(num_envs, 1)
    gripper_closed = torch.tensor([[0.0, 0.0]], device=sim.device).repeat(num_envs, 1)

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    
    # Track phase (0=waypoint, 1=grasp, 2=hold)
    phase = 0
    
    print("[INFO]: Starting IK control loop...")
    
    # Simulation loop
    while simulation_app.is_running():
        # Reset periodically for testing
        if count % 450 == 0:  # Full cycle: 150 (waypoint) + 150 (grasp) + 150 (hold)
            count = 0
            phase = 0
            print(f"\n[INFO]: Resetting to initial state...")
            
            # Reset robot
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            
            # Start with waypoint target
            print(f"[PHASE 0]: Moving to waypoint...")
            target_poses = waypoint_poses_w
            
            # Reset controller
            diff_ik_controller.reset()
            
            root_pose_w = robot.data.root_pose_w
            target_pos_b, target_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7],
                target_poses[:, 0:3], target_poses[:, 3:7]
            )
            
            diff_ik_controller.set_command(torch.cat([target_pos_b, target_quat_b], dim=-1))
            
        # Phase transitions within cycle
        elif count == 150 and phase == 0:
            # Switch from waypoint to grasp
            phase = 1
            target_poses = grasp_poses_w
            
            diff_ik_controller.reset()
            root_pose_w = robot.data.root_pose_w
            target_pos_b, target_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7],
                target_poses[:, 0:3], target_poses[:, 3:7]
            )
            diff_ik_controller.set_command(torch.cat([target_pos_b, target_quat_b], dim=-1))
            
        elif count == 300 and phase == 1:
            # Switch to hold phase
            phase = 2
            
            # Close gripper over next 30 steps
            for close_step in range(30):
                alpha = (close_step + 1) / 30.0
                gripper_pos = gripper_open * (1 - alpha) + gripper_closed * alpha
                
                robot.set_joint_position_target(
                    robot.data.joint_pos[:, robot_entity_cfg.joint_ids], 
                    joint_ids=robot_entity_cfg.joint_ids
                )
                robot.set_joint_position_target(gripper_pos, joint_ids=slice(7, 9))
                scene.write_data_to_sim()
                sim.step()
                scene.update(sim_dt)
                count += 1
        
        # Compute IK if in waypoint or grasp phase
        if phase < 2:
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
            ee_pose_w = robot.data.body_pose_w[:, ee_body_idx]
            root_pose_w = robot.data.root_pose_w
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7],
                ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
            
            # Apply actions
            robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
            robot.set_joint_position_target(gripper_open, joint_ids=slice(7, 9))
        else:
            # Hold phase
            robot.set_joint_position_target(
                robot.data.joint_pos[:, robot_entity_cfg.joint_ids], 
                joint_ids=robot_entity_cfg.joint_ids
            )
            robot.set_joint_position_target(gripper_closed, joint_ids=slice(7, 9))
        
        # Step simulation
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)

        # TODO: remove status updates
        if count % 50 == 0:
            ee_pose_w = robot.data.body_pose_w[:, ee_body_idx]
            if phase == 0:
                pos_error = torch.norm(ee_pose_w[:, 0:3] - waypoint_poses_w[:, 0:3], dim=-1)
                print(f"[Step {count:4d}] Phase 0 (Waypoint) | Error: {pos_error[0].item():.4f}m")
            elif phase == 1:
                pos_error = torch.norm(ee_pose_w[:, 0:3] - grasp_poses_w[:, 0:3], dim=-1)
                print(f"[Step {count:4d}] Phase 1 (Grasp)    | Error: {pos_error[0].item():.4f}m")
            else:
                pos_error = torch.norm(ee_pose_w[:, 0:3] - grasp_poses_w[:, 0:3], dim=-1)
                print(f"[Step {count:4d}] Phase 2 (Hold)     | Error: {pos_error[0].item():.4f}m")

        # Visualize
        ee_pose_w = robot.data.body_state_w[:, ee_body_idx, 0:7]
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        waypoint_marker.visualize(waypoint_poses_w[:, 0:3], waypoint_poses_w[:, 3:7])
        goal_marker.visualize(grasp_poses_w[:, 0:3], grasp_poses_w[:, 3:7])
        base_marker.visualize(root_pose_w[:, 0:3], root_pose_w[:, 3:7])

def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[-0.5, 2.0, 2.0], target=[0.5, 0.0, 0.0])

    scene_cfg = ManipulationSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
 
    task_configs = []
    for i in range(min(args_cli.num_envs, len(TASK_IDS))):
        obj_id = TASK_IDS[i]
        obj_cfg, obj_config = create_object_cfg(obj_id, i)
        setattr(scene_cfg, f"object_{i}", obj_cfg)
        task_configs.append({
            'obj_id': obj_id,
            'obj_config': obj_config,
            'env_idx': i
        })
        print(f"[INFO]: Added object {obj_id}")

    scene = InteractiveScene(scene_cfg)
    sim.reset()

    print("[INFO]: Setup complete - Robot + Object loaded")
    print(f"[INFO]: Object should be at x={args_cli.x_offset}, z={all_configs[TASK_IDS[0]]['height'] * args_cli.obj_scale}")
    
    # Run simulator
    run_simulator(sim, scene, task_configs)


if __name__ == "__main__":
    main()
    simulation_app.close()