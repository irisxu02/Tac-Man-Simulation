# run_isaaclab.py
import argparse
import os
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isaac Lab Manipulation Script")
parser.add_argument("--idx", default="46466", type=str)
parser.add_argument("--obj_scale", default=0.5, type=float)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to spawn."
)
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
from isaaclab.markers.config import FRAME_MARKER_CFG, VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    matrix_from_quat,
    subtract_frame_transforms,
    quat_apply,
    quat_from_angle_axis,
)

from pxr import UsdPhysics, PhysxSchema
import omni.physx
import omni.usd

from tacman_utils.sim_consts import CONTACT_AREAS, CONTACT_THRES
from tacman_utils.utils_3d import find_rigid_alignment

PWD = os.path.dirname(os.path.abspath(__file__))

# Load grasp and configuration data
all_grasps = json.load(open(f"{PWD}/data/gapartnet/grasps.json", "r"))
all_configs = json.load(open(f"{PWD}/data/gapartnet/selected.json", "r"))
all_grasps = {k: v for k, v in all_grasps.items() if v != {}}

# Task IDs to process
TASK_IDS = [args_cli.idx]

STATE_INIT, STATE_PROC, STATE_RECV, STATE_SUCC = -1, 0, 1, 2

delta_0 = 0.0004
alpha = 0.6


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
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class SimulationState:
    """Tracks manipulation state and marker displacement for tactile feedback control."""

    def __init__(self, num_envs=1, device="cuda"):
        self.num_envs = num_envs
        self.device = device
        self.state = STATE_INIT

        # Marker tracking
        self.locked_marker_idx = None
        self.unlocked_marker_idx = None
        self.init_marker_pos_world = None
        self.curr_marker_pos_world = None
        self.handle_marker_bound_pos = None
        self.init_marker_pos = None  # In hand frame
        self.unlocked_marker_pos = None

        # Displacement
        self.marker_dspl_r = None
        self.marker_dspl_transf = None
        self.marker_dspl_dist = 0.0

        # Control state
        self.attempt_counter = 0
        self.recv_stuck_count = 0
        self.curr_proceed_dir = None
        self.curr_proceed_base_transf = None
        self.grasp_q = None

    def lock_markers(self, contact_idx, marker_pos_world, handle_transf, hand_transf):
        """Lock markers on first contact detection"""
        self.locked_marker_idx = contact_idx
        num_markers = marker_pos_world.shape[0]
        all_idx = torch.arange(num_markers, device=self.device)
        mask = torch.ones(num_markers, dtype=torch.bool, device=self.device)
        mask[contact_idx] = False
        self.unlocked_marker_idx = all_idx[mask]

        locked_marker_pos = torch.cat(
            [
                marker_pos_world[contact_idx],
                torch.ones(len(contact_idx), 1, device=self.device),
            ],
            dim=-1,
        )

        unlocked_marker_pos = torch.cat(
            [
                marker_pos_world[self.unlocked_marker_idx],
                torch.ones(len(self.unlocked_marker_idx), 1, device=self.device),
            ],
            dim=-1,
        )

        # Save positions relative to handle and hand frames
        self.handle_marker_bound_pos = torch.matmul(
            torch.inverse(handle_transf), locked_marker_pos.T
        ).T
        self.unlocked_marker_pos = torch.matmul(
            torch.inverse(hand_transf), unlocked_marker_pos.T
        ).T.clone()
        self.init_marker_pos = torch.matmul(
            torch.inverse(hand_transf), locked_marker_pos.T
        ).T.clone()

        self.init_marker_pos_world = marker_pos_world[contact_idx].clone()

        print(f"[STATE] Locked {len(contact_idx)}/{num_markers} markers")

    def compute_displacement(self, handle_transf, hand_transf):
        """Compute marker displacement and rigid alignment."""
        if self.locked_marker_idx is None:
            return False

        # Current locked marker positions in world frame (bound to handle)
        self.curr_marker_pos_world = torch.matmul(
            handle_transf, self.handle_marker_bound_pos.T
        ).T[:, :3]

        # Current marker positions in hand frame
        curr_marker_pos_homo = torch.cat(
            [
                self.curr_marker_pos_world,
                torch.ones(len(self.curr_marker_pos_world), 1, device=self.device),
            ],
            dim=-1,
        )
        curr_marker_pos = torch.matmul(
            torch.inverse(hand_transf), curr_marker_pos_homo.T
        ).T

        # Displacement magnitude
        self.marker_dspl_dist = (
            (curr_marker_pos[:, :3] - self.init_marker_pos[:, :3])
            .norm(dim=-1)
            .mean()
            .item()
        )

        # Rigid alignment using Kabsch algorithm
        self.marker_dspl_r, marker_dspl_t = find_rigid_alignment(
            self.init_marker_pos_world, self.curr_marker_pos_world
        )

        self.marker_dspl_transf = torch.eye(4, device=self.device)
        self.marker_dspl_transf[:3, :3] = self.marker_dspl_r
        self.marker_dspl_transf[:3, 3] = marker_dspl_t

        return True


def create_object_cfg(obj_id: str, env_idx: int = 0):
    """Create object configuration for a specific object ID."""
    obj_config = all_configs[obj_id]

    # Calculate object position relative to robot
    obj_x = args_cli.x_offset + all_grasps[obj_id].get("x_offset", 0.0)
    obj_y = args_cli.y_offset + all_grasps[obj_id].get("y_offset", 0.0)
    obj_z = obj_config["height"] * args_cli.obj_scale + args_cli.z_offset

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
            joint_pos={"joint_0": 0.0, "joint_1": 0.0, "joint_2": 0.0, "joint_3": 0.0},
        ),
        actuators={
            "joints": ImplicitActuatorCfg(
                joint_names_expr=["joint_[0-9]"],
                stiffness=0.0,
                damping=5.0,
            ),
        },
    )

    return object_cfg, obj_config


def compute_grasp_pose(obj_id: str, env_idx: int):
    """Compute the grasp pose for a given object."""
    grasp_data = all_grasps[obj_id]
    obj_config = all_configs[obj_id]

    # Get grasp position and rotation from data
    p = np.asarray(grasp_data["p"]) * args_cli.obj_scale
    p[0] += args_cli.x_offset + grasp_data.get("x_offset", 0.0)
    p[1] += args_cli.y_offset + grasp_data.get("y_offset", 0.0)
    # p[0] += 0.046  # Offset for gripper
    p[2] += obj_config["height"] * args_cli.obj_scale + args_cli.z_offset

    r_euler = grasp_data["R"]

    # Offset for gripper in local frame
    r_matrix = R.from_euler("ZYX", r_euler, degrees=False).as_matrix()
    gripper_offset_local = np.array([0.0, 0.0, -0.046])  # Always backward
    p += r_matrix @ gripper_offset_local

    # Convert rotation from euler to quaternion (w, x, y, z format)
    r = R.from_euler("ZYX", r_euler, degrees=False).as_quat()  # Returns (x, y, z, w)
    r_wxyz = np.array([r[3], r[0], r[1], r[2]])  # Convert to (w, x, y, z)

    return p, r_wxyz


def get_proceeding_dir(hand_transf):
    num_envs = hand_transf.shape[0]
    dir_local = (
        torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=hand_transf.device)
        .unsqueeze(0)
        .tile([num_envs, 1])
    )
    dir_world = torch.zeros_like(dir_local)
    for i in range(num_envs):
        if hand_transf.shape[-1] == 7:
            rot = matrix_from_quat(hand_transf[i, 3:7])
        else:
            rot = hand_transf[i, :3, :3]
        dir_world[i] = torch.matmul(rot, dir_local[i])
    return dir_world


def pose_to_transform(pos, quat, device):
    """Convert position and quaternion to 4x4 transformation matrix."""
    transf = torch.eye(4, device=device)
    transf[:3, :3] = matrix_from_quat(quat)
    transf[:3, 3] = pos
    return transf


def run_simulator(
    sim: sim_utils.SimulationContext, scene: InteractiveScene, task_configs: list
):
    robot = scene["robot"]
    num_envs = scene.num_envs
    obj_id = TASK_IDS[0]
    obj_config = all_configs[obj_id]
    env_idx = 0
    print(f"[INFO]: Running {num_envs} environments with tasks: {TASK_IDS}")

    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=False, ik_method="dls"
    )
    diff_ik_controller = DifferentialIKController(
        diff_ik_cfg, num_envs=num_envs, device=sim.device
    )

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(
        frame_marker_cfg.replace(prim_path="/Visuals/ee_current")
    )
    goal_marker = VisualizationMarkers(
        frame_marker_cfg.replace(prim_path="/Visuals/ee_goal")
    )
    waypoint_marker = VisualizationMarkers(
        frame_marker_cfg.replace(prim_path="/Visuals/waypoint")
    )

    # Robot base frame marker
    base_marker_cfg = FRAME_MARKER_CFG.copy()
    base_marker_cfg.markers["frame"].scale = (0.15, 0.15, 0.15)
    base_marker = VisualizationMarkers(
        base_marker_cfg.replace(prim_path="/Visuals/robot_base")
    )

    # Contact area markers (small spheres)
    BLUE_VISUAL_MATERIAL = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0))

    contact_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/ContactMarkers",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.003,
                visual_material=BLUE_VISUAL_MATERIAL,
            )
        },
    )
    contact_markers = VisualizationMarkers(contact_marker_cfg)

    # green markers for displacement
    GREEN_VISUAL_MATERIAL = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))

    displacement_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/DisplacementMarkers",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.005,
                visual_material=GREEN_VISUAL_MATERIAL,
            )
        },
    )
    displacement_markers = VisualizationMarkers(displacement_marker_cfg)

    l_finger_kpt = torch.tensor(
        CONTACT_AREAS["panda"]["L"], dtype=torch.float32, device=sim.device
    )
    r_finger_kpt = torch.tensor(
        CONTACT_AREAS["panda"]["R"], dtype=torch.float32, device=sim.device
    )

    finger_xx = torch.linspace(0, 1, 10, device=sim.device)
    finger_yy = torch.linspace(0, 1, 10, device=sim.device)

    finger_grid = torch.stack(
        torch.meshgrid(finger_xx, finger_yy, indexing="ij"), dim=-1
    ).reshape(-1, 2)

    # Compute marker positions in finger local frame
    l_finger_markers = (
        l_finger_kpt[0].unsqueeze(0)
        + finger_grid[:, 0:1] * (l_finger_kpt[1] - l_finger_kpt[0]).unsqueeze(0)
        + finger_grid[:, 1:2] * (l_finger_kpt[3] - l_finger_kpt[0]).unsqueeze(0)
    )

    r_finger_markers = (
        r_finger_kpt[0].unsqueeze(0)
        + finger_grid[:, 0:1] * (r_finger_kpt[1] - r_finger_kpt[0]).unsqueeze(0)
        + finger_grid[:, 1:2] * (r_finger_kpt[3] - r_finger_kpt[0]).unsqueeze(0)
    )

    print(f"[INFO]: Created {len(l_finger_markers)} markers per finger")

    # Load handle mesh for contact detection
    import trimesh as tm

    handle_mesh = tm.load(
        os.path.join(PWD, obj_config["grasp_part_mesh"]), force="mesh"
    )
    scale_matrix = np.eye(4)
    scale_matrix[0, 0] = args_cli.obj_scale
    scale_matrix[1, 1] = args_cli.obj_scale
    scale_matrix[2, 2] = args_cli.obj_scale
    handle_mesh.apply_transform(scale_matrix)

    handle_pt = torch.tensor(
        handle_mesh.sample(4096), dtype=torch.float32, device=sim.device
    )
    print(f"[INFO]: Loaded handle mesh with {len(handle_pt)} points")

    # Simulation state tracker
    sim_state = SimulationState(num_envs=num_envs, device=sim.device)

    # Compute grasp poses from object configuration
    grasp_poses = []
    for i, task_obj_id in enumerate(TASK_IDS[:num_envs]):
        p, r = compute_grasp_pose(task_obj_id, i)
        grasp_poses.append(np.concatenate([p, r]))

    grasp_poses_w = torch.tensor(
        np.array(grasp_poses), device=sim.device, dtype=torch.float32
    )
    print(f"[INFO]: Target grasp poses (world): {grasp_poses_w}")

    # Create waypoint
    waypoint_offset = 0.15
    waypoint_poses_w = grasp_poses_w.clone()
    waypoint_poses_w[:, 0] -= waypoint_offset
    waypoint_poses_w[:, 2] += waypoint_offset
    print(f"[INFO]: Waypoint poses (world):     {waypoint_poses_w[0, 0:3]}")

    # Robot entity config
    robot_entity_cfg = SceneEntityCfg(
        "robot", joint_names=["panda_joint[1-7]"], body_names=["panda_hand"]
    )
    robot_entity_cfg.resolve(scene)
    ee_body_idx = robot_entity_cfg.body_ids[0]
    ee_jacobi_idx = ee_body_idx if not robot.is_fixed_base else ee_body_idx - 1

    # Gripper commands
    gripper_open = torch.tensor([[0.04, 0.04]], device=sim.device).repeat(num_envs, 1)
    gripper_closed = torch.tensor([[0.0, 0.0]], device=sim.device).repeat(num_envs, 1)

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    phase = 0  # (0=waypoint, 1=grasp, 2=hold)

    # Body indices - resolve once
    grasp_link = obj_config["grasp_link"]

    finger_entity_cfg = SceneEntityCfg("robot", body_names=["panda_rightfinger"])
    finger_entity_cfg.resolve(scene)
    r_finger_body_idx = finger_entity_cfg.body_ids[0]

    hand_entity_cfg = SceneEntityCfg("robot", body_names=["panda_hand"])
    hand_entity_cfg.resolve(scene)
    hand_body_idx = hand_entity_cfg.body_ids[0]

    obj_entity_cfg = SceneEntityCfg(f"object_{env_idx}", body_names=[grasp_link])
    obj_entity_cfg.resolve(scene)
    handle_body_idx = obj_entity_cfg.body_ids[0]

    print("[INFO]: Starting IK control loop...")

    marker_data = []

    # Simulation loop
    while simulation_app.is_running():
        # Reset periodically for testing
        # Phase 0: 150 (waypoint) + Phase 1: 150 (grasp) + Phase 2: 30 (close gripper) and until end execution/recovery
        if count % 2500 == 0 or sim_state.state == STATE_SUCC:
            phase = 0
            sim_state.locked_marker_idx = None
            sim_state.state = STATE_INIT
            print(f"\n[INFO]: Resetting to initial state...")  # TODO: reset object pose

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
                root_pose_w[:, 0:3],
                root_pose_w[:, 3:7],
                target_poses[:, 0:3],
                target_poses[:, 3:7],
            )

            diff_ik_controller.set_command(
                torch.cat([target_pos_b, target_quat_b], dim=-1)
            )
            count = 0

        # Phase transitions within cycle
        if count == 150 and phase == 0:
            phase = 1
            target_poses = grasp_poses_w
            print(f"[PHASE 1]: Moving to grasp pose...")

            diff_ik_controller.reset()
            root_pose_w = robot.data.root_pose_w
            target_pos_b, target_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3],
                root_pose_w[:, 3:7],
                target_poses[:, 0:3],
                target_poses[:, 3:7],
            )
            diff_ik_controller.set_command(
                torch.cat([target_pos_b, target_quat_b], dim=-1)
            )

        elif count == 300 and phase == 1:
            phase = 2
            print(f"[PHASE 2]: Closing gripper...")

            # Close gripper over next 30 steps
            for close_step in range(30):
                alpha_close = (close_step + 1) / 30.0
                gripper_pos = (
                    gripper_open * (1 - alpha_close) + gripper_closed * alpha_close
                )

                robot.set_joint_position_target(
                    robot.data.joint_pos[:, robot_entity_cfg.joint_ids],
                    joint_ids=robot_entity_cfg.joint_ids,
                )
                robot.set_joint_position_target(gripper_pos, joint_ids=slice(7, 9))
                scene.write_data_to_sim()
                sim.step()
                scene.update(sim_dt)
                count += 1

            # Initialize manipulation state
            sim_state.state = STATE_PROC
            sim_state.grasp_q = robot.data.joint_pos[:, 7:9].clone()

            hand_pose_w = robot.data.body_pose_w[:, hand_body_idx]
            hand_transf = pose_to_transform(
                hand_pose_w[0, :3], hand_pose_w[0, 3:7], sim.device
            )
            sim_state.curr_proceed_base_transf = hand_transf.clone()
            sim_state.curr_proceed_dir = get_proceeding_dir(hand_pose_w)

            print(f"[PHASE 2]: Starting manipulation (STATE_PROC)...")

        # Compute IK if in waypoint or grasp phase
        if phase < 2:
            jacobian = robot.root_physx_view.get_jacobians()[
                :, ee_jacobi_idx, :, robot_entity_cfg.joint_ids
            ]
            ee_pose_w = robot.data.body_pose_w[:, ee_body_idx]
            root_pose_w = robot.data.root_pose_w
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]

            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3],
                root_pose_w[:, 3:7],
                ee_pose_w[:, 0:3],
                ee_pose_w[:, 3:7],
            )

            joint_pos_des = diff_ik_controller.compute(
                ee_pos_b, ee_quat_b, jacobian, joint_pos
            )

            # Apply actions
            robot.set_joint_position_target(
                joint_pos_des, joint_ids=robot_entity_cfg.joint_ids
            )
            robot.set_joint_position_target(gripper_open, joint_ids=slice(7, 9))

        elif phase == 2:  # Manipulation phase
            root_pose_w = robot.data.root_pose_w
            ee_pose_w = robot.data.body_pose_w[:, ee_body_idx]
            hand_pose_w = robot.data.body_pose_w[:, hand_body_idx]
            hand_transf = pose_to_transform(
                hand_pose_w[0, :3], hand_pose_w[0, 3:7], sim.device
            )

            # Get handle transform
            handle_pose_w = scene[f"object_{env_idx}"].data.body_state_w[
                :, handle_body_idx, 0:7
            ]
            handle_transf = pose_to_transform(
                handle_pose_w[0, :3], handle_pose_w[0, 3:7], sim.device
            )

            # Transform finger markers to world frame
            r_finger_pose_w = robot.data.body_state_w[:, r_finger_body_idx, 0:7]
            r_markers_world = quat_apply(
                r_finger_pose_w[:, 3:7].repeat(len(r_finger_markers), 1),
                r_finger_markers.repeat(num_envs, 1),
            ) + r_finger_pose_w[:, :3].repeat(len(r_finger_markers), 1)

            # Contact detection and marker locking
            if sim_state.locked_marker_idx is None:
                handle_pt_world = quat_apply(
                    handle_pose_w[:, 3:7].repeat(len(handle_pt), 1),
                    handle_pt.repeat(num_envs, 1),
                ) + handle_pose_w[:, :3].repeat(len(handle_pt), 1)

                dist = torch.cdist(
                    handle_pt_world.unsqueeze(0), r_markers_world.unsqueeze(0)
                ).squeeze(0)
                finger_to_handle_dist = dist.min(dim=0)[0]
                finger_marker_contact = finger_to_handle_dist < CONTACT_THRES

                if finger_marker_contact.float().mean() > 0.1:
                    contact_idx = torch.where(finger_marker_contact)[0]
                    sim_state.lock_markers(
                        contact_idx, r_markers_world, handle_transf, hand_transf
                    )

            # Compute target transformation
            target_transf = hand_transf.clone()

            if sim_state.locked_marker_idx is not None:
                sim_state.compute_displacement(handle_transf, hand_transf)

                proc_flag = sim_state.state == STATE_PROC
                recv_flag = sim_state.state == STATE_RECV

                next_recv = proc_flag and (sim_state.marker_dspl_dist > delta_0)
                next_proc = recv_flag and (sim_state.marker_dspl_dist < delta_0 * alpha)

                if next_proc:
                    sim_state.state = STATE_PROC
                    sim_state.attempt_counter += 1
                    sim_state.recv_stuck_count = 0
                    sim_state.curr_proceed_base_transf = hand_transf.clone()
                    sim_state.curr_proceed_dir = get_proceeding_dir(hand_pose_w)
                    print(
                        f"[Step {count}] RECV -> PROC (dspl={sim_state.marker_dspl_dist:.6f})"
                    )

                if next_recv:
                    sim_state.state = STATE_RECV
                    print(
                        f"[Step {count}] PROC -> RECV (dspl={sim_state.marker_dspl_dist:.6f})"
                    )

                # Compute target based on state
                if sim_state.state == STATE_PROC:
                    sim_state.curr_proceed_base_transf[:3, 3] += (
                        sim_state.curr_proceed_dir[0] * 0.0001
                    )
                    target_transf = sim_state.curr_proceed_base_transf.clone()

                elif sim_state.state == STATE_RECV:
                    sim_state.recv_stuck_count += 1

                    recv_accept = torch.det(sim_state.marker_dspl_r) > 0.9999
                    if recv_accept:
                        target_transf = torch.matmul(
                            sim_state.marker_dspl_transf, hand_transf
                        ).float()

                    # Stuck recovery
                    if sim_state.recv_stuck_count % 250 == 249:
                        target_transf += torch.normal(
                            0, 0.01, size=target_transf.shape, device=sim.device
                        )
                        print(f"[Step {count}] Stuck, applying noise")

                curr_marker_pos_homo = torch.cat(
                    [
                        sim_state.curr_marker_pos_world,
                        torch.ones(
                            len(sim_state.curr_marker_pos_world), 1, device=sim.device
                        ),
                    ],
                    dim=-1,
                )
                curr_marker_pos_hand = torch.matmul(
                    torch.inverse(hand_transf), curr_marker_pos_homo.T
                ).T
                marker_data.append(
                    {
                        "step": count,
                        "state": "RECV" if sim_state.state == STATE_RECV else "PROC",
                        "init_pos": sim_state.init_marker_pos.cpu(),
                        "curr_pos": curr_marker_pos_hand.cpu(),
                        "displacement": sim_state.marker_dspl_dist,
                    }
                )

                # Visualize displacement
                if sim_state.curr_marker_pos_world is not None:
                    displacement_markers.visualize(
                        sim_state.curr_marker_pos_world,
                        torch.tensor([[1, 0, 0, 0]], device=sim.device).repeat(
                            len(sim_state.curr_marker_pos_world), 1
                        ),
                    )

            # Apply IK for manipulation target
            target_pos = target_transf[:3, 3].unsqueeze(0)
            target_rot_mat = target_transf[:3, :3].cpu().numpy()
            target_quat = torch.tensor(
                R.from_matrix(target_rot_mat).as_quat()[[3, 0, 1, 2]],
                device=sim.device,
                dtype=torch.float32,
            ).unsqueeze(0)

            target_pos_b, target_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], target_pos, target_quat
            )
            diff_ik_controller.set_command(
                torch.cat([target_pos_b, target_quat_b], dim=-1)
            )

            jacobian = robot.root_physx_view.get_jacobians()[
                :, ee_jacobi_idx, :, robot_entity_cfg.joint_ids
            ]
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3],
                root_pose_w[:, 3:7],
                ee_pose_w[:, 0:3],
                ee_pose_w[:, 3:7],
            )
            joint_pos_des = diff_ik_controller.compute(
                ee_pos_b, ee_quat_b, jacobian, joint_pos
            )

            robot.set_joint_position_target(
                joint_pos_des, joint_ids=robot_entity_cfg.joint_ids
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
                pos_error = torch.norm(
                    ee_pose_w[:, 0:3] - waypoint_poses_w[:, 0:3], dim=-1
                )
                print(
                    f"[Step {count:4d}] Phase 0 (Waypoint) | Error: {pos_error[0].item():.4f}m"
                )
            elif phase == 1:
                pos_error = torch.norm(
                    ee_pose_w[:, 0:3] - grasp_poses_w[:, 0:3], dim=-1
                )
                print(
                    f"[Step {count:4d}] Phase 1 (Grasp)    | Error: {pos_error[0].item():.4f}m"
                )
            elif phase == 2:
                print(
                    f"[Step {count:4d}] Phase 2 (Manip)    | State: {'RECV' if sim_state.state == STATE_RECV else 'PROC'} | Dspl: {sim_state.marker_dspl_dist:.6f}"
                )

        # Visualize
        ee_pose_w = robot.data.body_state_w[:, ee_body_idx, 0:7]
        root_pose_w = robot.data.root_pose_w
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        waypoint_marker.visualize(waypoint_poses_w[:, 0:3], waypoint_poses_w[:, 3:7])
        goal_marker.visualize(grasp_poses_w[:, 0:3], grasp_poses_w[:, 3:7])
        base_marker.visualize(root_pose_w[:, 0:3], root_pose_w[:, 3:7])

        # Contact markers visualization
        if phase >= 2:
            r_finger_pose_w = robot.data.body_state_w[:, r_finger_body_idx, 0:7]
            r_markers_world = quat_apply(
                r_finger_pose_w[:, 3:7].repeat(len(r_finger_markers), 1),
                r_finger_markers.repeat(num_envs, 1),
            ) + r_finger_pose_w[:, :3].repeat(len(r_finger_markers), 1)
            contact_markers.visualize(
                r_markers_world,
                torch.tensor([[1, 0, 0, 0]], device=sim.device).repeat(
                    len(r_markers_world), 1
                ),
            )

        # Save data periodically
        if count % 449 == 0 and len(marker_data) > 0:
            torch.save(marker_data, f"{PWD}/marker_displacement_data.pt")
            print(
                f"[INFO]: Saved {len(marker_data)} frames to marker_displacement_data.pt"
            )


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
        task_configs.append({"obj_id": obj_id, "obj_config": obj_config, "env_idx": i})
        print(f"[INFO]: Added object {obj_id}")

    scene = InteractiveScene(scene_cfg)
    sim.reset()
    
    # Run simulator
    run_simulator(sim, scene, task_configs)


if __name__ == "__main__":
    main()
    simulation_app.close()
