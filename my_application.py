import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--idx", default="46466", type=int)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--capture", default=False, action="store_true")
parser.add_argument("--obj_scale", default="0.5", type=float)
parser.add_argument("--x_offset", default="0.75", type=float)
parser.add_argument("--y_offset", default="0.0", type=float)
parser.add_argument("--z_offset", default="0.0", type=float)
args = parser.parse_args()

# launch Isaac Sim before any other imports
# default first two lines in any standalone application
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})  # we can also run as headless.


import json
import os
from collections import defaultdict
from datetime import datetime

import carb
import numpy as np
import omni.graph.action
import omni.kit
import omni.kit.commands
import omni.usd
import torch
import trimesh as tm
from isaacsim.asset.importer.urdf import _urdf
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.prims import SingleXFormPrim as XFormPrim
from isaacsim.core.prims import SingleArticulation as Articulation
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.robot_motion.motion_generation import RmpFlow, ArticulationMotionPolicy
from isaacsim.robot_motion.motion_generation.interface_config_loader import (
    load_supported_motion_policy_config,
)
from pxr import UsdLux, Sdf, Gf, UsdPhysics, UsdShade
from omni.physx.scripts import utils, physicsUtils
import xml.etree.ElementTree as ET
from common import set_drive_parameters

from tacman_utils.utils_sim import get_default_import_config, init_capture

# Hyperparams
delta_0 = 0.0004
alpha = 0.6

CONTACT_AREAS = {
    "panda": {
        "L": [
            [-0.008787, 0.000071, 0.036023],
            [0.008787, 0.000071, 0.036023],
            [0.008787, 0.000071, 0.053879],
            [-0.008787, 0.000071, 0.053879],
        ],
        "R": [
            [-0.008787, 0.000071, 0.036023],
            [0.008787, 0.000071, 0.036023],
            [0.008787, 0.000071, 0.053879],
            [-0.008787, 0.000071, 0.053879],
        ],
    }
}

PWD = os.path.dirname(os.path.abspath(__file__))

all_grasps = json.load(open(f"{PWD}/data/gapartnet/grasps.json", "r"))
all_configs = json.load(open(f"{PWD}/data/gapartnet/selected.json", "r"))
all_grasps = {k: v for k, v in all_grasps.items() if not v == {}}

TASK_IDS = [str(args.idx)]

quat_diff = lambda q1, q2: 2 * np.arccos(np.abs(np.dot(q1, q2)))
quat_diff_batch = lambda q1, q2: 2 * torch.acos(
    torch.abs(torch.bmm(q1.unsqueeze(1), q2.unsqueeze(-1)).squeeze(-1))
)

if args.capture:
    init_capture()


class Manipulation(BaseTask):
    def __init__(
        self, i_env, obj_id, friction_mat, report_dir, offset=None, hand_name="panda"
    ):
        super().__init__(name=f"{i_env}_{obj_id}", offset=offset)

        self.object_dir = f"{PWD}/data/gapartnet/{obj_id}"
        self.obj_config = all_configs[obj_id]
        self.time_stamp = str(int(datetime.now().timestamp()))
        self.capture_dir = os.path.join(self.object_dir, f"result-{self.time_stamp}")
        self.report_json = os.path.join(
            self.object_dir, f"result-{self.time_stamp}.json"
        )
        self.report_pt = os.path.join(self.object_dir, f"result-{self.time_stamp}.pt")

        self.i_env = i_env
        self.hand_name = "panda"
        self.obj_id = str(obj_id)

        # TODO: remove, since not used elsewhere
        # self.friction_mat = friction_mat

        self.scene_prim = f"/World/Env_{self.i_env}"
        self.object_prim_path = f"{self.scene_prim}/Obj_{self.obj_id}"
        self.franka_prim_path = f"{self.scene_prim}/Manipulator"

        self.locked_marker_idx = None

        # Data dump
        self.report_dir = report_dir
        os.makedirs(self.report_dir, exist_ok=True)

        self.data = []
        self.attempt_counter = 0

        # Dummy
        self.q = None
        self.hit_wall = False
        self.to_hi, self.to_lo = [], []

        self.x_offset = all_grasps[self.obj_id].get("x_offset", 0.0)
        self.y_offset = all_grasps[self.obj_id].get("y_offset", 0.0)
        self.z_offset = all_grasps[self.obj_id].get("z_offset", 0.0)

        print(f"Manipulating {obj_id}")

    def set_up_scene(self, scene):
        super().set_up_scene(scene)

        # Load robot and its IK solver using add_reference_to_stage
        path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
        add_reference_to_stage(path_to_robot_usd, self.franka_prim_path)
        self._franka = Articulation(self.franka_prim_path, name=f"manipulator_{self.i_env}")
        scene.add(self._franka)

        # Load object
        self.target_link = "handle"

        self.base_link = self.obj_config["base_link"]
        self.handle_base_link = self.obj_config["target_part"]
        self.target_link = self.obj_config["grasp_link"]
        self.all_links = self.obj_config["all_links"]

        import_config = get_default_import_config()
        urdf_path = os.path.join(self.object_dir, "mobility_relabel_gapartnet.urdf")

        print(f"Loading URDF from: {urdf_path}")
        result, prim_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=urdf_path,
            import_config=import_config,
        )

        if prim_path:
            print(f"URDF root at: {prim_path}, moving to: {self.object_prim_path}")
            omni.kit.commands.execute(
                "MovePrim", path_from=prim_path, path_to=self.object_prim_path
            )
        else:
            carb.log_error("Could not resolve imported URDF prim; object will not be placed.")

        self.target_joint = self.obj_config["target_joint"]
        self.target_joint_path = f"{self.object_prim_path}/{self.target_joint}"
        self.target_joint_name = self.target_joint_path.split("/")[-1]

        # Load point clouds for computation
        self.handle_mesh = tm.load(os.path.join(PWD, self.obj_config['grasp_part_mesh']), force='mesh')
        self.handle_pt = torch.tensor(self.handle_mesh.sample(4096), dtype=torch.float32, device='cuda')

        self.l_finger_kpt = torch.tensor(CONTACT_AREAS[self.hand_name]["L"], dtype=torch.float32, device='cuda')
        self.r_finger_kpt = torch.tensor(CONTACT_AREAS[self.hand_name]["R"], dtype=torch.float32, device='cuda')
        self.finger_xx, self.finger_yy = torch.linspace(0, 1, 10, device='cuda'), torch.linspace(0, 1, 10,
                                                                                                 device='cuda')
        self.l_finger_grid, self.r_finger_grid = torch.stack(
            torch.meshgrid(self.finger_xx, self.finger_yy, indexing='ij'), dim=-1
        ).reshape(-1, 2).clone(), torch.stack(
            torch.meshgrid(self.finger_xx, self.finger_yy, indexing='ij'), dim=-1
        ).reshape(-1, 2).clone()
        self.l_finger_pt = self.l_finger_kpt[0].unsqueeze(0) + self.l_finger_grid[:, 0].unsqueeze(-1) * (
                    self.l_finger_kpt[1] - self.l_finger_kpt[0]).unsqueeze(0) + self.l_finger_grid[:, 1].unsqueeze(
            -1) * (self.l_finger_kpt[3] - self.l_finger_kpt[0]).unsqueeze(0)
        self.r_finger_pt = self.r_finger_kpt[0].unsqueeze(0) + self.r_finger_grid[:, 0].unsqueeze(-1) * (
                    self.r_finger_kpt[1] - self.r_finger_kpt[0]).unsqueeze(0) + self.r_finger_grid[:, 1].unsqueeze(
            -1) * (self.r_finger_kpt[3] - self.r_finger_kpt[0]).unsqueeze(0)

        stage = omni.usd.get_context().get_stage()
        
        dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
        dome.CreateIntensityAttr().Set(500.0)

        self.hand_prim = stage.GetPrimAtPath(f"{self.franka_prim_path}/panda_hand")
        self.object_prim = stage.GetPrimAtPath(self.object_prim_path)
        self.r_finger_prim = stage.GetPrimAtPath(f"{self.franka_prim_path}/panda_rightfinger")
        self.l_finger_prim = stage.GetPrimAtPath(f"{self.franka_prim_path}/panda_leftfinger")
        self.finger_joint_prim_1 = stage.GetPrimAtPath(f"{self.franka_prim_path}/panda_hand/panda_finger_joint1")
        self.finger_joint_prim_2 = stage.GetPrimAtPath(f"{self.franka_prim_path}/panda_hand/panda_finger_joint2")
        self.handle_prim = stage.GetPrimAtPath(f"{self.object_prim_path}/{self.target_link}")
        self.base_link_prim = stage.GetPrimAtPath(f"{self.object_prim_path}/base_link")
        self.target_joint_type = self.obj_config['joint_type']

        self.object_prim.GetAttribute("xformOp:translate").Set(tuple(
            Gf.Vec3f(args.x_offset + self.x_offset, args.y_offset + self.y_offset,
                     self.obj_config['height'] * args.obj_scale + self.z_offset)))
        self.object_prim.GetAttribute("xformOp:scale").Set(
            tuple(Gf.Vec3f(args.obj_scale, args.obj_scale, args.obj_scale)))

        ## Set object position
        self._task_objects["Manipulator"] = self._franka
        self._task_objects["Object"] = XFormPrim(prim_path=self.object_prim_path, name=f"object-{self.i_env}")
        self._move_task_objects_to_their_frame()

        # Goal config
        self.succ_dof_pos = 0.25 if self.target_joint_type == "slider" else np.pi / 3

        # Configure finger drives using the helper from `common.py` which
        # safely creates/get attributes for drives. Use a position drive with
        # a large max_force so fingers can close reliably.
        finger_drive_1 = UsdPhysics.DriveAPI.Get(self.finger_joint_prim_1, "linear")
        finger_drive_2 = UsdPhysics.DriveAPI.Get(self.finger_joint_prim_2, "linear")
        # Use set_drive_parameters to create attrs if missing and set sensible
        # defaults: target=0 (closed), stiffness and damping large enough, and
        # max_force same as original implementation (1e4).
        set_drive_parameters(finger_drive_1, "position", 0, stiffness=1e6, damping=1e5, max_force=1e4)
        set_drive_parameters(finger_drive_2, "position", 0, stiffness=1e6, damping=1e5, max_force=1e4)
        
        for link in self.all_links:
            prim_path = f"{self.object_prim_path}/{link}"
            prim = stage.GetPrimAtPath(prim_path)
            if link == self.target_link or link == self.base_link or link == self.handle_base_link:
                UsdPhysics.MassAPI.Apply(prim).CreateMassAttr().Set(0.25)
                continue

            # TODO: Check if we should remove rigid body or not
            print(f"Removing collider for {link} (keeping rigid body)")
            utils.removeCollider(prim)
            #utils.removeRigidBody(prim)

        if self.obj_id in ["20043", "32932", "34610", "45243", "45677"]:
            utils.setRigidBody(self.handle_prim, "convexHull", False)

        elif self.obj_id in ["46462", "45756"]:
            utils.setRigidBody(self.handle_prim, "convexDecomposition", False)
        else:
            utils.setRigidBody(self.handle_prim, "boundingCube", False)

        print(f"Loaded {self.obj_id}")

    def get_observations(self):
        pass

    def pre_step(self, control_index, simulation_time):
        pass

    def report():
        pass

    def post_reset(self):
        pass

    # TODO: See if needed
    def _setup_physics_material(self, path, physics_material_path):
        stage = omni.usd.get_context().get_stage()
        collisionAPI = UsdPhysics.CollisionAPI.Get(stage, path)
        prim = stage.GetPrimAtPath(path)
        if not collisionAPI:
            collisionAPI = UsdPhysics.CollisionAPI.Apply(prim)
        # apply material
        physicsUtils.add_physics_material_to_prim(stage, prim, physics_material_path)

    def setup_ik_solver(self, franka):
        from isaacsim.robot_motion.motion_generation.interface_config_loader import load_supported_lula_kinematics_solver_config
        from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver, ArticulationKinematicsSolver

        kinematics_config = load_supported_lula_kinematics_solver_config("Franka")
        self._kine_solver = LulaKinematicsSolver(**kinematics_config)
        self._art_kine_solver = ArticulationKinematicsSolver(self._franka, self._kine_solver, "panda_hand")

    def set_grasp_pose(self):
        # p: world position, r: quaternion order [x,y,z,w] or appropriate conversion
        p, r = all_grasps[self.obj_id]['p'], all_grasps[self.obj_id]['R']
        p = np.asarray(p) * args.obj_scale
        p[0] += args.x_offset + self.x_offset
        p[1] += args.y_offset + self.y_offset
        p[2] += self.obj_config['height'] * args.obj_scale + self.z_offset
        r = R.from_euler("ZYX", r, False).as_quat()  # may need reordering to [x,y,z,w]

        self.grasp_p = p

        print(f"Setting grasp pose to {p}, {r}")

        robot_base_translation, robot_base_orientation = self._franka.get_world_pose()
        self._kine_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)

        action, ik_success = self._art_kine_solver.compute_inverse_kinematics(p + self._offset, r)
        if ik_success:
            action.joint_positions[-1] = 0.04
            action.joint_positions[-2] = 0.04
            self._franka.set_joint_positions(action.joint_positions)
            self._franka.set_joint_velocities([0.0] * self._franka.get_num_dofs())
            
            # moved the gripper into success branch
            from isaacsim.core.utils.types import ArticulationAction
            self._franka.gripper.apply_action(ArticulationAction([0.00, 0.00]))
        
        return ik_success

    def do_ik(self, p, r):
        return self._art_kine_solver.compute_inverse_kinematics(p + self._offset, r)


world = World()
world.scene.add_default_ground_plane()
stage = omni.usd.get_context().get_stage()
world._physics_context.set_gravity(value=0.0)

stage.GetPrimAtPath("/World/defaultGroundPlane/Environment").GetAttribute(
    "xformOp:translate"
).Set(Gf.Vec3f(0.0, 0.0, -1.0))
stage.GetPrimAtPath("/World/defaultGroundPlane/GroundPlane").GetAttribute(
    "xformOp:translate"
).Set(Gf.Vec3f(0.0, 0.0, -1.0))

n_tasks = len(TASK_IDS)

_tasks = []
_frankas = []
_rmpflows = []
_art_rmpflows = []
_art_controllers = []
target_joints = []
target_joint_drive = []

# 0 - Exploration, 1 - Modification, 2 - Finished.
env_states = torch.zeros([n_tasks], dtype=torch.int32, device="cuda")

assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find nucleus server with /Isaac folder")

# draw = _debug_draw.acquire_debug_draw_interface()

n_each_row = 3
spacing = 2.0

time_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
data_dir = os.path.join(PWD, "data", "manipulation", time_tag)


light_prim = stage.GetPrimAtPath("/World/defaultGroundPlane/SphereLight")
light_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3f(0, -3.0, 6.0))
light_prim.GetAttribute("xformOp:scale").Set(Gf.Vec3f(0.015, 0.015, 0.015))

_material_static_friction = 1.0
_material_dynamic_friction = 1.0
_material_restitution = 0.0
_physicsMaterialPath = None

if _physicsMaterialPath is None:
    _physicsMaterialPath = (
        stage.GetPrimAtPath("/World").GetPath().AppendChild("physicsMaterial")
    )
    UsdShade.Material.Define(stage, _physicsMaterialPath)
    material = UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(_physicsMaterialPath))
    material.CreateStaticFrictionAttr().Set(_material_static_friction)
    material.CreateDynamicFrictionAttr().Set(_material_dynamic_friction)
    material.CreateRestitutionAttr().Set(_material_restitution)

for i_task, obj_id in enumerate(TASK_IDS):
    task_data_dir = os.path.join(data_dir, f"{i_task}_{obj_id}")
    # Create the environment prim before adding the task
    env_prim_path = f"/World/Env_{i_task}"
    if not stage.GetPrimAtPath(env_prim_path):
        stage.DefinePrim(env_prim_path, "Xform")
    
    world.add_task(
        Manipulation(
            i_task,
            obj_id,
            _physicsMaterialPath,
            report_dir=task_data_dir,
            offset=np.array([i_task // n_each_row, i_task % n_each_row, 0.0]) * spacing,
        )
    )

# Resetting the world needs to be called before querying anything related to an articulation specifically.
# Its recommended to always do a reset after adding your assets, for physics handles to be propagated properly
world.reset()

# Initialize RmpFlow objects for each task
# see https://docs.isaacsim.omniverse.nvidia.com/5.0.0/manipulators/manipulators_rmpflow.html#loading-rmpflow-for-supported-robots
for i_task, obj_id in enumerate(TASK_IDS):
    _tasks.append(world.get_task(name=f"{i_task}_{obj_id}"))

    _frankas.append(_tasks[i_task]._franka)
    _art_controllers.append(_frankas[i_task].get_articulation_controller())
    
    # Only append target joint if it exists
    if hasattr(_tasks[i_task], 'target_joint_prim') and _tasks[i_task].target_joint_prim:
        target_joints.append(_tasks[i_task].target_joint_prim)
    else:
        carb.log_warn(f"Task {i_task} has no target joint prim")

    # Load RMPflow configuration for supported Franka robot
    rmp_config = load_supported_motion_policy_config("Franka", "RMPflow")
    
    # Initialize RmpFlow object
    _rmpflows.append(RmpFlow(**rmp_config))
    
    # Use ArticulationMotionPolicy wrapper to connect rmpflow to the Franka robot articulation
    _art_rmpflows.append(ArticulationMotionPolicy(_frankas[i_task], _rmpflows[i_task]))

    # Setup IK solver and disable gravity
    f = world.scene.get_object(f"manipulator_{i_task}")
    if hasattr(_tasks[i_task], 'setup_ik_solver'):
        _tasks[i_task].setup_ik_solver(f)
    f.disable_gravity()

target_joint_drive = [UsdPhysics.DriveAPI.Apply(target_joints[j].GetPrim(),
                                                "linear" if _tasks[j].target_joint_type == "slider" else "angular") for
                      j in range(len(target_joints))]

# Keep simulation running until manually closed
print("Simulation loaded. Press Ctrl+C or close the window to exit.")
while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()
