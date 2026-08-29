from tarec.cram_interfaces.hsrb_setup_world import hsrb_setup_world
from tarec.tarec_system.event_handler import EventDispatcher

from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.robots.hsrb import HSRB

import os
import time
import numpy as np

from pycram.datastructures.dataclasses import Context
from pycram.locations.costmaps import VisibilityCostmap
from pycram.robot_plans.actions.composite.utils.rviz import CameraVisiblePointsRviz
from pycram.tf_transformations import quaternion_multiply
from pycram.utils import get_quaternion_between_two_vectors

from semantic_digital_twin.spatial_computations.raytracer import RayTracer
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.spatial_types.spatial_types import Pose, Point3, Quaternion
from semantic_digital_twin.world import World

RVIZ_PUBLISH_WAIT_SECONDS = float(os.environ.get("MOVE_AND_PERCEIVE_RVIZ_WAIT", "0.2"))
REASON_WORLD_EACH_PERCEPTION = os.environ.get("MOVE_AND_PERCEIVE_REASON_EACH", "0") != "0"
_PERCEPTION_DEBUG_NODE = None


def get_perception_debug_node():
    global _PERCEPTION_DEBUG_NODE
    if _PERCEPTION_DEBUG_NODE is not None:
        return _PERCEPTION_DEBUG_NODE

    try:
        import rclpy

        try:
            rclpy.init()
        except:
            pass

        _PERCEPTION_DEBUG_NODE = rclpy.create_node("pycram_perception_debug")
    except ImportError:
        _PERCEPTION_DEBUG_NODE = None
    return _PERCEPTION_DEBUG_NODE


def look_at(location: Pose, robot_world: World):
    vis = VisibilityCostmap(
        min_height=1,
        max_height=1.3,
        origin=location,
        resolution=0.02,
        width=200,
        height=200,
        world=robot_world,
    )
    return vis

def simulate_perception(
    world: World,
    dispatcher: EventDispatcher,
    context: Context,
    hsrb: AbstractRobot,
):
    """
    returns a list of visible bodies detected using raytracing
    """
    node = get_perception_debug_node()

    # hsrb = HSRB.from_world(world)
    # context = Context(world=world, robot=hsrb)

    if REASON_WORLD_EACH_PERCEPTION:
        with world.modify_world():
            world_reasoner = WorldReasoner(world)
            world_reasoner.reason()
    else:
        # skip world reasoning: MOVE_AND_PERCEIVE_REASON_EACH=0
        pass



    context.evaluate_conditions = False


    ####################################################

    visualize = look_at(
        location=Pose(
            Point3(2.3, 8, 1.25),
            orientation=(Quaternion(z=-0.9995140, w=0.03117147)),
        ),
        robot_world=world,
    )

    # print(visualize.map)

    if node is not None:
        camera_details = hsrb.get_default_camera()  # for simulation: use default stats
        if hsrb.name.name == "HSRB":
            camera = world.get_body_by_name("head_rgbd_sensor_link")  # we use head mount instead of default camera
            camera_pose = Pose(
                position=Point3(x=camera.global_pose.x, y=camera.global_pose.y, z=camera.global_pose.z + 0.03),
                orientation=camera.global_pose.to_quaternion())
            forward_axis = np.asarray(
                camera_details.forward_facing_axis.to_list(), dtype=float
            ).reshape(-1)[:3]
            alignment_quaternion = get_quaternion_between_two_vectors(
                np.array([1.0, 0.0, 0.0], dtype=float),
                forward_axis,
            )
            camera_quaternion = np.asarray(camera_pose.to_quaternion().to_list(), dtype=float)
            aligned_quaternion = quaternion_multiply(camera_quaternion, alignment_quaternion)
            camera_position = np.asarray(camera_pose.to_position().to_list(), dtype=float)[:3]
        else:
            camera = camera_details
            camera_pose = camera.root.global_pose
            forward_axis = np.asarray(
                camera.forward_facing_axis.to_list(), dtype=float
            ).reshape(-1)[:3]
            alignment_quaternion = get_quaternion_between_two_vectors(
                np.array([1.0, 0.0, 0.0], dtype=float),
                forward_axis,
            )
            camera_quaternion = np.asarray(camera_pose.to_quaternion().to_list(), dtype=float)
            aligned_quaternion = quaternion_multiply(camera_quaternion, alignment_quaternion)
            camera_position = np.asarray(camera_pose.to_position().to_list(), dtype=float)[:3]

        raytracer_camera_pose = Pose.from_xyz_quaternion(
            *camera_position,
            *aligned_quaternion,
            reference_frame=camera_pose.reference_frame,
        )

        ray_tracer = RayTracer(world)
        segmentation = ray_tracer.create_segmentation_mask(
            raytracer_camera_pose,
            resolution=128,
        )
        visible_body_ids = {int(idx) for idx in segmentation[segmentation >= 0].flatten()}
        robot_body_ids = {body.index for body in hsrb.bodies}
        visible_bodies = [
            ray_tracer.index_to_body[idx]
            for idx in sorted(visible_body_ids)
            if idx not in robot_body_ids and idx in ray_tracer.index_to_body
        ]

        CameraVisiblePointsRviz(
            world=world,
            camera_pose=raytracer_camera_pose,
            node=node,
            frame_id=str(world.root.name),
            topic="/debug/camera/visible_points",
            marker_ns="camera_visible_points",
            resolution=128,
            point_scale=0.01,
            show_rays=True,
            ray_stride=12,
            ray_alpha=0.15,
            origin_scale=0.04,
            republish_hz=None,
            clear_existing=True,
        )
        # RViz publish wait sleep
        if RVIZ_PUBLISH_WAIT_SECONDS > 0:
            time.sleep(RVIZ_PUBLISH_WAIT_SECONDS)

        # dispatch perceived bodies event
        dispatcher.trigger_event(visible_bodies, world, context)
        return visible_bodies

if __name__ == '__main__':
    this_world = hsrb_setup_world()
    hsrb = HSRB.from_world(this_world[0])
    context = Context(world=this_world[0], robot=hsrb)
    simulate_perception(this_world[0], this_world[1], context, hsrb)
