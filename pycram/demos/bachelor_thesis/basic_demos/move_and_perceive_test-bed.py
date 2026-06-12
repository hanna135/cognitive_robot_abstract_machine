import os
import random
from contextlib import contextmanager
from enum import Enum

from docutils.nodes import reference

from demos.bachelor_thesis.actions.random_location_generator import random_location_list, \
    pose_to_homogeneous_transformation_matrix_from_xyz_quaternion
from demos.bachelor_thesis.actions.simulate_perception import simulate_perception
from demos.bachelor_thesis.events.event_handler import EventDispatcher
from pycram import plans
from pycram.datastructures.enums import Arms
from pycram.motion_executor import simulated_robot
from pycram.plans.factories import sequential, execute_single
from pycram.robot_plans.actions.core.navigation import NavigateAction
from pycram.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction, ParkArmsWithHighTorsoAction
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import WorldEntityNotFoundError
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.semantic_annotations.mixins import HasSupportingSurface, HasRootBody
from semantic_digital_twin.world_description.geometry import Color, Scale
from semantic_digital_twin.semantic_annotations.semantic_annotations import Bowl, Spoon, Bottle, Cup, ShelfLayer, \
    CounterTop, Table, Wardrobe, Cabinet, Oven, DishwasherTab, Banana, Bread, Knife, Plate
from semantic_digital_twin.spatial_types import Point3, Quaternion
from semantic_digital_twin.spatial_types.spatial_types import Pose, HomogeneousTransformationMatrix
from semantic_digital_twin.robots.hsrb import HSRB
from pycram.datastructures.dataclasses import Context
from demos.bachelor_thesis.hsrb_setup_world import hsrb_setup_world
from time import sleep

from demos.bachelor_thesis.classes_and_methods.helper_classes_and_methods import Environment, \
    timed_plan, timed_parse_stl, debug_task_list_for_demo, print_sorted_task_list, sort_tasks, \
    compare_robot_world_with_real, print_object_locations

environment = Environment.TestBed
random_set_of_objects = False

#------------------ standard setup -------------------------------------------------------------------------------------
world, dispatcher = hsrb_setup_world(environment=environment)

with world.modify_world():
    dishwasher_rack = Table.create_with_new_body_in_world(
                    world=world,
                    name=PrefixedName("dishwasher_rack"),
                    world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=2, y=1.42, z=0.07),
                    scale=Scale(x=0.744, y=0.650, z=0.14)
                )
    for color in dishwasher_rack.bodies[0].visual.shapes:
        color.color = Color.RED()
#
# for bod in world.bodies:
#     if not "hsrb" == bod.name.prefix:
#         print(bod.name)
dispatcher.known_furniture = world.bodies

with world.modify_world():
    world.add_semantic_annotations(
        [
            Table(root=world.get_body_by_name("table_bedside__table_bedside__base_link"), name=PrefixedName("table_bedside__table_bedside__base_link")),
            Table(root=world.get_body_by_name("nightstand1__bedroom_nightstand__link_0"), name=PrefixedName("nightstand1__bedroom_nightstand__link_0")),
            Table(root=world.get_body_by_name("nightstand2__bedroom_nightstand__link_0"), name=PrefixedName("nightstand2__bedroom_nightstand__link_0")),
            Table(root=world.get_body_by_name("dinning_room_table__dinning_room_table__base_link"), name=PrefixedName("dinning_room_table__dinning_room_table__base_link")),
            Table(root=world.get_body_by_name("table_living_room__table_living_room__base_link"), name=PrefixedName("table_living_room__table_living_room__base_link")),
            Table(root=world.get_body_by_name("tv_table__tv_table__link_0"), name=PrefixedName("tv_table__tv_table__link_0")),
            CounterTop(root=world.get_body_by_name("cabinet_diamondback__cabinet_diamondback__base_link"), name=PrefixedName("cabinet_diamondback__cabinet_diamondback__base_link")),
            CounterTop(root=world.get_body_by_name("cabinet_ikea_galant__cabinet_ikea_galant__base_link"), name=PrefixedName("cabinet_ikea_galant__cabinet_ikea_galant__base_link")),
        ]
    )

dispatcher.dining_table = world.get_semantic_annotation_by_name("dinning_room_table__dinning_room_table__base_link")
dispatcher.correct_location_food = world.get_semantic_annotation_by_name("table_living_room__table_living_room__base_link")
dispatcher.correct_location_drinks = world.get_semantic_annotation_by_name("nightstand1__bedroom_nightstand__link_0")
dispatcher.correct_location_tableware_clean = world.get_semantic_annotation_by_name("cabinet_diamondback__cabinet_diamondback__base_link")
dispatcher.correct_location_tableware_dirty = world.get_semantic_annotation_by_name("cabinet_ikea_galant__cabinet_ikea_galant__base_link")
dispatcher.correct_location_all_other_items = world.get_semantic_annotation_by_name("tv_table__tv_table__link_0")
dispatcher.environment_boundaries = [
    Point3(x=-3.07, y=5.51, z=0),
    Point3(x=3.37, y=5.51, z=0),
    Point3(x=-3.08, y=-3.22, z=0),
    Point3(x=3.37, y=-1.65),
]
#
# #-----------------------------------------------------------------------------------------------------------------------
#
#
semantic_objects = []

bowl = timed_parse_stl("bowl", "bowl.stl")
semantic_objects.append(bowl)

spoon = timed_parse_stl("spoon", "spoon.stl")
semantic_objects.append(spoon)

pitcher = timed_parse_stl("pitcher", "Static_MilkPitcher.stl")
semantic_objects.append(pitcher)

coke = timed_parse_stl("coke", "Static_CokeBottle.stl")
semantic_objects.append(coke)

jeroen_cup = timed_parse_stl("jeroen cup", "jeroen_cup.stl")
semantic_objects.append(jeroen_cup)

dishwasher_tab = timed_parse_stl("dishwasher tab", "dishwasher_tab.stl")
semantic_objects.append(dishwasher_tab)

banana = timed_parse_stl("banana", "banana.stl")
semantic_objects.append(banana)

bread = timed_parse_stl("bread", "bread.stl")
semantic_objects.append(bread)

knife = timed_parse_stl("knife", "knife.stl")
semantic_objects.append(knife)

plate = timed_parse_stl("plate", "plate.stl")
semantic_objects.append(plate)

locs = random_location_list(world, len(semantic_objects))

with world.modify_world():
    for i in range(0, len(semantic_objects)):
        print(i)
        # true if not random_set, if random_set than randomness has to decide if true
        if not random_set_of_objects or random.random() < 0.7:
            world.merge_world_at_pose(
                semantic_objects[i],
                pose_to_homogeneous_transformation_matrix_from_xyz_quaternion(locs[i], world)
            )

    semantic_annotations = [
        (Bowl, "bowl.stl"),
        (Spoon, "spoon.stl"),
        (Bottle, "Static_MilkPitcher.stl"),
        (Bottle, "Static_CokeBottle.stl"),
        (Cup, "jeroen_cup.stl"),
        (DishwasherTab, "dishwasher_tab.stl"),
        (Banana, "banana.stl"),
        (Bread, "bread.stl"),
        (Knife, "knife.stl"),
        (Plate, "plate.stl")
    ]

    for sem_ann in semantic_annotations:
        try:
            obj = sem_ann[0](root=world.get_body_by_name(sem_ann[1]), name=PrefixedName(sem_ann[1]))
            world.add_semantic_annotation(obj)
            print(obj)
        except WorldEntityNotFoundError:
            pass

    supporting_surfaces = []

    supporting_surfaces.append(world.get_semantic_annotation_by_name("table_bedside__table_bedside__base_link"))
    supporting_surfaces.append(world.get_semantic_annotation_by_name("nightstand1__bedroom_nightstand__link_0"))
    supporting_surfaces.append(world.get_semantic_annotation_by_name("nightstand2__bedroom_nightstand__link_0"))
    supporting_surfaces.append(world.get_semantic_annotation_by_name("dinning_room_table__dinning_room_table__base_link"))
    supporting_surfaces.append(world.get_semantic_annotation_by_name("table_living_room__table_living_room__base_link"))
    supporting_surfaces.append(world.get_semantic_annotation_by_name("tv_table__tv_table__link_0"))
    supporting_surfaces.append(world.get_semantic_annotation_by_name("cabinet_diamondback__cabinet_diamondback__base_link"))
    supporting_surfaces.append(world.get_semantic_annotation_by_name("cabinet_ikea_galant__cabinet_ikea_galant__base_link"))

    for surface in supporting_surfaces:
        if isinstance(surface, HasSupportingSurface):
            surface.calculate_supporting_surface()

try:
    import rclpy
    try:
        rclpy.init()
    except:
        pass
    from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
        VizMarkerPublisher,
    )

    node = rclpy.create_node("viz_marker")
    v = VizMarkerPublisher(_world=world, node=node).with_tf_publisher()
except ImportError:
    node = None

hsrb = HSRB.from_world(world)

context = Context(world=world, robot=hsrb)

with world.modify_world():
    world_reasoner = WorldReasoner(world)
    world_reasoner.reason()


context.evaluate_conditions = False

# TODO: plan stuff here