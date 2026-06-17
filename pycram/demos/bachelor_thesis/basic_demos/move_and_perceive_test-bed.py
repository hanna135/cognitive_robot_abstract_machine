import os
import random
from contextlib import contextmanager
from enum import Enum

from docutils.nodes import reference
from typing_extensions import Any

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
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.mixins import HasSupportingSurface, HasRootBody
from semantic_digital_twin.world_description.geometry import Color, Scale
from semantic_digital_twin.semantic_annotations.semantic_annotations import Bowl, Spoon, Bottle, Cup, ShelfLayer, \
    CounterTop, Table, Wardrobe, Cabinet, Oven, DishwasherTab, Banana, Bread, Knife, Plate, Milk
from semantic_digital_twin.spatial_types import Point3, Quaternion
from semantic_digital_twin.spatial_types.spatial_types import Pose, HomogeneousTransformationMatrix
from semantic_digital_twin.robots.hsrb import HSRB
from pycram.datastructures.dataclasses import Context
from demos.bachelor_thesis.hsrb_setup_world import hsrb_setup_world
from time import sleep

from demos.bachelor_thesis.classes_and_methods.helper_classes_and_methods import Environment, \
    timed_plan, timed_parse_stl, debug_task_list_for_demo, print_sorted_task_list, sort_tasks, \
    compare_robot_world_with_real, print_object_locations, print_locs_as_copy_paste_list


def main(robot: type[PR2] | type[HSRB], locations: list[Any] = None, random_set_of_objects: bool = True):
    environment = Environment.TestBed


    #------------------ standard setup -------------------------------------------------------------------------------------
    world, dispatcher = hsrb_setup_world(environment=environment, robot=robot)

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
                CounterTop(root=world.get_body_by_name("shelf__shelf__base_link"), name=PrefixedName("shelf__shelf__base_link")),
                CounterTop(root=world.get_body_by_name("cabinet_ikea_galant__cabinet_ikea_galant__base_link"), name=PrefixedName("cabinet_ikea_galant__cabinet_ikea_galant__base_link")),
            ]
        )

    dispatcher.dining_table = world.get_semantic_annotation_by_name("dinning_room_table__dinning_room_table__base_link")
    dispatcher.correct_location_food = world.get_semantic_annotation_by_name("table_living_room__table_living_room__base_link")
    dispatcher.correct_location_drinks = world.get_semantic_annotation_by_name("nightstand1__bedroom_nightstand__link_0")
    dispatcher.correct_location_tableware_clean = world.get_semantic_annotation_by_name("shelf__shelf__base_link")
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

    if locations is None:
        locs = random_location_list(world, len(semantic_objects))
        print_locs_as_copy_paste_list(locs)
    else:
        locs = locations

    with world.modify_world():
        for i in range(0, len(semantic_objects)):
            # true if not random_set, if random_set than randomness has to decide if true
            if not random_set_of_objects or random.random() < 0.7:
                world.merge_world_at_pose(
                    semantic_objects[i],
                    pose_to_homogeneous_transformation_matrix_from_xyz_quaternion(locs[i], world)
                )

        semantic_annotations = [
            (Bowl, "bowl.stl"),
            (Spoon, "spoon.stl"),
            (Milk, "Static_MilkPitcher.stl"),
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
            except WorldEntityNotFoundError:
                pass

        supporting_surfaces = []

        supporting_surfaces.append(world.get_semantic_annotation_by_name("table_bedside__table_bedside__base_link"))
        supporting_surfaces.append(world.get_semantic_annotation_by_name("nightstand1__bedroom_nightstand__link_0"))
        supporting_surfaces.append(world.get_semantic_annotation_by_name("nightstand2__bedroom_nightstand__link_0"))
        supporting_surfaces.append(world.get_semantic_annotation_by_name("dinning_room_table__dinning_room_table__base_link"))
        supporting_surfaces.append(world.get_semantic_annotation_by_name("table_living_room__table_living_room__base_link"))
        supporting_surfaces.append(world.get_semantic_annotation_by_name("tv_table__tv_table__link_0"))
        supporting_surfaces.append(world.get_semantic_annotation_by_name("shelf__shelf__base_link"))
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

    hsrb = robot.from_world(world)
    if robot == HSRB:
        arms = Arms.LEFT
    else:
        arms = Arms.BOTH

    context = Context(world=world, robot=hsrb)

    with world.modify_world():
        world_reasoner = WorldReasoner(world)
        world_reasoner.reason()


    context.evaluate_conditions = False

    # get coordinates from publish point in rviz2
    plan_labels = [
        "park arms",
        "move to entrance",
        "move to table white",
        "move to table wood",
        "move to kitchen counter",
        "move to dishwasher",
        "move to shelve",
        "collision avoidance pose 1",
        "move to tv table",
        "move to other table",
        "collision avoidance pose 2",
        "collision avoidance pose 3",
        "move to bedside table 1",
        "move to bedside table 2",
        "move high",
        "park arms high",
    ]

    plan_driving = [
        timed_plan("park arms", ParkArmsAction(arms), context),

        # dishwasher
        timed_plan("move to entrance", NavigateAction(
            target_location=Pose(Point3(0.9006584, -3.266999, 0.0),
                                 orientation=(Quaternion(z=0.78234807, w=0.622841458)),
                                 reference_frame=world.root), keep_joint_states=True), context),

        # kitchen counter
        timed_plan("move to table white", NavigateAction(
            target_location=Pose(Point3(0.89542120, -2.18257379, 0), orientation=(Quaternion(z=0.04796719, w=0.998848911)),
                                 reference_frame=world.root), keep_joint_states=True), context),

        timed_plan("move to table wood", NavigateAction(
            target_location=Pose(Point3(0.77867496, -0.152554, 0), orientation=(Quaternion(z=0.0823947553, w=0.99659977)),
                                 reference_frame=world.root), keep_joint_states=True), context),

        # high kitchen counter
        timed_plan("move to kitchen counter", NavigateAction(
            target_location=Pose(Point3(1.00579440, 0.764235258, 0), orientation=(Quaternion(z=0.500546298, w=0.8657097681)),
                                 reference_frame=world.root), keep_joint_states=True), context),

        timed_plan("move to dishwasher", NavigateAction(
            target_location=Pose(Point3(1.1154820, 0.96872162, 0), orientation=(Quaternion(z=0.13903434, w=0.99028756005)),
                                 reference_frame=world.root), keep_joint_states=True), context),

        # transition point to not drive through counter
        timed_plan("move to shelve", NavigateAction(
            target_location=Pose(Point3(-0.658700, 1.57239699, 0), orientation=(Quaternion(z=-0.9968945, w=0.078747932)),
                                 reference_frame=world.root), keep_joint_states=True), context),

        # pc desk
        timed_plan("collision avoidance pose 1", NavigateAction(
            target_location=Pose(Point3(-2.03033256, 1.341100692, 0), orientation=(Quaternion(z=-0.690487, w=0.72334475)),
                                 reference_frame=world.root), keep_joint_states=True), context),

        # popcorn table
        timed_plan("move to tv table", NavigateAction(
            target_location=Pose(Point3(-1.6441851, -1.02006351, 0), orientation=(Quaternion(z=0.999606, w=0.02804165)),
                                 reference_frame=world.root), keep_joint_states=True), context),

        # transition point to not drive in wall
        timed_plan("move to other table", NavigateAction(
            target_location=Pose(Point3(-1.93026709, -2.5748128, 0), orientation=(Quaternion(z=0.9998065, w=0.019670357)),
                                 reference_frame=world.root), keep_joint_states=True), context),

        # sofa table
        timed_plan("collision avoidance pose 2", NavigateAction(
            target_location=Pose(Point3(-1.907081723, 1.5224895, 0), orientation=(Quaternion(z=0.0043536, w=0.99999052)),
                                 reference_frame=world.root), keep_joint_states=True), context),

        # living room table
        timed_plan("collision avoidance pose 3", NavigateAction(
            target_location=Pose(Point3(-0.72870624, 1.615085601, 0), orientation=(Quaternion(z=0.0043536, w=0.9999905)),
                                 reference_frame=world.root), keep_joint_states=True), context),

        # shelf
        timed_plan("move to bedside table 1", NavigateAction(
            target_location=Pose(Point3(-0.7082238, 4.04637527, 0), orientation=(Quaternion(z=0.7035752681, w=0.710620744)),
                                 reference_frame=world.root), keep_joint_states=True), context),

        timed_plan("move to bedside table 2", NavigateAction(
            target_location=Pose(Point3(-0.9456942, 2.70876407, 0),
                                 orientation=(Quaternion(z=0.91994690, w=0.39204296280)),
                                 reference_frame=world.root), keep_joint_states=True), context),

        timed_plan("move high", MoveTorsoAction(TorsoState.HIGH), context),

        timed_plan("park arms high", ParkArmsWithHighTorsoAction(Arms.LEFT), context),

    ]

    with simulated_robot:
        for index, (label, plan) in enumerate(zip(plan_labels, plan_driving), start=1):
            # skip park arms high for every other robot than hsrb
            step_label = f"{index:02d}/{len(plan_driving)} {label}"
            if label == "park arms high" and not hsrb.name.name == "HSRB":
                pass
            else:
                print(step_label)
                plan.perform()
                visible_bodies = simulate_perception(
                    world,
                    dispatcher,
                    context,
                    hsrb,
                )
            visible_count = len(visible_bodies) if visible_bodies is not None else 0
        debug_task_list_for_demo(dispatcher)

    print_sorted_task_list(sort_tasks(dispatcher.activated_tasks, 300), 300)

    res = compare_robot_world_with_real(dispatcher, world, context)
    print(res)

    print_object_locations(dispatcher, world)

    return res, dispatcher.perceived_objects, world.bodies
if __name__ == "__main__":
    locats = [
        Pose(Point3(x=2.10113, y=1.32413, z=0.14), Quaternion(x=0, y=0, z=0, w=1)),
        Pose(Point3(x=-2.71624, y=-2.61606, z=0.4), Quaternion(x=0, y=0, z=0, w=1)),
        Pose(Point3(x=-0.703271, y=4.9674, z=0.554529), Quaternion(x=0, y=0, z=0, w=1)),
        Pose(Point3(x=-2.90538, y=4.8972, z=0.554529), Quaternion(x=0, y=0, z=0, w=1)),
        Pose(Point3(x=2.18391, y=-1.95688, z=0.8), Quaternion(x=0, y=0, z=0, w=1)),
        Pose(Point3(x=1.978, y=0.300084, z=0.79), Quaternion(x=0, y=0, z=0, w=1)),
        Pose(Point3(x=-2.79475, y=-1.06288, z=0.45), Quaternion(x=0, y=0, z=0, w=1)),
        Pose(Point3(x=-2.62165, y=-2.4946, z=0.4), Quaternion(x=0, y=0, z=0, w=1)),
        Pose(Point3(x=-2.92837, y=-2.63041, z=0.4), Quaternion(x=0, y=0, z=0, w=1)),
        Pose(Point3(x=-0.596321, y=4.99409, z=0.554529), Quaternion(x=0, y=0, z=0, w=1)),
    ]

    main(HSRB)