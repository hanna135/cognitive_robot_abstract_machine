import os
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
from semantic_digital_twin.semantic_annotations.mixins import HasSupportingSurface
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.semantic_annotations.semantic_annotations import Bowl, Spoon, Bottle, Cup, ShelfLayer, \
    CounterTop, Table, DishwasherTab, Banana, Bread, Knife, Plate
from semantic_digital_twin.spatial_types import Point3, Quaternion
from semantic_digital_twin.spatial_types.spatial_types import Pose, HomogeneousTransformationMatrix
from semantic_digital_twin.robots.hsrb import HSRB
from pycram.datastructures.dataclasses import Context
from demos.bachelor_thesis.hsrb_setup_world import hsrb_setup_world

import random

from demos.bachelor_thesis.classes_and_methods.helper_classes_and_methods import Environment, \
    timed_plan, timed_parse_stl, debug_task_list_for_demo, print_sorted_task_list, sort_tasks, \
    compare_robot_world_with_real


def main():
    environment = Environment.SuturoApartmentLab
    random_set_of_objects = True

    #------------------ standard setup -------------------------------------------------------------------------------------
    world, dispatcher = hsrb_setup_world(environment=environment)

    dispatcher.known_furniture = world.bodies


    #-----------------------------------------------------------------------------------------------------------------------
    dispatcher.correct_location_tableware_clean = world.get_semantic_annotation_by_name("shelf_2")
    dispatcher.correct_location_tableware_dirty = world.get_semantic_annotation_by_name("counterTop")
    dispatcher.correct_location_food = world.get_semantic_annotation_by_name("table")
    dispatcher.correct_location_drinks = world.get_semantic_annotation_by_name("desk")
    dispatcher.correct_location_all_other_items = world.get_semantic_annotation_by_name("shelf_1")

    dispatcher.dining_table = world.get_semantic_annotation_by_name("dining_table")


    #-----------------------------------------------------------------------------------------------------------------------

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

        # world.add_semantic_annotations(
        #     [
        #         Bowl(root=world.get_body_by_name("bowl.stl"), name=PrefixedName("bowl.stl")),
        #         Spoon(root=world.get_body_by_name("spoon.stl"), name=PrefixedName("spoon.stl")),
        #         Bottle(root=world.get_body_by_name("Static_MilkPitcher.stl"), name=PrefixedName("Static_MilkPitcher.stl")),
        #         Bottle(root=world.get_body_by_name("Static_CokeBottle.stl"), name=PrefixedName("Static_CokeBottle.stl")),
        #         Cup(root=world.get_body_by_name("jeroen_cup.stl"), name=PrefixedName("jeroen_cup.stl")),
        #         DishwasherTab(root=world.get_body_by_name("dishwasher_tab.stl"), name=PrefixedName("dishwasher_tab.stl")),
        #         Banana(root=world.get_body_by_name("banana.stl"), name=PrefixedName("banana.stl")),
        #         Bread(root=world.get_body_by_name("bread.stl"), name=PrefixedName("bread.stl")),
        #         Knife(root=world.get_body_by_name("knife.stl"), name=PrefixedName("knife.stl")),
        #         Plate(root=world.get_body_by_name("plate.stl"), name=PrefixedName("plate.stl")),
        #     ]
        # )

        supporting_surfaces = []
        supporting_surfaces.append(world.get_semantic_annotation_by_name("shelf_1"))
        supporting_surfaces.append(world.get_semantic_annotation_by_name("shelf_2"))
        supporting_surfaces.append(world.get_semantic_annotation_by_name("counterTop"))
        supporting_surfaces.append(world.get_semantic_annotation_by_name("table"))
        supporting_surfaces.append(world.get_semantic_annotation_by_name("lowerTable"))
        supporting_surfaces.append(world.get_semantic_annotation_by_name("desk"))
        supporting_surfaces.append(world.get_semantic_annotation_by_name("cooking_table"))
        supporting_surfaces.append(world.get_semantic_annotation_by_name("dining_table"))
        supporting_surfaces.append(world.get_semantic_annotation_by_name("dishwasher_rack"))




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

    # get coordinates from publish point in rviz2
    plan_labels = [
        "park left arm",
        "navigate dishwasher",
        "navigate kitchen counter 1",
        "navigate kitchen counter 2",
        "navigate high kitchen counter 1",
        "navigate high kitchen counter 2",
        "navigate transition before counter",
        "navigate pc desk",
        "navigate popcorn table",
        "navigate transition before wall",
        "navigate sofa table",
        "navigate living room table",
        "navigate shelf",
        "move high",
        "park arms high",
    ]

    plan_driving = [
            timed_plan("park left arm", ParkArmsAction(Arms.LEFT), context),

            # dishwasher
            timed_plan("navigate dishwasher", NavigateAction(
                target_location=Pose(Point3(1.2266756, -0.2182769775390625, 0.0), orientation=(Quaternion(z=-0.316469, w=0.948602)),
                                    reference_frame=world.root), keep_joint_states=True), context),

            # kitchen counter
            timed_plan("navigate kitchen counter 1", NavigateAction(
                target_location=Pose(Point3(1.677089, -0.91819, 0), orientation=(Quaternion(z=-0.673775, w=0.7389362)),
                                    reference_frame=world.root), keep_joint_states=True), context),
            timed_plan("navigate kitchen counter 2", NavigateAction(
                target_location=Pose(Point3(3.099597, -0.897218, 0), orientation=(Quaternion(z=-0.679143, w=0.734005727)),
                                     reference_frame=world.root), keep_joint_states=True), context),

            # high kitchen counter
            timed_plan("navigate high kitchen counter 1", NavigateAction(
                target_location=Pose(Point3(3.393497, -0.3331599, 0), orientation=(Quaternion(z=0.748984068, w=0.6625880)),
                                     reference_frame=world.root), keep_joint_states=True), context),
            timed_plan("navigate high kitchen counter 2", NavigateAction(
                target_location=Pose(Point3(4.839765, -0.061004, 0), orientation=(Quaternion(z=0.7564081, w=0.654099959)),
                                     reference_frame=world.root), keep_joint_states=True), context),

            # transition point to not drive through counter
            timed_plan("navigate transition before counter", NavigateAction(
                target_location=Pose(Point3(1.777967, -0.090250, 0), orientation=(Quaternion(z=0.900024, w=0.435840186)),
                                     reference_frame=world.root), keep_joint_states=True), context),

            # pc desk
            timed_plan("navigate pc desk", NavigateAction(
                target_location=Pose(Point3(1.0679969, 1.530962, 0), orientation=(Quaternion(z=-0.9981287, w=0.0611478)),
                                                    reference_frame=world.root), keep_joint_states=True), context),

            # popcorn table
            timed_plan("navigate popcorn table", NavigateAction(
                target_location=Pose(Point3(1.09943246, 5.53489685, 0), orientation=(Quaternion(z=0.7474030, w=0.6643709)),
                                                    reference_frame=world.root), keep_joint_states=True), context),

            # transition point to not drive in wall
            timed_plan("navigate transition before wall", NavigateAction(
                target_location=Pose(Point3(1.7850532, 3.3190565, 0), orientation=(Quaternion(z=0.1006678, w=0.99492009)),
                                                    reference_frame=world.root), keep_joint_states=True), context),

            # sofa table
            timed_plan("navigate sofa table", NavigateAction(
                target_location=Pose(Point3(3.57369399, 3.0707988, 0), orientation=(Quaternion(z=0.0701156, w=0.99753887)),
                                     reference_frame=world.root), keep_joint_states=True), context),

            # living room table
            timed_plan("navigate living room table", NavigateAction(
                target_location=Pose(Point3(3.2095706, 6.522722, 0), orientation=(Quaternion(z=-0.9995140, w=0.03117147)),
                                     reference_frame=world.root), keep_joint_states=True), context),

            # shelf
            timed_plan("navigate shelf", NavigateAction(
                target_location=Pose(Point3(3.3593473, 5.40832, 0), orientation=(Quaternion(z=0.0721225, w=0.997395779)),
                                     reference_frame=world.root), keep_joint_states=True), context),

            timed_plan("move high", MoveTorsoAction(TorsoState.HIGH), context),

            timed_plan("park arms high", ParkArmsWithHighTorsoAction(Arms.LEFT), context),

            ]




    with simulated_robot:
        for index, (label, plan) in enumerate(zip(plan_labels, plan_driving), start=1):
            step_label = f"{index:02d}/{len(plan_driving)} {label}"
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

    return res, dispatcher.perceived_objects, world.bodies



if __name__ == "__main__":
    main()
