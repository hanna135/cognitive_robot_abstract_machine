"""The kitchen the generated command of :func:`main` is mapped to actions
in."""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field, replace
from datetime import timedelta
from enum import StrEnum
from pathlib import Path

import coraplex
import mujoco
from challenges.GPSR.action_mapping.action_mapping import main as action_main
from coraplex.demonstrations import RobotDemonstration
from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.executors import PhysicallySimulatedPlanExecutor, PlanExecutor
from cramph.composites import Sequence
from cramph.context import ContextExtension, StatechartContext
from cramph.statechart import Statechart
from krrood.patterns.path_enumeration import PathEnumeration
from semantic_digital_twin.adapters.multi_sim import MujocoSim, MujocoSynchronizer
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.predefined_maps.kitchen_environment import KitchenEnvironment
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bedroom,
    Human,
    Kitchen,
    LivingRoom,
    Milk,
    Room,
)
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.specifications.connections import (
    Connection6DoFSpecification,
)
from semantic_digital_twin.specifications.kinematic_structure_entities import (
    BodySpecification,
)
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.semantic_annotations import (
    SemanticAnnotationWithRootSpecification,
)
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.contact import ContactParameters
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% Scene vocabulary


class SceneName(StrEnum):
    """The names of the scene's parts, as the kitchen and the generated command
    call them."""

    COUNTER_TOP = "counter_top"
    KITCHEN = "kitchen"
    LIVING_ROOM = "living_room"
    BED_ROOM = "bed_room"
    MILK = "milk"
    PERSON = "person"


class SceneFile(PathEnumeration):
    """The files the added objects are built from, under the coraplex
    resources."""

    MILK = (
        Path(coraplex.__file__).resolve().parents[2]
        / "resources"
        / "objects"
        / "milk.stl"
    )
    """The mesh of the milk carton."""

    PERSON = (
        Path(coraplex.__file__).resolve().parents[2]
        / "resources"
        / "objects"
        / "box.urdf"
    )
    """A box standing in for the person's body."""

# %% Classes of the named objects


@dataclass(eq=False)
class Person(Human):
    """A person, named so that the generated command's ``"person"`` finds this
    class."""


# %% Simulation


class OncePerCallSyncedMujocoSim(MujocoSim):
    """
    A MuJoCo simulation that copies its state into the world once per call of
    :meth:`step_simulation`.
    """

    def step_simulation(self, duration: timedelta) -> None:
        """Differs from :meth:`MujocoSim.step_simulation` in copying the
        physics back into the world only after the last physics step of the
        call."""
        last_step = timedelta(seconds=self.simulator.step_size)
        self.synchronizer.sync_rate_hz = 0
        super().step_simulation(duration - last_step)
        self.synchronizer.sync_rate_hz = MujocoSynchronizer.UNTHROTTLED_SYNC_RATE_HZ
        super().step_simulation(last_step)


# %% Demonstration


@dataclass
class GeneratedCommandDemonstration(RobotDemonstration):
    """
    The robot carries out the actions mapped from the generated command of
    :func:`~challenges.GPSR.action_mapping.action_mapping.main`, in the kitchen of
    :class:`~semantic_digital_twin.predefined_maps.kitchen_environment.KitchenEnvironment`
    with rooms of their kind, a milk carton on its counter top and a person in its
    kitchen room.
    """

    used_robot: type[AbstractRobot] = HSRB

    ros_node_name: str = "gpsr_generated_command_demo"

    executor_type: type[PlanExecutor] = PhysicallySimulatedPlanExecutor

    collision_avoidance: bool = False

    scene_joints_movable: bool = False

    room_types: dict[SceneName, type[Room]] = field(
        default_factory=lambda: {
            SceneName.KITCHEN: Kitchen,
            SceneName.LIVING_ROOM: LivingRoom,
            SceneName.BED_ROOM: Bedroom,
        }
    )
    """The kind of each of the kitchen's rooms, by the room's name."""

    friction_cone: mujoco.mjtCone = mujoco.mjtCone.mjCONE_ELLIPTIC
    """The friction cone the contacts use; the elliptic cone keeps a held
    carton from creeping between the fingers."""

    impedance_ratio: float = 10.0
    """How much harder sliding is than pressing in, as MuJoCo recommends for
    grasping with an elliptic cone."""

    no_slip_iterations: int = 10
    """Passes spent removing the slip soft contacts leave, so a held carton
    stays where the fingers hold it."""

    def scene_specification(self) -> WorldSpecification:
        """Never called: :meth:`build_scene` builds the kitchen directly, since
        :class:`~semantic_digital_twin.predefined_maps.kitchen_environment.Kitc
        henEnvironment` builds a :class:`~semantic_digital_twin.world.World`,
        not a :class:`WorldSpecification`."""
        raise NotImplementedError

    def build_scene(self) -> World:
        """Differs from :meth:`RobotDemonstration.build_scene` in starting from
        the kitchen instead of :meth:`scene_specification`."""
        scene = KitchenEnvironment().create_world()
        self.populate_scene(scene)
        if not self.scene_joints_movable:
            self.replace_joints_with_fixed_connections(scene)
        return scene

    def robot_specification(self) -> RobotSpecification:
        return RobotSpecification(self.used_robot)

    def build_simulation(self, world: World) -> MujocoSim:
        """Differs from :meth:`RobotDemonstration.build_simulation` in contacts
        that let the gripper hold a carton, and in copying the physics into the
        world once per tick, see :class:`OncePerCallSyncedMujocoSim`."""
        return OncePerCallSyncedMujocoSim(
            world=world,
            headless=self.headless,
            cone=self.friction_cone,
            impratio=self.impedance_ratio,
            noslip_iterations=self.no_slip_iterations,
        )

    def is_scene_populated(self, world: World) -> bool:
        return world.is_kinematic_structure_entity_in_world_by_name(SceneName.MILK)

    def populate_scene(self, world: World) -> None:
        """Give the kitchen's rooms their kinds, stand the milk on the counter
        top and the person in the kitchen room."""
        self.assign_room_types(world)
        self.place_milk(world)
        self.place_person(world)

    def build_context_extensions(self, world: World) -> list[ContextExtension]:
        return [RobotAccess(world.get_semantic_annotations_by_type(self.used_robot)[0])]

    def build_statechart(self, context: StatechartContext) -> Statechart:
        """Perform the actions of every plan of the generated command, one
        after another."""
        command = action_main(context)
        statechart = Statechart(context=context)
        statechart.add_node(
            Sequence(
                nodes=[
                    planned_action.action_type
                    for plan in command.plans
                    for planned_action in plan.actions
                ]
            )
        )
        return statechart

    def assign_room_types(self, world: World) -> None:
        """Replace each of the kitchen's untyped rooms named in
        :attr:`room_types` by a room of its kind on the same floor.

        :param world: The kitchen.
        """
        with world.modify_world():
            for room_name, room_type in self.room_types.items():
                room = world.get_semantic_annotation_by_name(room_name)
                world.remove_semantic_annotation(room)
                world.add_semantic_annotation(
                    room_type(floor=room.floor, name=room.name)
                )

    def place_milk(self, world: World) -> Milk:
        """
        :param world: The kitchen.
        :return: A new milk carton standing on the counter top as close to its front edge as it
            can without overhanging, free to be picked up.
        """
        milk = SemanticAnnotationWithRootSpecification(
            name=SceneName.MILK,
            semantic_annotation_type=Milk,
            root_specification=BodySpecification.mesh(
                SceneName.MILK,
                str(SceneFile.MILK),
                connection_specification=Connection6DoFSpecification(),
            ),
        ).spawn(world)
        ContactParameters.create_for_grasped_object().apply_to([milk.root])
        counter_top = world.get_body_by_name(SceneName.COUNTER_TOP)
        self.stand_on(milk.root, counter_top, world)
        self.slide_to_front_edge(milk.root, counter_top, world)
        return milk

    def place_person(self, world: World) -> Person:
        """
        :param world: The kitchen.
        :return: A new red person fixed in the middle of the kitchen room, so it stays
            in place although the floors have no collision shapes. It has no collision
            shape of its own, so the robot drives through it.
        """
        person_world = URDFParser.from_file(
            file_path=str(SceneFile.PERSON)
        ).create_world()
        person_body = person_world.root
        person_body.visual = ShapeCollection(
            shapes=[
                replace(shape, color=Color.RED())
                for shape in person_body.collision.shapes
            ],
            reference_frame=person_body,
        )
        kitchen = world.get_semantic_annotation_by_name(SceneName.KITCHEN)
        world.merge_world(
            person_world,
            root_connection=FixedConnection(
                parent=world.root,
                child=person_body,
                parent_T_connection_expression=self.centred_standing_pose(
                    person_body, kitchen.floor.root, world
                ),
            ),
        )
        person_body.collision = ShapeCollection(reference_frame=person_body)
        person = Person(root=person_body, name=PrefixedName(SceneName.PERSON))
        with world.modify_world():
            world.add_semantic_annotation(person)
        return person

    def stand_on(self, body: Body, surface: Body, world: World) -> None:
        """Move ``body`` to :meth:`centred_standing_pose` on ``surface``.

        :param body: The body to move, attached to the world by a 6DoF
            connection.
        :param surface: The body to stand it on.
        :param world: The world both are in.
        """
        body.parent_connection.origin = self.centred_standing_pose(body, surface, world)

    @staticmethod
    def centred_standing_pose(
        body: Body, surface: Body, world: World
    ) -> HomogeneousTransformationMatrix:
        """
        :param body: The body to stand, in this world or still in a world of its own.
        :param surface: The body to stand it on, in ``world``.
        :param world: The world ``surface`` is in.
        :return: Where ``body`` stands upright in the middle of ``surface``, in the frame
            of the root of ``world``.
        """
        surface_box = surface.collision.as_bounding_box_collection_in_frame(
            world.root
        ).bounding_box()
        body_box = body.collision.as_bounding_box_collection_in_frame(
            body
        ).bounding_box()
        return HomogeneousTransformationMatrix.from_xyz_rpy(
            x=(surface_box.min_x + surface_box.max_x) / 2,
            y=(surface_box.min_y + surface_box.max_y) / 2,
            z=surface_box.max_z - body_box.min_z,
            reference_frame=world.root,
        )

    @staticmethod
    def slide_to_front_edge(body: Body, surface: Body, world: World) -> None:
        """Slide ``body``, standing on ``surface``, along the y-axis until its
        side is flush with the edge of ``surface`` facing the origin of the
        world, where the robot starts.

        :param body: The body to slide, attached to the world by a 6DoF
            connection.
        :param surface: The body it stands on.
        :param world: The world both are in.
        """
        surface_box = surface.collision.as_bounding_box_collection_in_frame(
            world.root
        ).bounding_box()
        body_box = body.collision.as_bounding_box_collection_in_frame(
            body
        ).bounding_box()
        world_root_T_body = body.parent_connection.origin
        if surface_box.min_y > 0:
            world_root_T_body.y = surface_box.min_y - body_box.min_y
        else:
            world_root_T_body.y = surface_box.max_y - body_box.max_y
        body.parent_connection.origin = world_root_T_body


def main(headless: bool = False) -> World:
    """Run the demonstration.

    :param headless: Whether the physics runs without a viewer.
    :return: The world the command was carried out in.
    """
    return GeneratedCommandDemonstration(
        scene_joints_movable=False, headless=headless
    ).run()


if __name__ == "__main__":
    main()
