from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field
from enum import StrEnum

from challenges.GPSR.action_mapping.action_mapping import main
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.mixins import (
    HasRootBody,
    HasSupportingSurface,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cupboard,
    Food,
    Human,
    Shelf,
    TrashCan,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world_description.world_entity import Body

# %% Scene vocabulary


class MockObjectName(StrEnum):
    """
    The names of the objects in the mock world, as the generated command refers to them.
    """

    STORAGE_RACK = "storage rack"
    LAUNDRY_ROOM_TRASHCAN = "laundry room trashcan"
    BOOK_SHELF = "book shelf"
    CUPBOARD = "cupboard"
    BUTTER = "butter"
    PERSON_POINTING_TO_THE_RIGHT = "person pointing to the right"


# %% Classes of the named locations and objects


@dataclass(eq=False)
class StorageRack(Shelf):
    """
    A rack things are stored on.
    """


@dataclass(eq=False)
class BookShelf(Shelf):
    """
    A shelf books are kept on.
    """


@dataclass(eq=False)
class LaundryRoomTrashcan(TrashCan, HasSupportingSurface):
    """
    The trashcan of the laundry room, whose closed lid things can be put on.
    """


@dataclass(eq=False)
class Butter(Food):
    """
    A block of butter.
    """


# %% Scene objects


@dataclass
class MockObject:
    """
    One object of the mock world: what it is, what it is called and where it stands.
    """

    semantic_annotation_type: type[HasRootBody]
    """
    What the object is.
    """

    object_name: MockObjectName
    """
    The name the object is annotated with, so the generated command can refer to it.
    """

    world_root_T_object: HomogeneousTransformationMatrix
    """
    Where the object stands, in the frame of the world root.
    """

    scale: Scale | None = None
    """
    The size of the object's box; ``None`` keeps the default size of its type.
    """

    def spawn(self, world: World) -> HasRootBody:
        """
        :param world: The world to put the object into.
        :return: The annotation of the new object.
        """
        with world.modify_world():
            return self.semantic_annotation_type.create_with_new_body_in_world(
                name=self.object_name,
                world=world,
                world_root_T_self=self.world_root_T_object,
                scale=self.scale,
            )


# %% Mock world


@dataclass
class MockWorld:
    """
    A robot among the objects the generated command of :func:`main` names, without ROS
    or a scene file.
    """

    robot_type: type[AbstractRobot] = PR2
    """
    The robot that carries out the command.
    """

    objects: list[MockObject] = field(
        default_factory=lambda: [
            MockObject(
                StorageRack,
                MockObjectName.STORAGE_RACK,
                HomogeneousTransformationMatrix.from_xyz_rpy(2.0, 0.0, 0.5),
                Scale(0.4, 0.8, 1.0),
            ),
            MockObject(
                Butter,
                MockObjectName.BUTTER,
                HomogeneousTransformationMatrix.from_xyz_rpy(2.0, 0.0, 1.025),
                Scale(0.1, 0.05, 0.05),
            ),
            MockObject(
                LaundryRoomTrashcan,
                MockObjectName.LAUNDRY_ROOM_TRASHCAN,
                HomogeneousTransformationMatrix.from_xyz_rpy(0.0, 3.0, 0.35),
                Scale(0.5, 0.5, 0.7),
            ),
            MockObject(
                BookShelf,
                MockObjectName.BOOK_SHELF,
                HomogeneousTransformationMatrix.from_xyz_rpy(-2.0, 0.0, 0.9),
                Scale(0.4, 1.0, 1.8),
            ),
            MockObject(
                Human,
                MockObjectName.PERSON_POINTING_TO_THE_RIGHT,
                HomogeneousTransformationMatrix.from_xyz_rpy(-1.5, 1.0, 0.9),
                Scale(0.3, 0.5, 1.8),
            ),
            MockObject(
                Cupboard,
                MockObjectName.CUPBOARD,
                HomogeneousTransformationMatrix.from_xyz_rpy(0.0, -3.0, 0.5),
                Scale(0.5, 0.8, 1.0),
            ),
        ]
    )
    """
    The objects around the robot, which starts at the origin of the world.
    """

    def build(self) -> World:
        """
        :return: A new world holding the robot and :attr:`objects`.
        """
        world = World()
        with world.modify_world():
            world.add_body(Body(name=PrefixedName("map")))
        RobotSpecification(self.robot_type).spawn(world)
        for mock_object in self.objects:
            mock_object.spawn(world)
        return world


if __name__ == "__main__":
    main(MockWorld().build())
