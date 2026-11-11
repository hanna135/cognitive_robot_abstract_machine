from __future__ import annotations

# %% Imports
from dataclasses import dataclass

from krrood.exceptions import DataclassException
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from challenges.GPSR.action_mapping.action_mapping import ActionType, PayloadKey
    from semantic_digital_twin.robots.robot_parts import EndEffector
    from semantic_digital_twin.semantic_annotations.mixins import (
        HasGraspCandidates,
        HasRootBody,
        HasSupportingSurface,
    )
    from semantic_digital_twin.semantic_annotations.semantic_annotations import Room


# %% Mapping actions


@dataclass
class MissingActionArgument(DataclassException):
    """Raised when a planned action lacks an argument its kind of action
    needs."""

    action: ActionType
    """
    The kind of action that lacks the argument.
    """

    argument: PayloadKey
    """The key of the missing argument."""

    def error_message(self) -> str:
        return f"The '{self.action}' action has no '{self.argument}'."

    def suggest_correction(self) -> str:
        return (
            f"Make the command generator give every '{self.action}' action a "
            f"'{self.argument}' that names something in the world."
        )


# %% Placing


@dataclass
class NoFreeTableInRoom(DataclassException):
    """Raised when no table in a room has space to put an object down on."""

    room: Room
    """
    The room the object was to be put down in.
    """

    placed_object: HasRootBody
    """The object that was to be put down."""

    def error_message(self) -> str:
        return (
            f"No table in '{self.room.name}' has space for '{self.placed_object.name}'."
        )

    def suggest_correction(self) -> str:
        return (
            "Name a placing location with a surface, or clear or add a table in the "
            "room."
        )


@dataclass
class NoFreeSurface(DataclassException):
    """Raised when no surface of the named kind has space to put an object down
    on."""

    surface_type: type[HasSupportingSurface]
    """
    The kind of surface the object was to be put down on.
    """

    placed_object: HasRootBody
    """The object that was to be put down."""

    def error_message(self) -> str:
        return (
            f"No {self.surface_type.__name__} has space for "
            f"'{self.placed_object.name}'."
        )

    def suggest_correction(self) -> str:
        return "Clear a surface of that kind, or name another placing location."


# %% Picking up


@dataclass
class NoGraspFitsGripper(DataclassException):
    """Raised when every grasp an object offers is wider than the gripper
    opens."""

    graspable: HasGraspCandidates
    """
    The object none of whose grasps fits.
    """

    gripper: EndEffector
    """The gripper that was to take hold of it."""

    def error_message(self) -> str:
        return (
            f"No grasp of '{self.graspable.name}' fits {self.gripper.name}, which opens "
            f"only {self.gripper.separation_when_open:.3f} m wide."
        )

    def suggest_correction(self) -> str:
        return "Pick up a narrower object, or use a robot whose gripper opens wider."
