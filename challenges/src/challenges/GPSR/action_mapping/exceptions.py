from __future__ import annotations

# %% Imports
from dataclasses import dataclass

from krrood.exceptions import DataclassException
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from challenges.GPSR.action_mapping.action_mapping import MessageRole
    from semantic_digital_twin.robots.robot_parts import EndEffector
    from semantic_digital_twin.semantic_annotations.mixins import HasGraspCandidates


# %% Parsing


@dataclass
class MissingMessage(DataclassException):
    """
    Raised when a generated command contains no message of the expected role.
    """

    role: MessageRole
    """
    The role no message was found for.
    """

    def error_message(self) -> str:
        return f"The generated command has no message with the role '{self.role}'."

    def suggest_correction(self) -> str:
        return (
            f"Check that the command generator answered with a '{self.role}' message."
        )


# %% Picking up


@dataclass
class NoGraspFitsGripper(DataclassException):
    """
    Raised when every grasp an object offers is wider than the gripper opens.
    """

    graspable: HasGraspCandidates
    """
    The object none of whose grasps fits.
    """

    gripper: EndEffector
    """
    The gripper that was to take hold of it.
    """

    def error_message(self) -> str:
        return (
            f"No grasp of '{self.graspable.name}' fits {self.gripper.name}, which opens "
            f"only {self.gripper.separation_when_open:.3f} m wide."
        )

    def suggest_correction(self) -> str:
        return "Pick up a narrower object, or use a robot whose gripper opens wider."
