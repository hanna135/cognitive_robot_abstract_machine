from __future__ import annotations

# %% Imports
from dataclasses import dataclass

from krrood.exceptions import DataclassException
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from challenges.GPSR.action_mapping.action_mapping import MessageRole


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
