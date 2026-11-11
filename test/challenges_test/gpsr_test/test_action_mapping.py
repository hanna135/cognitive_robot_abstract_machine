from __future__ import annotations

# %% Imports
import json
from pathlib import Path

import pytest
from typing_extensions import Any

from challenges.GPSR.action_mapping.action_mapping import (
    ActionType,
    GeneratedCommand,
    PayloadKey,
    PlannedAction,
)


# %% Fixtures


@pytest.fixture
def payload() -> dict[str, Any]:
    """
    A command with three tasks, as the command generator answers it.
    """
    return json.loads(
        (Path(__file__).parent / "resources" / "generated_command.json").read_text()
    )


# %% Parsing a generated command


def test_actions_keep_their_order_and_fields(payload: dict[str, Any]) -> None:
    command = GeneratedCommand.from_json(payload)
    assert command.plans[0].actions == [
        PlannedAction(
            action_type=ActionType.SAY,
            text="I will pick up butter from the storage rack and put it on the laundry room trashcan",
        ),
        PlannedAction(action_type=ActionType.NAVIGATE, location="storage rack"),
        PlannedAction(action_type=ActionType.PICK_UP, object_name="butter"),
        PlannedAction(
            action_type=ActionType.NAVIGATE, location="laundry room trashcan"
        ),
        PlannedAction(
            action_type=ActionType.PLACE,
            object_name="butter",
            location="laundry room trashcan",
        ),
    ]


def test_unknown_action_type_is_rejected() -> None:
    with pytest.raises(ValueError):
        PlannedAction.from_json({PayloadKey.ACTION: "dance"})
