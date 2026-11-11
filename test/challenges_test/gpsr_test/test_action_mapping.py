from __future__ import annotations

# %% Imports
import json
from pathlib import Path

import pytest
from typing_extensions import Any

from challenges.GPSR.action_mapping.action_mapping import (
    ActionType,
    GeneratedCommand,
    GeneratedPlan,
    PayloadKey,
    PlannedAction,
)
from coraplex.robot_plans.actions.core.talking import TalkingAction
from coraplex.robot_plans.transformation_by_room import RoomVisitPlanner
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Floor,
    Kitchen,
    LivingRoom,
    Room,
)
from semantic_digital_twin.world_description.world_entity import Body


# %% Fixtures


@pytest.fixture
def payload() -> dict[str, Any]:
    """
    A command with three tasks, as the command generator answers it.
    """
    return json.loads(
        (Path(__file__).parent / "resources" / "generated_command.json").read_text()
    )


@pytest.fixture
def kitchen() -> Kitchen:
    """
    A kitchen outside of any world, as the location a step names.
    """
    return Kitchen(
        floor=Floor(root=Body(name=PrefixedName("kitchen_floor"))),
        name=PrefixedName("kitchen"),
    )


@pytest.fixture
def living_room() -> LivingRoom:
    """
    A living room outside of any world, as the location a step names.
    """
    return LivingRoom(
        floor=Floor(root=Body(name=PrefixedName("living_room_floor"))),
        name=PrefixedName("living_room"),
    )


@pytest.fixture
def any_room() -> set[type[Room]]:
    """
    The kinds of room a step that names no room may be performed in.
    """
    return {Kitchen, LivingRoom}


def saying(text: str, location: Room | None = None) -> PlannedAction:
    """
    :param text: What the robot says.
    :param location: The room the step names, if any.
    :return: A step that says ``text``, performed in ``location``.
    """
    return PlannedAction(
        action_type=TalkingAction(speaking_text=text), text=text, location=location
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


# %% Rewrapping a plan into a command for the room transformation


def test_leading_step_without_room_joins_the_first_room(
    kitchen: Kitchen, any_room: set[type[Room]]
) -> None:
    plan = GeneratedPlan(
        actions=[saying("announce"), saying("go", kitchen), saying("take")]
    )
    [step] = plan.rewrap_into_command("bring it", any_room).steps
    assert step.rooms == {Kitchen}
    assert step.actions == [action.action_type for action in plan.actions]


def test_change_of_room_starts_a_new_step(
    kitchen: Kitchen, living_room: LivingRoom, any_room: set[type[Room]]
) -> None:
    plan = GeneratedPlan(
        actions=[
            saying("go", kitchen),
            saying("take"),
            saying("go on", living_room),
            saying("put"),
        ]
    )
    steps = plan.rewrap_into_command("bring it", any_room).steps
    assert [step.rooms for step in steps] == [{Kitchen}, {LivingRoom}]
    assert [step.actions for step in steps] == [
        [plan.actions[0].action_type, plan.actions[1].action_type],
        [plan.actions[2].action_type, plan.actions[3].action_type],
    ]


def test_plan_naming_no_room_may_run_in_any_room(any_room: set[type[Room]]) -> None:
    plan = GeneratedPlan(actions=[saying("hello")])
    [step] = plan.rewrap_into_command("greet", any_room).steps
    assert step.rooms == any_room


def test_rewrapped_plans_share_a_visit_to_the_same_room(
    kitchen: Kitchen, living_room: LivingRoom, any_room: set[type[Room]]
) -> None:
    transport = GeneratedPlan(
        actions=[saying("take", kitchen), saying("put", living_room)]
    )
    question = GeneratedPlan(actions=[saying("ask", kitchen)])
    commands = [
        plan.rewrap_into_command("tasks", any_room) for plan in (transport, question)
    ]
    room_visits = RoomVisitPlanner().transform_plan_by_rooms(commands)
    assert [room_visit.room for room_visit in room_visits] == [Kitchen, LivingRoom]
    assert room_visits[0].actions == [
        transport.actions[0].action_type,
        question.actions[0].action_type,
    ]
