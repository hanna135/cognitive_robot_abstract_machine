from __future__ import annotations

# %% Imports
import json
from dataclasses import dataclass
from enum import StrEnum

from typing_extensions import Any, Self

from challenges.GPSR.action_mapping.exceptions import MissingMessage
from challenges.semdt_helper_functions.find_class_by_name import SemanticAnnotationClassFinder
from coraplex.demos.coraplex_real_tracy.demo import world

from coraplex.locations.locations import ReachabilityLocation
from coraplex.plans.context_extensions import RobotAccess, StatementGrounding
from coraplex.robot_plans.actions.base import Action
from coraplex.robot_plans.actions.composite.searching import SearchAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from cramph.context import StatechartContext
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_types import Pose, Point3
from semantic_digital_twin.world import World


# %% Payload vocabulary


class PayloadKey(StrEnum):
    """
    The keys of a generated command's JSON payload.
    """

    MESSAGES = "messages"
    ROLE = "role"
    CONTENT = "content"
    PLANS = "plans"
    RULE_CHAINS = "rule_chains"
    ACTION = "action"
    TEXT = "text"
    LOCATION = "location"
    OBJECT = "object"
    PERSON = "person"


class MessageRole(StrEnum):
    """
    Who wrote a message of a generated command.
    """

    USER = "user"
    """
    The operator who gave the spoken command.
    """

    ASSISTANT = "assistant"
    """
    The command generator that answered with the plans.
    """


class ActionType(StrEnum):
    """
    The kinds of action a generated plan consists of.
    """

    SAY = "say"
    NAVIGATE = "navigate"
    PICK_UP = "pick_up"
    PLACE = "place"
    FIND = "find"
    OPEN = "open"


# %% Plans


@dataclass
class PlannedAction:
    """
    One step of a generated plan, as the command generator described it.
    """

    action_type: Action
    """
    What the robot does in this step.
    """

    text: str | None = None
    """
    What the robot says, for :attr:`ActionType.SAY`.
    """

    location: str | None = None
    """
    The named place the step goes to, places at or opens.
    """

    object_name: str | None = None
    """
    The name of the object the step picks up or places.
    """

    person: str | None = None
    """
    The description of the person the step looks for.
    """

    world: World | None = None

    @classmethod
    def from_json(cls, data: dict[str, str]) -> Self:
        """
        :param data: One action of a generated plan.
        :return: The action it describes.
        """
        action = None
        semdt_finder = SemanticAnnotationClassFinder()
        location = semdt_finder.find(data.get(PayloadKey.LOCATION))

        #-- LOCATION ---------------------------------------------------------------------------------------------------
        location_coordinates = None
        if location is not None:
            # location_coordinates = find_free_space_in_room(location, world)       # missing on this branch yet
            location_coordinates = Pose(position=Point3(0, 0)) #placeholder

        #-- OBJECT -----------------------------------------------------------------------------------------------------
        searched_object = None
        try:
            if location is not None:
                #searched_object = find_objects_in_room(semdt_finder.find(data.get(PayloadKey.OBJECT)), location, world)[0]     # missing on this branch yet
                pass #placeholder
            else:
                searched_object = world.get_semantic_annotation_by_name(data.get(PayloadKey.OBJECT))
                # location = find_objects_room(searched_object)     # missing on this branch yet

        except Exception as e:
            raise Exception("object seems not to be in the world already, Search Action needs to be implemented to find it.")


        match data[PayloadKey.ACTION]:
            case ActionType.SAY:
                pass # speak action not implemented yet
            case ActionType.NAVIGATE:
                if location_coordinates is None:
                    raise Exception("Can't navigate if no location was given.")
                action = NavigateAction(target_location=location_coordinates)
            case ActionType.PICK_UP:
                robot = world.get_semantic_annotations_by_type(AbstractRobot)[0].robot
                context = StatechartContext.with_extensions(world, [RobotAccess(robot), StatementGrounding()])

                object_pose = Pose(searched_object.root)  # e.g. a Milk from get_semantic_annotations_by_type
                location = ReachabilityLocation(
                    target_pose=object_pose, arm=robot.right_arm, context=context
                )

                pickup_pose = location.ground()  # first sampled standing pose
                action = PickUpAction(grasp=robot._grasp_from(pickup_pose, searched_object), arm=robot.right_arm)
            case ActionType.PLACE:

                # TODO use Costmap
                action = PlaceAction(object_designator=searched_object, target_location=location_coordinates)
            case ActionType.FIND:
                pass
            case ActionType.OPEN:
                pass
            case _:
                action = None

        # TODO change from strings to semantic annotations
        return cls(
            action_type=action,
            text=data.get(PayloadKey.TEXT),
            location=data.get(PayloadKey.LOCATION),
            object_name=data.get(PayloadKey.OBJECT),
            person=data.get(PayloadKey.PERSON),
        )


@dataclass
class GeneratedPlan:
    """
    The actions that carry out one task of a command.
    """

    actions: list[PlannedAction]
    """
    The actions in the order the robot performs them.
    """

    rule_chain: str
    """
    The grammar rules of the command generator the task was built from, joined by ``>``.
    """


@dataclass
class GeneratedCommand:
    """
    A spoken command together with the plans the command generator derived from it.
    """

    instruction: str
    """
    The command as the operator spoke it.
    """

    plans: list[GeneratedPlan]
    """
    One plan for each task of the command, in the order they were given.
    """

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Self:
        """
        :param data: The command generator's JSON payload.
        :return: The command it describes.
        """
        instruction = cls._message_content(data, MessageRole.USER)
        plans = cls._message_content(data, MessageRole.ASSISTANT)[PayloadKey.PLANS]
        return cls(
            instruction=instruction,
            plans=[
                GeneratedPlan(
                    actions=[PlannedAction.from_json(action) for action in plan],
                    rule_chain=rule_chain,
                )
                for plan, rule_chain in zip(
                    plans, data[PayloadKey.RULE_CHAINS], strict=True
                )
            ],
        )

    @staticmethod
    def _message_content(data: dict[str, Any], role: MessageRole) -> Any:
        """
        :param data: The command generator's JSON payload.
        :param role: Whose message to read.
        :return: The content of the first message written by ``role``.
        """
        for message in data[PayloadKey.MESSAGES]:
            if message[PayloadKey.ROLE] == role:
                return message[PayloadKey.CONTENT]
        raise MissingMessage(role=role)


def main():
    json_string = """
{
  "messages": [
    {
      "role": "user",
      "content": "Can you take butter from the storage rack and put it on the laundry room trashcan also please head to the book shelf then find the person pointing to the right and say your team's name also open the cupboard"
    },
    {
      "role": "assistant",
      "content": {
        "plans": [
          [
            { "action": "say", "text": "I will pick up butter from the storage rack and put it on the laundry room trashcan" },
            { "action": "navigate", "location": "storage rack" },
            { "action": "pick_up", "object": "butter" },
            { "action": "navigate", "location": "laundry room trashcan" },
            { "action": "place", "object": "butter", "location": "laundry room trashcan" }
          ],
          [
            { "action": "say", "text": "I will go to the book shelf, find the person pointing to the right and tell them my team's name" },
            { "action": "navigate", "location": "book shelf" },
            { "action": "find", "person": "person pointing to the right" },
            { "action": "say", "text": "<team name>" }
          ],
          [
            { "action": "say", "text": "I will open the cupboard" },
            { "action": "navigate", "location": "cupboard" },
            { "action": "open", "location": "cupboard" }
          ]
        ]
      }
    }
  ],
  "rule_chains": [
    "take_object_from_placement>place_object_on_placement",
    "go_to_destination>find_described_person>tell_fact",
    "open_location"
  ]
}
"""
    command = GeneratedCommand.from_json(json.loads(json_string))
    for plan in command.plans:
        print(plan.rule_chain)
        for action in plan.actions:
            print("  ", action)

if __name__ == "__main__":
    main()