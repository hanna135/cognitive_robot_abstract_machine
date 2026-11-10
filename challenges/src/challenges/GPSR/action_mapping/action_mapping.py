from __future__ import annotations

# %% Imports
import json
from dataclasses import dataclass
from enum import StrEnum

from typing_extensions import Any, Self

from challenges.GPSR.action_mapping.exceptions import (
    MissingMessage,
    NoGraspFitsGripper,
)
from challenges.GPSR.action_mapping.mock_world import Office
from challenges.semdt_helper_functions.find_class_by_name import SemanticAnnotationClassFinder


from coraplex.locations.locations import ReachabilityLocation
from coraplex.plans.context_extensions import RobotAccess, StatementGrounding
from coraplex.querying.grasp_closeness import GraspsByCloseness
from coraplex.robot_plans.actions.base import Action
from coraplex.robot_plans.actions.composite.transporting import MoveAndPickUpAction
from coraplex.robot_plans.actions.composite.searching import SearchAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.talking import TalkingAction
from coraplex.robot_plans.task_definitions import Command, Step
from cramph.context import StatechartContext
from krrood.entity_query_language.query.match import Match
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.mixins import HasGraspCandidates, HasRootBody
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Room,
    Kitchen,
    LivingRoom,
    Bedroom,
)
from semantic_digital_twin.spatial_types import Pose, Point3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation


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

    action_type: Action | Match | None
    """
    What the robot does in this step, as an action or as a statement grounded when the
    step starts; ``None`` for a kind of step that is not mapped yet.
    """

    text: str | None = None
    """
    What the robot says, for :attr:`ActionType.SAY`.
    """

    location: SemanticAnnotation | None = None
    """
    The named place the step goes to, places at or opens.
    """

    object_annotation: str | None = None
    """
    The name of the object the step picks up or places.
    """

    person: str | None = None
    """
    The description of the person the step looks for.
    """

    @classmethod
    def from_json(cls, data: dict[str, str], context: StatechartContext) -> Self:
        """
        :param data: One action of a generated plan.
        :param context: The context of the plan the action runs in, whose robot performs
            it in its world.
        :return: The action it describes.
        """
        world = context.world
        action = None
        semdt_finder = SemanticAnnotationClassFinder()
        if data.get(PayloadKey.LOCATION) is not None:
            semdt_type = semdt_finder.find(data.get(PayloadKey.LOCATION))
            print(semdt_type)
            location = world.get_semantic_annotations_by_type(semdt_type)[0]
        else:
            location = None

        #-- LOCATION ---------------------------------------------------------------------------------------------------
        location_coordinates = None
        if location is not None:
            if isinstance(location, Room):
                # location_coordinates = find_free_space_in_room(location, world)       # missing on this branch yet
                location_coordinates = Pose.from_xyz_rpy(x=1, y=1, reference_frame=world.root)  # placeholder
                print("ROOM")
            else:
                #location_coordinates = location.root.global_pose # use rodys query to find place near
                location_coordinates = Pose.from_xyz_rpy(
                    x=1, y=1, reference_frame=world.root
                )
                print("NOT ROOM", type[location])

        #-- OBJECT -----------------------------------------------------------------------------------------------------
        searched_object = None
        try:
            if location is not None and data.get(PayloadKey.OBJECT) is not None:
                #searched_object = find_objects_in_room(semdt_finder.find(data.get(PayloadKey.OBJECT)), location, world)[0]     # missing on this branch yet
                searched_object = world.get_semantic_annotation_by_name(
                    data.get(PayloadKey.OBJECT)
                ) #placeholder
            elif data.get(PayloadKey.OBJECT) is not None:
                searched_object = world.get_semantic_annotation_by_name(data.get(PayloadKey.OBJECT))
                # location = find_objects_room(searched_object)     # missing on this branch yet

        except Exception as e:
            raise Exception("object seems not to be in the world already, Search Action needs to be implemented to find it.")

        match data[PayloadKey.ACTION]:
            case ActionType.SAY:
                action = TalkingAction(speaking_text=data.get(PayloadKey.TEXT))
            case ActionType.NAVIGATE: #DONE
                if location_coordinates is None:
                    raise Exception("Can't navigate if no location was given.")
                action = NavigateAction(target_location=location_coordinates)
            case ActionType.PICK_UP:
                action = cls.pick_up_from_reachable_standing_pose(
                    searched_object, context
                )
            case ActionType.PLACE:
                pass
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
            location=location,
            object_annotation=searched_object,
            person=data.get(PayloadKey.PERSON),
        )

    @staticmethod
    def pick_up_from_reachable_standing_pose(
        graspable: HasGraspCandidates, context: StatechartContext
    ) -> MoveAndPickUpAction:
        """
        :param graspable: The object to pick up.
        :param context: The context of the plan, whose robot picks the object up with
            its first arm.
        :return: A pick-up that drives to the first standing pose the reachability
            costmap offers, and takes the object by the grasp closest to the robot
            standing there that the gripper opens wide enough for.
        :raises NoGraspFitsGripper: If every grasp is wider than the gripper opens.
        """
        arm = context.require_extension(RobotAccess).robot.all_arms[0]
        standing_pose = ReachabilityLocation(
            target_pose=Pose(reference_frame=graspable.root), arm=arm, context=context
        ).ground()
        grasps_closest_first = GraspsByCloseness(
            graspable, standing_pose, number_of_grasps=None
        )()
        fitting_grasp = next(
            (
                grasp
                for grasp in grasps_closest_first
                if grasp.width_along_closing_axis
                <= arm.end_effector.separation_when_open
            ),
            None,
        )
        if fitting_grasp is None:
            raise NoGraspFitsGripper(graspable=graspable, gripper=arm.end_effector)
        return MoveAndPickUpAction.from_standing_position(
            standing_pose, fitting_grasp, arm
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
    def from_json(cls, data: dict[str, Any], context: StatechartContext) -> Self:
        """
        :param data: The command generator's JSON payload.
        :param context: The context of the plan the command runs in.
        :return: The command it describes.
        """
        instruction = cls._message_content(data, MessageRole.USER)
        plans = cls._message_content(data, MessageRole.ASSISTANT)[PayloadKey.PLANS]

        return cls(
            instruction=instruction,
            plans=[
                GeneratedPlan(
                    actions=[
                        PlannedAction.from_json(action, context) for action in plan
                    ],
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

    def rewrap_into_command(self) -> Command:
        location_iteration = None
        current_location = None
        first_action = True

        room_list = []
        i = 0
        action_list_sorted = [[]] # sorted by room
        for plan in self.plans:

            for action in plan.actions:
                if first_action:
                    current_location = action.location
                    location_iteration = action.location
                    room_list.append(action.location)
                if isinstance(action.location, Room):
                    current_location = action.location
                elif isinstance(action.location, HasRootBody):
                    # current_location = find_objects_room(action.location) # TODO: query currently missing on this branch
                    pass

                if current_location != location_iteration:
                    room_list.append(current_location)
                    location_iteration = current_location
                    action_list_sorted.append([action.action_type])
                    i += 1
                else:
                    action_list_sorted[i].append(action.action_type)

        steps = []
        j = 0
        for step in action_list_sorted:
            if room_list[j] is None:
                steps.append(Step(rooms={Kitchen, LivingRoom, Office, Bedroom}, actions=step))
            else:
                steps.append(Step(room_list[j], actions=step))

        return Command(instruction=self.instruction, steps=steps)





def main(context: StatechartContext) -> GeneratedCommand:
    json_ms0_mock = """

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
            { "action": "navigate", "location": "kitchen" },
            { "action": "pick_up", "object": "milk" },
            { "action": "navigate", "location": "living room" }
          ],
          [
            { "action": "say", "text": "I will go to the book shelf, find the person pointing to the right and tell them my team's name" },
            { "action": "navigate", "location": "kitchen" },
            { "action": "say", "text": "<team name>" }
          ]
        ]
      }
    }
  ],
  "rule_chains": [
    "take_object_from_placement>place_object_on_placement",
    "go_to_destination>find_described_person>tell_fact"
  ]
}
    """
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
    command = GeneratedCommand.from_json(json.loads(json_ms0_mock), context)
    for plan in command.plans:
        print(plan.rule_chain)
        for action in plan.actions:
            print(
                "  ",
                type(action.action_type).__name__,
                "| location:",
                None if action.location is None else type(action.location).__name__,
                "| object:",
                None
                if action.object_annotation is None
                else type(action.object_annotation).__name__,
                "| person:",
                action.person,
            )
            print(action.action_type)
    return command


if __name__ == "__main__":
    main()