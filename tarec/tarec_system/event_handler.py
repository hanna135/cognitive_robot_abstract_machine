# source for base: https://medium.com/@idelossantosruiz/events-in-python-e2b3cb76ac2d
from tarec.tarec_system.system_defined_values import TaRecValues
from tarec.tarec_system.tasks import SetTableTask, CleanTableTask, PutAwayObjectTask, \
    LoadDishwasherTask, UnloadDishwasherTask
from pycram.datastructures.dataclasses import Context
from semantic_digital_twin.exceptions import WorldEntityNotFoundError
from semantic_digital_twin.semantic_annotations.mixins import HasSupportingSurface
from semantic_digital_twin.semantic_annotations.semantic_annotations import Bowl, Cuttlery, Plate, Cup, Tableware
from semantic_digital_twin.spatial_types import Point3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body, SemanticAnnotation
from tarec.cram_interfaces.predicate_mock import (
    reachable,
    misplaced,
    human_near,
    is_supported_by_surface_cached, is_empty,
)
import datetime
from time import time as tm

from tarec.tarec_system.world_state import WorldState


class EventDispatcher:
    def __init__(self):
        self.world_state = None
        self.tarec_values = None

        self._listeners = []

        self.add_listener(update_perceived_objects)
        self.add_listener(trigger_task)

    def add_listener(self, listener) -> None:
        """Register a new listener."""
        self._listeners.append(listener)

    def remove_listener(self, listener) -> None:
        """Unregister an existing listener."""
        if listener in self._listeners:
            self._listeners.remove(listener)

    def trigger_event(self, event_data : list[Body], world : World, context: Context) -> None:
        """Fire the event, passing event_data to every listener."""
        sem_annotations = []
        for data in event_data:
            if not data in self.world_state.known_furniture:
                try:
                    annotation = world.get_semantic_annotation_by_name(data.name)
                    sem_annotations.append(annotation)
                except WorldEntityNotFoundError:
                    print(f"Couldn't find Semantic Annotation for {data.name}")
            else:
                self.world_state.perceived_furniture.append(data)

        for listener in self._listeners:
            listener(self.world_state, self.tarec_values, self, sem_annotations, world, context)


# Usage example
def update_perceived_objects(world_state : WorldState, tarec_values : TaRecValues, handler : EventDispatcher,
                             data : list[SemanticAnnotation], world : World, context: Context) -> None:
    is_none = []
    if (handler.world_state is None) or not isinstance(handler.world_state, WorldState):
        is_none.append("handler.world_state")
    if (handler.tarec_values is None)  or not isinstance(handler.tarec_values, TaRecValues):
        is_none.append("handler.tarec_values")
    # if (handler.correct_location_tableware_dirty is None)  or not isinstance(handler.correct_location_tableware_dirty, HasSupportingSurface):
    #     is_none.append("handler.correct_location_tableware_dirty")
    # if (handler.correct_location_tableware_clean is None) or not isinstance(handler.correct_location_tableware_clean, HasSupportingSurface):
    #     is_none.append("handler.correct_location_tableware_clean")
    # if (handler.correct_location_all_other_items is None)  or not isinstance(handler.correct_location_all_other_items, HasSupportingSurface):
    #     is_none.append("handler.correct_location_all_other_items")
    # if (handler.dining_table is None) or not isinstance(handler.dining_table, HasSupportingSurface):
    #     is_none.append("handler.dining_table")
    # if handler.environment_boundaries is []:
    #     raise Exception("No environment coordinate boundaries set. Please input list of corner points.")
    #
    # if is_none:
    #     raise Exception(f"{is_none} is not set or is not a supporting surface.")
    print(world.root)
    print("#"*110 + "\n" + "#"*110)
    print(f"NEW UPDATE No. {world_state.trigger_nr}\n \n")
    world_state.trigger_nr += 1
    new_objects = 0

    for obj in data:
        if obj not in world_state.perceived_objects: # furniture already filtered out
            new_objects += 1

            world_state.perceived_objects.append(obj)

            if isinstance(obj, Tableware) and world_state.dishwasher_exists and is_supported_by_surface_cached(obj, world.get_semantic_annotation_by_name("dishwasher_rack"), world_state.support_relation_cache):
                obj.clean = True
            elif isinstance(obj, Tableware):
                obj.clean = False


            out_misplaced = misplaced(
                obj,
                world,
                tarec_values.correct_location_tableware_dirty,
                tarec_values.correct_location_tableware_clean,
                tarec_values.correct_location_food,
                tarec_values.correct_location_drinks,
                tarec_values.correct_location_all_other_items,
                world_state.surface_annotation_cache,
                world_state.support_relation_cache,
            )

            is_reachable = reachable(obj, context, tarec_values.environment_boundaries)

            if is_reachable:
                world_state.reachable_objects.append(obj)

            if out_misplaced[0]:
                world_state.misplaced_objects.append(obj)

            world_state.locations.append([obj, out_misplaced[1]])
    world_state.perceived_objects_changed = new_objects > 0
    print_perceived_objects(world_state)


def trigger_task(world_state: WorldState, tarec_values : TaRecValues, handler: EventDispatcher, data : list[SemanticAnnotation], world : World, context: Context) -> None:
    _trigger_set_table(world_state, tarec_values, world)
    _trigger_clean_table(world_state, tarec_values, data, world)
    _trigger_put_away_object(world_state, world)
    _trigger_load_dishwasher(world_state, tarec_values, world)
    _trigger_unload_dishwasher(world_state, world)

    print_tasks(world_state)

def _trigger_set_table(world_state: WorldState, tarec_values : TaRecValues, world: World) -> None:
    # trigger set the table
    ts = tm()
    # time = datetime.datetime.fromtimestamp(ts)
    time = datetime.datetime(year=2026, month=5, day=6, hour=9, minute=10)  # for testing set the table
    # time = datetime.datetime(year=2026, month=5, day=6, hour=11, minute=10)  # for testing clean the table

    if (time.hour == 9 or time.hour == 13 or time.hour == 19) and not human_near() and furniture_perceived(world_state, tarec_values.dining_table.name.name):
        table_name = tarec_values.dining_table.name.name
        exists = False
        for task in world_state.activated_tasks:
            if task.name == ("set_table_task_" + table_name):
                exists = True
                if world_state.perceived_objects_changed:
                    task.update_to_current_world_state(
                        world,
                        world_state.perceived_objects,
                        world_state.reachable_objects,
                        world_state.surface_annotation_cache,
                    )
                else:
                    # skip update task, perceived objects unchanged
                    pass
        if not exists:
            world_state.activated_tasks.append(
                SetTableTask(
                    "set_table_task_" + table_name,
                    tarec_values.dining_table,
                    world=world,
                    perceived_objects=world_state.perceived_objects,
                    reachable_objects=world_state.reachable_objects,
                    surface_cache=world_state.surface_annotation_cache,
                )
            )

def _trigger_clean_table(world_state: WorldState, tarec_values: TaRecValues, data: list[SemanticAnnotation], world: World) -> None:
    # trigger clean the table
    ts = tm()
    # time = datetime.datetime.fromtimestamp(ts)
    # time = datetime.datetime(year=2026, month=5, day=6, hour=9, minute=10)  # for testing set the table
    time = datetime.datetime(year=2026, month=5, day=6, hour=11, minute=10)  # for testing clean the table
    if (time.hour != 9 and time.hour != 13 and time.hour != 19) and not human_near() \
            and not is_empty(tarec_values.dining_table, data, world, world_state.surface_annotation_cache) \
            and furniture_perceived(world_state, tarec_values.dining_table.name.name):
        table_name = tarec_values.dining_table.name.name
        exists = False
        for task in world_state.activated_tasks:
            if task.name == ("clean_table_task_" + table_name):
                exists = True
                if world_state.perceived_objects_changed:
                    task.update_to_current_world_state(world, world_state.perceived_objects, world_state.reachable_objects)
                else:
                    # skip update task {task.name}: perceived objects unchanged
                    pass

        if not exists:
            world_state.activated_tasks.append(
                CleanTableTask("clean_table_task_" + table_name, tarec_values.dining_table, world=world,
                               perceived_objects=world_state.perceived_objects, reachable_objects= world_state.reachable_objects))

def _trigger_put_away_object(world_state: WorldState, world: World) -> None:
    # trigger put away object
    for obj in world_state.misplaced_objects:
        task_name = "put_away_object_task_" + obj.name.name
        exists = False
        for task in world_state.activated_tasks:
            if task.name == task_name:
                exists = True
                if world_state.perceived_objects_changed:
                    task.update_to_current_world_state(world, world_state.perceived_objects, world_state.reachable_objects)
                else:
                    # skip update task, perceived objects unchanged
                    pass
        if not exists:
            world_state.activated_tasks.append(PutAwayObjectTask(task_name, required_objects=[obj], world=world,
                                                             perceived_objects=world_state.perceived_objects,
                                                             reachable_objects=world_state.reachable_objects))

def _perceive_dishware(world_state: WorldState) -> list[SemanticAnnotation]:
    perceived_dishware = []
    for obj in world_state.perceived_objects:
        if isinstance(obj, (Cuttlery, Plate, Bowl, Cup)):
            perceived_dishware.append(obj)

    return perceived_dishware

def _trigger_load_dishwasher(world_state : WorldState, tarec_values : TaRecValues, world: World) -> None:
    perceived_dishware = _perceive_dishware(world_state)

    # trigger load dishwasher task
    load_dishwasher_task = None
    load_dishwasher_objects = []
    if len(perceived_dishware) > 0:
        for obj in perceived_dishware:
            if obj not in world_state.misplaced_objects and not human_near():
                load_dishwasher_task = "load_dishwasher_task"
                load_dishwasher_objects.append(obj)
    else:
        # skip counterTop query: no perceived dishware/cutlery
        pass

    if load_dishwasher_task is not None and world_state.dishwasher_exists \
            and furniture_perceived(world_state, "dishwasher_rack"):
        exists = False
        for task in world_state.activated_tasks:
            if task.name == load_dishwasher_task:
                exists = True
                if world_state.perceived_objects_changed:
                    task.update_to_current_world_state(
                        world,
                        world_state.perceived_objects,
                        world_state.reachable_objects,
                        surface_cache=world_state.surface_annotation_cache,
                        support_cache=world_state.support_relation_cache,
                        required_objects=load_dishwasher_objects,
                    )
                else:
                    # skip update task {task.name}: perceived objects unchanged
                    pass
        if not exists:
            world_state.activated_tasks.append(
                LoadDishwasherTask(
                    load_dishwasher_task,
                    world_state.perceived_objects,
                    reachable_objects=world_state.reachable_objects,
                    location_dishes=tarec_values.correct_location_tableware_dirty,
                    world=world,
                    surface_cache=world_state.surface_annotation_cache,
                    support_cache=world_state.support_relation_cache,
                    required_objects=load_dishwasher_objects,
                )
            )

def _trigger_unload_dishwasher(world_state : WorldState, world: World) -> None:
    perceived_dishware = _perceive_dishware(world_state)
    # trigger unload dishwasher task
    unload_dishwasher_task = None
    unload_dishwasher_objects = []
    if len(perceived_dishware) > 0:
        dishwasher_rack = world.get_semantic_annotation_by_name("dishwasher_rack")
        for annotation in perceived_dishware:
            if is_supported_by_surface_cached(
                    annotation,
                    dishwasher_rack,
                    world_state.support_relation_cache,
            ) and not human_near():
                unload_dishwasher_task = "unload_dishwasher_task"
                unload_dishwasher_objects.append(annotation)
    else:
        # skip dishwasher_rack query: no perceived dishware/cutlery
        pass

    if unload_dishwasher_task is not None and world_state.dishwasher_exists and furniture_perceived(world_state, "dishwasher_rack"):
        exists = False
        for task in world_state.activated_tasks:
            if task.name == unload_dishwasher_task:
                exists = True
                if world_state.perceived_objects_changed:
                    task.update_to_current_world_state(
                        world,
                        world_state.perceived_objects,
                        reachable_objects=world_state.reachable_objects,
                        surface_cache=world_state.surface_annotation_cache,
                        required_objects=unload_dishwasher_objects,
                    )
                else:
                    # skip update task {task.name}: perceived objects unchanged
                    pass
        if not exists:
            world_state.activated_tasks.append(
                UnloadDishwasherTask(
                    unload_dishwasher_task,
                    world_state.perceived_objects,
                    reachable_objects=world_state.reachable_objects,
                    world=world,
                    surface_cache=world_state.surface_annotation_cache,
                    required_objects=unload_dishwasher_objects,
                )
            )

def print_tasks(world_state : WorldState) -> None:
    print("\n \n")
    print("ACTIVE TASKS")
    print("-" * 110)

    header = f"{'Name':<60} | {'Feasibility':<15} | {'Score':<10} | {'Normalized Score':<18}"
    print(header)
    print("-" * 110)

    for task in world_state.activated_tasks:
        name = task.name
        feasibility = task.calculate_feasibility()
        score = task.reward * feasibility
        if task.duration == 0:
            norm_score = 0
        else:
            norm_score = score / task.duration

        line = f"{name:<60} | {feasibility:<15.3f} | {score:<10.3f} | {norm_score:<18.3f}"
        print(line)


def print_perceived_objects(world_state : WorldState) -> None:
    print("PERCEIVED OBJECTS")
    print("-"*110)
    print(
        f"{'Name':<35} | "
        f"{'Reachable':<10} | "
        f"{'Misplaced':<10} | "
        f"{'Belongs at Location':<30} ")
    print("-"*110)
    
    for obj in world_state.perceived_objects:
        is_reachable = False
        is_misplaced = False
        location = None

        if obj in world_state.reachable_objects:
            is_reachable = True
        if obj in world_state.misplaced_objects:
            is_misplaced = True
        for loc in world_state.locations:
            if loc[0]==obj:
                location = loc[1].name.name

        print(
            f"{obj.name.name:<35} | "
            f"{'YES' if is_reachable else 'NO':<10} | "
            f"{'YES' if is_misplaced else 'NO':<10} | "
            f"{str(location):<30} ")

def furniture_perceived(world_state : WorldState, furniture_name_to_check: str) -> bool:
    for fur in world_state.perceived_furniture:
        if fur.name.name == furniture_name_to_check:
            return True
        if furniture_name_to_check in fur.name.name:
            return True

    return False
