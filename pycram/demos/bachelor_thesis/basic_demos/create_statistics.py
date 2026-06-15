import subprocess
from time import sleep
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import datetime
from time import sleep, time as tm

import move_and_perceive, move_and_perceive_pr2_apartment
from semantic_digital_twin.spatial_types import Pose, Point3, Quaternion
from semantic_digital_twin.world_description.world_entity import Body, SemanticAnnotation

# TODO: add other environments
def main():
    dictionary = {
        "correctly recognized": 0,
        "partially recognized": 0,
        "not recognized": 0,

        # tuple[times_found, times_existing_in_world]
        "banana": [0, 0],
        "bowl": [0, 0],
        "bread": [0, 0],
        "dishwasher_tab": [0, 0],
        "jeroen_cup": [0, 0],
        "knife": [0, 0],
        "milk": [0, 0],
        "plate": [0, 0],
        "spoon": [0, 0],
        "Coke": [0, 0],
        "MilkPitcher": [0, 0],
    }


    stats = []
    objects_perceived = []
    objects_in_world = []

    # CHANGE ITERATION COUNT HERE
    iterations = 5

    for i in range(0, iterations):
        locs_suturo_lab = [
            Pose(Point3(x=4.34451, y=5.20153, z=0.52), Quaternion(x=0, y=0, z=0, w=1)),
            Pose(Point3(x=4.41704, y=5.70009, z=0.52), Quaternion(x=0, y=0, z=0, w=1)),
            Pose(Point3(x=3.14758, y=-1.88804, z=0.545), Quaternion(x=0, y=0, z=0, w=1)),
            Pose(Point3(x=2.64776, y=-1.36981, z=0.545), Quaternion(x=0, y=0, z=0, w=1)),
            Pose(Point3(x=3.14174, y=0.69104, z=0.845), Quaternion(x=0, y=0, z=0, w=1)),
            Pose(Point3(x=4.32238, y=2.5949, z=0.44), Quaternion(x=0, y=0, z=0, w=1)),
            Pose(Point3(x=0.182056, y=2.0052, z=0.75), Quaternion(x=0, y=0, z=0, w=1)),
            Pose(Point3(x=1.53983, y=6.52527, z=0.71), Quaternion(x=0, y=0, z=0, w=1)),
            Pose(Point3(x=2.37098, y=6.53524, z=0.73), Quaternion(x=0, y=0, z=0, w=1)),
            Pose(Point3(x=2.10353, y=-1.20035, z=0.14), Quaternion(x=0, y=0, z=0, w=1)),
        ]

        # FOR COMPARABILITY BETWEEN HSRB AND PR2: set location list and set random_set_of_objects on false. Do one
        # statistic test with the HSRB and one with the PR2
        percents, objs_perceived, objs_in_world = move_and_perceive.main()
        stats.append(percents)
        objects_perceived.append(objs_perceived)
        objects_in_world.append(objs_in_world)

        subprocess.run(["pkill", "-f", "rclpy"])
        sleep(3)

    for j in range(0, iterations):
        statistics_per_iteration(stats[j], objects_perceived[j], objects_in_world[j], dictionary)

    print(('#' * 110 + '\n')*3)
    print(stats)
    print(('#' * 110 + '\n')*3)
    print(calculate_average_stats(stats))
    print(('#' * 110 + '\n') * 3)
    print(dictionary)
    recognized_objects_barchart(dictionary, iterations)
    recognized_tasks_barchart(dictionary, iterations)

def calculate_average_stats(percentage_list: list[list[Any]]):
    iteration_results = []
    for iteration in percentage_list:
        task_results = []
        for task in iteration:
            task_results.append(calculate_average_list(task))

        avg_iteration = calculate_average_list(task_results)
        iteration_results.append(avg_iteration)

    return iteration_results



def calculate_average_list(percentage_list: list[Any]):
    i = 0
    sum_p = 0
    for p in percentage_list:
        if p is None:
            break

        i = i + 1
        sum_p = sum_p + p
    if i != 0:
        return sum_p / i
    else:
        return 0

def statistics_per_iteration(percentage_list: list[Any], objects_perceived: list[Any], objects_in_world: list[Any], dictionary: dict[str, Any]):
    count_complete = 0
    count_partial = 0
    count_none = 0

    # check normalized score, because that is dependent on score and feasibility
    for percentage in percentage_list:
        if percentage[2] is None:
            count_none = count_none + 1
        elif percentage[2] != 1:
            count_partial = count_partial + 1
        elif percentage[2] == 1:
            count_complete = count_complete + 1

    dictionary["not recognized"] = dictionary["not recognized"] + count_none
    dictionary["partially recognized"] = dictionary["partially recognized"] + count_partial
    dictionary["correctly recognized"] = dictionary["correctly recognized"] + count_complete

    for obj in objects_in_world:
        dictionary = update_dict("banana", obj, dictionary, objects_perceived)
        dictionary = update_dict("bowl", obj, dictionary, objects_perceived)
        dictionary = update_dict("bread", obj, dictionary, objects_perceived)
        dictionary = update_dict("dishwasher_tab", obj, dictionary, objects_perceived)
        dictionary = update_dict("jeroen_cup", obj, dictionary, objects_perceived)
        dictionary = update_dict("knife", obj, dictionary, objects_perceived)
        dictionary = update_dict("milk", obj, dictionary, objects_perceived)
        dictionary = update_dict("plate", obj, dictionary, objects_perceived)
        dictionary = update_dict("spoon", obj, dictionary, objects_perceived)
        dictionary = update_dict("Coke", obj, dictionary, objects_perceived)
        dictionary = update_dict("MilkPitcher", obj, dictionary, objects_perceived)




def update_dict(name: str, obj: SemanticAnnotation, dictionary: dict[str, Any], objects_perceived: list[Any]):
    if name in obj.name.name:
        if find_obj_by_name_body(name, objects_perceived):
            dictionary[name] = [dictionary[name][0] + 1, dictionary[name][1] + 1]
        else:
            dictionary[name] = [dictionary[name][0], dictionary[name][1] + 1]
    return dictionary


def find_obj_by_name_body(name: str, list_bodies: list[Any]):
    for body in list_bodies:
        if isinstance(body, Body) or isinstance(body, SemanticAnnotation):
           if name in body.name.name:
               return True

    else:
        return False


def recognized_tasks_barchart(dictionary: dict[str, Any], iterations: int):
    def percentages(x, pos):
        """The two arguments are the value and tick position."""
        return f'{x}'

    fig, ax = plt.subplots()
    # set_major_formatter internally creates a FuncFormatter from the callable.
    ax.yaxis.set_major_formatter(percentages)
    plt.title(f"Recognized Tasks in {iterations} Iterations")


    found_values = [
        dictionary["correctly recognized"],
        dictionary["partially recognized"],
        dictionary["not recognized"],
    ]

    bars = ["correctly recognized", "partially recognized", "not recognized"]

    x_pos_raw = [0, 1, 2]
    x_pos = [i * 7 for i in x_pos_raw]

    # Create bars
    bars_plot = plt.bar(x_pos, found_values, width=4)

    for bar, pct in zip(bars_plot, found_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,  # x: center of bar
            bar.get_height(),  # y: top of bar
            f'{round(pct, 1)}',  # label
            ha='center', va='bottom',  # alignment
            fontsize=8
        )

    # Create names on the x-axis
    plt.xticks(x_pos, bars, rotation='horizontal')

    ts = tm()

    plt.subplots_adjust(bottom=0.25)
    plt.savefig(f'images/task_statistics/tasks_{datetime.datetime.fromtimestamp(ts)}.png')
    plt.close()


# source: https://matplotlib.org/stable/gallery/ticks/custom_ticker1.html
def recognized_objects_barchart(dictionary: dict[str, Any], iterations: int):

    def percentages(x, pos):
        """The two arguments are the value and tick position."""
        return f'{round(x * 100, ndigits=1)}%'

    fig, ax = plt.subplots()
    plt.title(f"Recognizing Objects Percentage in {iterations} Iterations")
    # set_major_formatter internally creates a FuncFormatter from the callable.
    ax.yaxis.set_major_formatter(percentages)


    found_percentage = [
        calculate_obj_percentage("banana", dictionary),
        calculate_obj_percentage("bowl", dictionary),
        calculate_obj_percentage("bread", dictionary),
        calculate_obj_percentage("dishwasher_tab", dictionary),
        calculate_obj_percentage("jeroen_cup", dictionary),
        calculate_obj_percentage("knife", dictionary),
        calculate_obj_percentage("milk", dictionary),
        calculate_obj_percentage("plate", dictionary),
        calculate_obj_percentage("spoon", dictionary),
        calculate_obj_percentage("Coke", dictionary),
        calculate_obj_percentage("MilkPitcher", dictionary),
    ]

    bars = ["banana", "bowl", "bread", "dishwasher tab", "cup", "knife", "milk", "plate", "spoon", "bottle", "milk pitcher"]

    x_pos_raw = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    x_pos = [i * 7 for i in x_pos_raw]

    # Create bars
    bars_plot = plt.bar(x_pos, found_percentage, width=4)

    for bar, pct in zip(bars_plot, found_percentage):
        ax.text(
            bar.get_x() + bar.get_width() / 2,  # x: center of bar
            bar.get_height(),  # y: top of bar
            f'{round(pct * 100, 1)}%',  # label
            ha='center', va='bottom',  # alignment
            fontsize=6
        )

    # Create names on the x-axis
    plt.xticks(x_pos, bars, rotation='vertical')

    ts = tm()

    plt.subplots_adjust(bottom=0.25)
    plt.savefig(f'images/object_statistics/objects_{datetime.datetime.fromtimestamp(ts)}.png')
    plt.close()


def calculate_obj_percentage(key: str, dictionary: dict[str, Any]):
    if dictionary[key][1] == 0:
        return 1 # if not in world -> technically 100% recognized
    else:
        res = dictionary[key][0]/dictionary[key][1]
        return res

def calculate_feasibility_average(percentage_list: list[Any]) -> float:


def create_diagram_for_feasibility_comparison(feasibility_hsrb: float, feasibility_pr2):
    # TODO
    pass




if __name__ == "__main__":
    main()