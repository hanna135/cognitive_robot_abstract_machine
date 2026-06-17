import subprocess
from time import sleep
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import datetime
from time import sleep, time as tm

import move_and_perceive, move_and_perceive_pr2_apartment
from semantic_digital_twin.robots.pr2 import PR2
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
    feasibility_dicts = []

    # CHANGE ITERATION COUNT HERE
    iterations = 1
    robot = PR2

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
        percents, objs_perceived, objs_in_world, dictionary_feasibilities = move_and_perceive.main(robot, locations=locs_suturo_lab)
        stats.append(percents)
        objects_perceived.append(objs_perceived)
        objects_in_world.append(objs_in_world)
        feasibility_dicts.append(dictionary_feasibilities)

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
    if robot == PR2:
        str_robot = "pr2"
    else:
        str_robot = "hsrb"
    save_feasibility_statistics(calculate_feasibility_average(feasibility_dicts), str_robot)

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

def calculate_feasibility_average(feasibility_dicts: list[dict]) -> dict | None:
    if feasibility_dicts is None:
        return None

    dictio = {
        "set_the_table": 0,
        "clean_the_table": 0,
        "load_the_dishwasher": 0,
        "unload_the_dishwasher": 0,
        "put_away_object_task_bowl": 0,
        "put_away_object_task_spoon": 0,
        "put_away_object_task_Static_MilkPitcher": 0,
        "put_away_object_task_Static_CokeBottle": 0,
        "put_away_object_task_jeroen_cup": 0,
        "put_away_object_task_dishwasher_tab": 0,
        "put_away_object_task_banana": 0,
        "put_away_object_task_bread": 0,
        "put_away_object_task_knife": 0,
        "put_away_object_task_plate": 0,

    }

    for dicti in feasibility_dicts:
        dictio["set_the_table"] = dictio["set_the_table"] + dicti["set_the_table"]
        dictio["clean_the_table"] = dictio["clean_the_table"] + dicti["clean_the_table"]
        dictio["load_the_dishwasher"] = dictio["load_the_dishwasher"] + dicti["load_the_dishwasher"]
        dictio["unload_the_dishwasher"] = dictio["unload_the_dishwasher"] + dicti["unload_the_dishwasher"]
        dictio["put_away_object_task_bowl"] = dictio["put_away_object_task_bowl"] + dicti["put_away_object_task_bowl"]
        dictio["put_away_object_task_spoon"] = dictio["put_away_object_task_spoon"] + dicti["put_away_object_task_spoon"]
        dictio["put_away_object_task_Static_MilkPitcher"] = dictio["put_away_object_task_Static_MilkPitcher"] + dicti["put_away_object_task_Static_MilkPitcher"]
        dictio["put_away_object_task_Static_CokeBottle"] = dictio["put_away_object_task_Static_CokeBottle"] + dicti["put_away_object_task_Static_CokeBottle"]
        dictio["put_away_object_task_jeroen_cup"] = dictio["put_away_object_task_jeroen_cup"] + dicti["put_away_object_task_jeroen_cup"]
        dictio["put_away_object_task_dishwasher_tab"] = dictio["put_away_object_task_dishwasher_tab"] + dicti["put_away_object_task_dishwasher_tab"]
        dictio["put_away_object_task_banana"] = dictio["put_away_object_task_banana"] + dicti["put_away_object_task_banana"]
        dictio["put_away_object_task_bread"] = dictio["put_away_object_task_bread"] + dicti["put_away_object_task_bread"]
        dictio["put_away_object_task_knife"] = dictio["put_away_object_task_knife"] + dicti["put_away_object_task_knife"]
        dictio["put_away_object_task_plate"] = dictio["put_away_object_task_plate"] + dicti["put_away_object_task_plate"]

    dictio["set_the_table"] = dictio["set_the_table"] / len(feasibility_dicts)
    dictio["clean_the_table"] = dictio["clean_the_table"] / len(feasibility_dicts)
    dictio["load_the_dishwasher"] = dictio["load_the_dishwasher"] / len(feasibility_dicts)
    dictio["unload_the_dishwasher"] = dictio["unload_the_dishwasher"] / len(feasibility_dicts)
    dictio["put_away_object_task_bowl"] = dictio["put_away_object_task_bowl"] / len(feasibility_dicts)
    dictio["put_away_object_task_spoon"] = dictio["put_away_object_task_spoon"] / len(feasibility_dicts)
    dictio["put_away_object_task_Static_MilkPitcher"] = dictio["put_away_object_task_Static_MilkPitcher"]  / len(feasibility_dicts)
    dictio["put_away_object_task_Static_CokeBottle"] = dictio["put_away_object_task_Static_CokeBottle"] / len(feasibility_dicts)
    dictio["put_away_object_task_jeroen_cup"] = dictio["put_away_object_task_jeroen_cup"] / len(feasibility_dicts)
    dictio["put_away_object_task_dishwasher_tab"] = dictio["put_away_object_task_dishwasher_tab"] / len(feasibility_dicts)
    dictio["put_away_object_task_banana"] = dictio["put_away_object_task_banana"] / len(feasibility_dicts)
    dictio["put_away_object_task_bread"] = dictio["put_away_object_task_bread"] / len(feasibility_dicts)
    dictio["put_away_object_task_knife"] = dictio["put_away_object_task_knife"] / len(feasibility_dicts)
    dictio["put_away_object_task_plate"] = dictio["put_away_object_task_plate"] / len(feasibility_dicts)

    return dictio


def save_feasibility_statistics(dictionary: dict, robot: str):
    ts = tm()

    try:
        with open(f"feasibility_stats/{robot}_{datetime.datetime.fromtimestamp(ts)}", "x", encoding="utf-8") as file:
            file.write(f"set_the_table: {dictionary['set_the_table']}\n")
            file.write(f"clean_the_table: {dictionary['clean_the_table']}\n")
            file.write(f"load_the_dishwasher: {dictionary['load_the_dishwasher']}\n")
            file.write(f"unload_the_dishwasher: {dictionary['unload_the_dishwasher']}\n")
            file.write(f"put_away_object_task_bowl: {dictionary['put_away_object_task_bowl']}\n")
            file.write(f"put_away_object_task_spoon: {dictionary['put_away_object_task_spoon']}\n")
            file.write(f"put_away_object_task_Static_MilkPitcher: {dictionary['put_away_object_task_Static_MilkPitcher']}\n")
            file.write(f"put_away_object_task_Static_CokeBottle: {dictionary['put_away_object_task_Static_CokeBottle']}\n")
            file.write(f"put_away_object_task_jeroen_cup: {dictionary['put_away_object_task_jeroen_cup']}\n")
            file.write(f"put_away_object_task_dishwasher_tab: {dictionary['put_away_object_task_dishwasher_tab']}\n")
            file.write(f"put_away_object_task_banana: {dictionary['put_away_object_task_banana']}\n")
            file.write(f"put_away_object_task_bread: {dictionary['put_away_object_task_bread']}\n")
            file.write(f"put_away_object_task_knife: {dictionary['put_away_object_task_knife']}\n")
            file.write(f"put_away_object_task_plate: {dictionary['put_away_object_task_plate']}\n")


    except FileExistsError:
        print("file.txt already exists, exclusive creation aborted.")







## compare robots

"""
Usage:
    python plot_robot_tasks.py --pr2 pr2_results.txt --hsrb hsrb_results.txt
    python plot_robot_tasks.py --pr2 pr2_results.txt --hsrb hsrb_results.txt --output my_plot.png

File format (one entry per line):
    task_name: 0.6666666666666666
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ── File loader ───────────────────────────────────────────────────────────────

def load_file(path: str) -> dict:
    data = {}
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ": " not in line:
                print(f"Warning: skipping malformed line {lineno} in {path!r}: {line!r}")
                continue
            key, _, val = line.partition(": ")
            try:
                data[key.strip()] = float(val.strip())
            except ValueError:
                print(f"Warning: could not parse value on line {lineno} in {path!r}: {val!r}")
    return data


# ── Label prettifier ──────────────────────────────────────────────────────────

def pretty(name: str) -> str:
    prefix = "put_away_object_task_"
    if name.startswith(prefix):
        name = "put away " + name[len(prefix):]
    return name.replace("_", " ")


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot(pr2_data: dict, hsrb_data: dict, output_path: str) -> None:
    # Use all tasks that appear in either file, preserving PR2 order then any HSRB extras
    task_names = list(pr2_data.keys())
    for t in hsrb_data:
        if t not in pr2_data:
            task_names.append(t)

    labels    = [pretty(t) for t in task_names]
    pr2_vals  = [pr2_data.get(t,  0.0) for t in task_names]
    hsrb_vals = [hsrb_data.get(t, 0.0) for t in task_names]

    x     = np.arange(len(task_names))
    width = 0.38
    gap   = 0.04

    fig, ax = plt.subplots(figsize=(16, 6))

    bars_pr2  = ax.bar(x - width / 2 - gap / 2, pr2_vals,  width, label="PR2",  color="#4C72B0", zorder=3)
    bars_hsrb = ax.bar(x + width / 2 + gap / 2, hsrb_vals, width, label="HSRB", color="#DD8452", zorder=3)

    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title("Task Success Rate: PR2 vs HSRB", fontsize=14, fontweight="bold", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.yaxis.grid(True, linestyle="--", alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    def label_bars(bars):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.01,
                f"{h:.0%}",
                ha="center", va="bottom", fontsize=7.5, color="#333333",
            )

    label_bars(bars_pr2)
    label_bars(bars_hsrb)

    ax.legend(fontsize=11, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def compare():
    parser = argparse.ArgumentParser(
        description="Plot task success rates for PR2 and HSRB robots."
    )
    parser.add_argument("--pr2",    required=True, help="Path to PR2 results file")
    parser.add_argument("--hsrb",   required=True, help="Path to HSRB results file")
    parser.add_argument("--output", default="robot_task_comparison.png",
                        help="Output image path (default: robot_task_comparison.png)")
    args = parser.parse_args()

    for path in (args.pr2, args.hsrb):
        if not Path(path).is_file():
            print(f"Error: file not found: {path!r}", file=sys.stderr)
            sys.exit(1)

    pr2_data  = load_file(args.pr2)
    hsrb_data = load_file(args.hsrb)

    if not pr2_data:
        print(f"Error: no data loaded from PR2 file {args.pr2!r}", file=sys.stderr)
        sys.exit(1)
    if not hsrb_data:
        print(f"Error: no data loaded from HSRB file {args.hsrb!r}", file=sys.stderr)
        sys.exit(1)

    plot(pr2_data, hsrb_data, args.output)

def create_diagram_for_feasibility_comparison():
    ts = tm()

    directory = Path("/home/hanna/bachelor_ws/src/cognitive_robot_abstract_machine/pycram/demos/bachelor_thesis/basic_demos/feasibility_stats")

    print(list(directory.glob("*")))

    pr2_file = max(directory.glob("pr2_*"), key=lambda f: f.stat().st_mtime)
    hsrb_file = max(directory.glob("hsrb_*"), key=lambda f: f.stat().st_mtime)

    pr2_data = load_file(pr2_file.as_posix())
    hsrb_data = load_file(hsrb_file.as_posix())
    plot(pr2_data, hsrb_data, f"images/comparison/comparison_{datetime.datetime.fromtimestamp(ts)}.png")


if __name__ == "__main__":
    #main()
    create_diagram_for_feasibility_comparison()
