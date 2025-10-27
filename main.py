from task import *
import sys
import json


def solve_task_id(task_file, task_type="training"):
    """
    solves a given task and saves the solution to a file
    """
    if task_type == "training":
        data_path = "dataset/training/"
    else:
        data_path = "dataset/evaluation/"
    task = Task(data_path + task_file)

    abstraction, solution_apply_call, error, train_error, solving_time, nodes_explored = task.solve(
        shared_frontier=True, time_limit=1800, do_constraint_acquisition=True, save_images=True)

    solution = {"abstraction": abstraction, "apply_call": solution_apply_call, "train_error": train_error,
                "test_error": error, "time": solving_time, "nodes_explored": nodes_explored}
    if error == 0:
        with open('solutions/correct/solutions_{}'.format(task_file), 'w') as fp:
            json.dump(solution, fp)
    else:
        with open('solutions/incorrect/solutions_{}'.format(task_file), 'w') as fp:
            json.dump(solution, fp)
    print(solution)


if __name__ == "__main__":

    # example tasks:
    # recolor task: d2abd087.json
    # dynamic recolor task: ddf7fa4f.json
    # movement task: 3906de3d.json
    # augmentation task: d43fd935.json

    # usage:
    # python main.py <task_file.json> <training|evaluation> [abstraction1,abstraction2,abstraction3]
    # e.g.:
    # python main.py ddf7fa4f.json training ccg,na,mcccg

    task_file = str(sys.argv[1])
    task_type = str(sys.argv[2])

    # optional third argument: comma-separated list of abstraction names to restrict search to
    allowed_abstractions = None
    if len(sys.argv) >= 4 and sys.argv[3].strip() != "":
        allowed_abstractions = [s.strip() for s in sys.argv[3].split(",") if s.strip()]

    task = Task(( "dataset/training/" if task_type == "training" else "dataset/evaluation/" ) + task_file)

    # if user provided allowed_abstractions, attach them to the Task instance
    if allowed_abstractions:
        task.allowed_abstractions = allowed_abstractions

    abstraction, solution_apply_call, error, train_error, solving_time, nodes_explored = task.solve(
        shared_frontier=True, time_limit=1800, do_constraint_acquisition=True, save_images=True)

    solution = {"abstraction": abstraction, "apply_call": solution_apply_call, "train_error": train_error,
                "test_error": error, "time": solving_time, "nodes_explored": nodes_explored}
    if error == 0:
        with open('solutions/correct/solutions_{}'.format(task_file), 'w') as fp:
            json.dump(solution, fp)
    else:
        with open('solutions/incorrect/solutions_{}'.format(task_file), 'w') as fp:
            json.dump(solution, fp)
    print(solution)


