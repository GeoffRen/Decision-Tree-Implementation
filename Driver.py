from ID3 import id3
from pandas import read_csv


def tennis():
    """Builds, displays, and tests the tennis.csv on the id3 algorithm."""
    data = read_csv('tennis.csv')
    decision_tree = id3(data)
    print(decision_tree)
    for idx in range(len(data)):
        print("Correct?: ",
              is_correct_label(decision_tree, data, idx),
              " Value: ",
              decision_tree.traverse(data.iloc[idx]))


def zoo():
    """Builds, displays, and tests the zoo.csv on the id3 algorithm."""
    data = read_csv('zoo.csv').drop('animal_name', 1)
    decision_tree = id3(data.iloc[:80])
    print(decision_tree)
    for idx in range(80, len(data)):
        print("Correct?: ",
              is_correct_label(decision_tree, data, idx),
              " Value: ",
              decision_tree.traverse(data.iloc[idx]))


def zoo_iterative_id3(training_set_size):
    """
    Builds and tests the zoo.csv on the id3 algorithm.
    Returns a plot of training and testing error vs number of instances trained on from 1 to training_set_size.
    """
    error_points = []
    data = read_csv('zoo.csv').drop('animal_name', 1)
    for cur_training_set_size in range(1, training_set_size):
        decision_tree = id3(data.iloc[:cur_training_set_size])
        error_points.append((calc_error(range(80, len(data)), decision_tree, data, 20),
                            calc_error(range(cur_training_set_size), decision_tree, data, cur_training_set_size)))
        print(error_points[len(error_points) - 1])
    return error_points


def calc_error(data_set_range, decision_tree, data, total):
    """Calculates the training/test error."""
    correct = sum([1 for idx in data_set_range if is_correct_label(decision_tree, data, idx)])
    return correct / total


def is_correct_label(decision_tree, data, instance_idx):
    return decision_tree.traverse(data.iloc[instance_idx]) == data.iloc[instance_idx, data.shape[1] - 1]


print(zoo_iterative_id3(80))
