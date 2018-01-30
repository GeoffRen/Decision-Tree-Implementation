from ID3 import id3
from pandas import read_csv
import matplotlib.pyplot as plt


def tennis():
    """Builds, displays, and tests the tennis.csv on the id3 algorithm."""
    data = read_csv('tennis.csv')
    decision_tree = id3(data)
    print(decision_tree)
    for idx in range(len(data)):
        print("Correct?: ",
              _is_correct_label(decision_tree, data, idx),
              " Value: ",
              decision_tree.traverse(data.iloc[idx]))


def zoo():
    """Builds, displays, and tests the zoo.csv on the id3 algorithm."""
    data = read_csv('zoo.csv').drop('animal_name', 1)
    decision_tree = id3(data.iloc[:80])
    print(decision_tree)
    for idx in range(80, len(data)):
        print("Correct?: ",
              _is_correct_label(decision_tree, data, idx),
              " Value: ",
              decision_tree.traverse(data.iloc[idx]))


def zoo_iterative_id3(training_set_size):
    """
    Builds and tests the zoo.csv on the id3 algorithm.
    Returns lists of training and testing error vs number of instances trained on from 1 to training_set_size.
    """
    train_error_points = []
    test_error_points = []
    data = read_csv('zoo.csv').drop('animal_name', 1)
    for cur_training_set_size in range(1, training_set_size):
        decision_tree = id3(data.iloc[:cur_training_set_size])
        train_error_points.append(_calc_error(range(cur_training_set_size), decision_tree, data, cur_training_set_size))
        test_error_points.append(_calc_error(range(80, len(data)), decision_tree, data, len(data) - 80))
        if cur_training_set_size % 5 == 0:
            print("Iterations left: ", training_set_size - cur_training_set_size)
    return train_error_points, test_error_points


def iterative_reduced_error_pruning(training_set_size):
    """
    Implements reduced error pruning on a tree build by the id3 algorithm.
    Returns lists of training, validation, and testing error vs number of instances trained on from 1 to
    training_set_size.
    """
    train_error_points = []
    validation_error_points = []
    test_error_points = []
    data = read_csv('zoo.csv').drop('animal_name', 1)
    for cur_training_set_size in range(1, training_set_size):
        decision_tree = id3(data.iloc[:cur_training_set_size])
        validation_error = _calc_error(range(cur_training_set_size, 80), decision_tree, data,
                                       80 - cur_training_set_size)
        _mark_nodes(decision_tree, decision_tree, data, validation_error, cur_training_set_size)
        train_error_points.append(_calc_error(range(cur_training_set_size), decision_tree, data,
                                              cur_training_set_size))
        validation_error_points.append(_calc_error(range(cur_training_set_size, 80), decision_tree, data,
                                                   80 - cur_training_set_size))
        test_error_points.append(_calc_error(range(80, len(data)), decision_tree, data,
                                             len(data) - 80))
        if cur_training_set_size % 5 == 0:
            print("Iterations left: ", training_set_size - cur_training_set_size)
    return train_error_points, validation_error_points, test_error_points


def plot_training_testing_error(train_error_points, test_error_points, validation_error_points=None):
    """Plots train_error_points and test_error_points."""
    plt.title("Training and Testing Error")
    plt.xlabel("Iteration")
    plt.ylabel("Training/Testing Error")
    training_error = plt.scatter(range(1, len(train_error_points) + 1), train_error_points)
    testing_error = plt.scatter(range(1, len(test_error_points) + 1), test_error_points)
    if validation_error_points:
        validation_error = plt.scatter(range(1, len(validation_error_points) + 1), validation_error_points)
        plt.legend((training_error, testing_error, validation_error),
                   ("Training Error", "Testing Error", "Validation Error"),
                   scatterpoints=1,
                   loc=1,
                   ncol=3,
                   fontsize=8)
    else:
        plt.legend((training_error, testing_error),
                   ("Training Error", "Testing Error"),
                   scatterpoints=1,
                   loc=1,
                   ncol=3,
                   fontsize=8)
    plt.show()


# Prunes nodes by marking them. Marked nodes act as if they're leaf nodes during traversal by returning their _default
# value. _default is the majority value of the dataset at that node.
def _mark_nodes(root, node, data, validation_error, training_set_size):
    if not node.get_children():  # So if node is a LabelNode
        return True, validation_error
    prune_this = True
    for child in node.get_children():
        pruned_child, validation_error = _mark_nodes(root, child, data, validation_error, training_set_size)
        if not pruned_child:
            prune_this = False
    if prune_this:  # Only prune the current node if all its FeatureNode children were pruned too.
        node.set_marked(True)
        pruned_validation_error = _calc_error(range(training_set_size, 80), root, data,
                                              80 - training_set_size)
        if pruned_validation_error <= validation_error:
            return True, pruned_validation_error  # Propagate the new pruned_validation_error to the next calls.
        else:
            node.set_marked(False)
            return False, validation_error
    else:
        return False, validation_error


# Calculates the training/test error.
def _calc_error(data_set_range, decision_tree, data, total):
    return sum([1 for idx in data_set_range if not _is_correct_label(decision_tree, data, idx)]) / total


# Determines if an instance is labeled correctly by the decision_tree.
def _is_correct_label(decision_tree, data, instance_idx):
    return decision_tree.traverse(data.iloc[instance_idx]) == data.iloc[instance_idx, data.shape[1] - 1]


# Part 1
# tennis()

# Part 2
# train_error_points, test_error_points = zoo_iterative_id3(80)
# plot_training_testing_error(train_error_points, test_error_points)

# Part 3
train_error_points, validation_error_points, test_error_points = iterative_reduced_error_pruning(50)
plot_training_testing_error(train_error_points, test_error_points, validation_error_points)
