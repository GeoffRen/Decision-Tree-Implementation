from ID3 import id3
from pandas import read_csv

# TENNIS = True
TENNIS = False
data = read_csv('tennis.csv') if TENNIS else read_csv('zoo.csv').drop('animal_name', 1)
TRAIN_SET_SIZE = 20
TRAIN_SET = data if TENNIS else data.iloc[:TRAIN_SET_SIZE]
TEST_SET_RANGE = range(len(data)) if TENNIS else range(TRAIN_SET_SIZE, len(data))

# Creates the decision tree from data.
decision_tree = id3(TRAIN_SET)

# Prints the graphical representation of the decision tree
print(decision_tree)

# Traverses the decision tree for each row in the tennis csv and verifies it against the 'Play?' value.

for idx in TEST_SET_RANGE:
    print("Correct?: ",
          decision_tree.traverse(data.iloc[idx]) == data.iloc[idx, data.shape[1] - 1],
          " Value: ",
          decision_tree.traverse(data.iloc[idx]))
