from ID3 import id3
from pandas import read_csv

# data = read_csv('tennis.csv')
data = read_csv('zoo.csv').drop('animal_name', 1)

# Creates the
decision_tree = id3(data)

# Prints the graphical representation of the decision tree
print(decision_tree)

# Traverses the decision tree for each row in the tennis csv and verifies it against the 'Play?' value.
for idx in range(len(data)):
    print(decision_tree.traverse(data.iloc[idx]) == data.iloc[idx, data.shape[1] - 1])
