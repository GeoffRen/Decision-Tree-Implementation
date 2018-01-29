from ID3 import id3
from pandas import read_csv

# data = read_csv('tennis.csv')
data = read_csv('zoo.csv')
test = id3(data)
# Prints the graphical representation of the decision tree
print(test)
# Traverses the decision tree for each row in the tennis csv and verifies it against the 'Play?' value.
for idx in range(len(data)):
    print(test.traverse(data.iloc[idx]) == data.iloc[idx, data.shape[1] - 1])
