from math import log2
from abc import ABC, abstractmethod


def id3(examples):
    """Creates a decision tree using the ID3 algorithm."""
    labels = examples.iloc[:, examples.shape[1] - 1]
    if examples.shape[1] == 1:
        return LabelNode(labels.value_counts().index[0])
    elif len(labels.unique()) == 1:
        return LabelNode(labels.iloc[0])
    else:
        best_feature = _get_best_feature(examples)
        node = FeatureNode(best_feature)
        node.set_default(examples.iloc[:, examples.shape[1] - 1].mode()[0])
        for val in examples[best_feature].unique():
            branch_examples = examples[examples[best_feature] == val].drop(best_feature, axis=1)
            node.add_edge(val, id3(branch_examples))
        return node


# Returns the feature that results in the greatest information gain from a dataframe of examples.
def _get_best_feature(examples):
    cur_entropy = _calc_entropy(examples.iloc[:, examples.shape[1] - 1].value_counts())
    features = list(examples)[:examples.shape[1] - 1]  # Cuts off the last feature (the label).
    features = list(map(lambda feature: (feature, _calc_information_gain(examples, cur_entropy, feature)), features))
    return max(features, key=lambda feature: feature[1])[0]


# Calculates information gain if a dataframe of examples were split on a specific col.
def _calc_information_gain(examples, entropy, col):
    for val in examples[col].unique():
        # Gets the value_counts() of the label of the rows with val in col.
        classification_counts = examples.loc[examples[col] == val, list(examples)[examples.shape[1] - 1]].value_counts()
        entropy -= sum(classification_counts) / len(examples) * _calc_entropy(classification_counts)
    return entropy


# Calculates the entropy from a series simply containing a count of labels.
def _calc_entropy(classification_counts):
    ret = 0
    for classification_count in classification_counts:
        proportion = classification_count / sum(classification_counts)
        ret -= proportion * log2(proportion)
    return ret


class BaseNode(ABC):
    """The node all others are derived from."""

    @abstractmethod
    def get_data(self):
        pass

    @abstractmethod
    def traverse(self, instance):
        pass


class FeatureNode(BaseNode):
    """A feature in the decision tree."""

    def __init__(self, feature):
        self._feature = feature
        self._edges = {}
        self._default = None

    def set_default(self, val):
        self._default = val

    def add_edge(self, val, node):
        self._edges[val] = node

    def get_data(self):
        return self._feature

    def traverse(self, instance):
        try:
            return self._edges[instance[self._feature]].traverse(instance)
        except KeyError: # Happens when instance doesn't have _feature as a feature.
            return self._default

    # Pretty crappy implementation. Just quick and dirty level order traverse with a queue.
    # This should be refactored but probably won't be.
    def __repr__(self):
        ret = ""
        queue = [self]
        cur_level = []
        cur_edges = []
        level_count = 1
        next_level_count = 0
        while len(queue) > 0:
            cur_level.append(queue[0].get_data())
            node = queue.pop(0)
            level_count -= 1
            try:
                for val, branch in node._edges.items():
                    queue.append(branch)
                    cur_edges.append("({}, {})".format(val, branch.get_data()))
                    next_level_count += 1
            except:
                pass
            if not level_count:
                ret = ''.join([ret, " ".join(map(str, cur_level)), "\n", " ".join(map(str, cur_edges)), "\n"])
                level_count = next_level_count
                next_level_count = 0
                cur_level = []
                cur_edges = []
        return ret


class LabelNode(BaseNode):
    """A label in the decision tree."""

    def __init__(self, label):
        self._label = label

    def get_data(self):
        return self._label

    def traverse(self, instance):
        return self._label

    def __repr__(self):
        return self._label
