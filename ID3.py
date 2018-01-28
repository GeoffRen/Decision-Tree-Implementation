from math import log2
from abc import ABC, abstractmethod


def id3(examples):
    """Creates a decision tree using the ID3 algorithm."""
    classifications = examples['Play?']
    if classifications.str.contains('Yes').all():
        return LabelNode('Yes')
    elif classifications.str.contains('No').all():
        return LabelNode('No')
    elif len(examples.columns) <= 1:
        return LabelNode(classifications.value_counts().index[0])
    else:
        best_feature = _get_best_feature(examples)
        node = FeatureNode(best_feature)
        for val in examples[best_feature].unique():
            branch_examples = examples[examples[best_feature] == val].drop(best_feature, axis=1)
            if branch_examples.empty: # Will never happen in the current implementation
                node.add_edge(val, LabelNode(examples['Play?'].mode()[0]))
            else:
                node.add_edge(val, id3(branch_examples))
        return node


# Returns the feature that results in the greatest information gain from a dataframe of examples.
def _get_best_feature(examples):
    cur_entropy = _calc_entropy(examples['Play?'].value_counts())
    best_feature = ('', 0)
    for feature in examples.columns.values:
        if feature == 'Play?':
            continue
        information_gain = _calc_information_gain(examples, cur_entropy, feature)
        if information_gain > best_feature[1]:
            best_feature = (feature, information_gain)
    return best_feature[0]


# Calculates information gain if a dataframe of examples were split on a specific col.
def _calc_information_gain(examples, entropy, col):
    for val in examples[col].unique():
        classification_counts = examples.loc[examples[col] == val, 'Play?'].value_counts()
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

    def add_edge(self, val, node):
        self._edges[val] = node

    def get_data(self):
        return self._feature

    def traverse(self, instance):
        return self._edges[instance[self._feature]].traverse(instance)


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
                ret = ''.join([ret, " ".join(cur_level), "\n", " ".join(cur_edges), "\n"])
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
