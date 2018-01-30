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
        """Returns the main data info in the node."""
        pass

    @abstractmethod
    def traverse(self, instance):
        """Traverses the node."""
        pass

    @abstractmethod
    def get_children(self):
        """Returns a list of the node's children."""
        pass


class FeatureNode(BaseNode):
    """A feature in the decision tree."""

    def __init__(self, feature):
        """
        _feature is the feature this FeatureNode represents.
        _edges is a dictionary mapping values of _feature to other BaseNodes.
        _default is the majority label at this FeatureNode.
        _default is used if the FeatureNode doesn't contain a certain value as an edge or during reduced error pruning.
        _marked determines if a FeatureNode is being considered for pruning.
        """
        self._feature = feature
        self._edges = {}
        self._default = None
        self._marked = False

    def set_default(self, val):
        """Setter method that sets _default."""
        self._default = val

    def add_edge(self, val, node):
        """Adds an edge from val to node in _edges."""
        self._edges[val] = node

    def prune_node(self, old_node):
        """Prune a FeatureNode by turning it into a LabelNode with the _default as the label."""
        self._edges = {val:LabelNode(old_node.get_default()) for val, node in self._edges.items()}
        # for val, node in self._edges.items():
        #     if node is old_node:
        #         self._edges[val] = LabelNode(old_node.get_default())

    def get_data(self):
        """Getter method that gets _feature."""
        return self._feature

    def get_default(self):
        """Getter method that gets _default."""
        return self._default

    def traverse(self, instance):
        """Traverses the FeatureNode."""
        try:
            return self._default if self._marked else self._edges[instance[self._feature]].traverse(instance)
        except KeyError:  # Happens when _edges doesn't have instance[self._feature] as an edge.
            return self._default

    def get_children(self):
        """Returns this FeatureNode's children, so the values of _edges."""
        return self._edges.values()

    def get_marked(self):
        """Getter method that gets _marked."""
        return self._marked

    def set_marked(self, marked):
        self._marked = marked

    def __repr__(self):
        """
        Pretty crappy implementation. Just quick and dirty level order traverse with a queue.
        This should be refactored but probably won't be.
        """
        if self._marked:
            return self._feature
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
            if node.get_children() and not node._marked:
                for val, branch in node._edges.items():
                    queue.append(branch)
                    cur_edges.append("({}, {})".format(val, branch.get_data()))
                    next_level_count += 1
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
        """_label is the label this LabelNode represents."""
        self._label = label

    def get_data(self):
        """Getter method that gets _label."""
        return self._label

    def traverse(self, instance):
        """Since LabelNodes are leaf nodes, they represent the final classification. Hence, just returns _label."""
        return self._label

    def get_children(self):
        """Since LabelNodes are leaf nodes, they represent the final classification. Hence, just returns None."""
        return None

    def __repr__(self):
        """Since LabelNodes are leaf nodes, they represent the final classification. Hence, just returns _label."""
        return self._label
