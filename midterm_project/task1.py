import numpy as np


    
class BST:
    """
    This class is a binary search tree.

    @args:
    - condition: a boolean expression associated with the node
    """
    def __init__(self):
        """
        @args:
        - condition: a boolean expression associated with the node
        """
        self.condition = None
        self.left_child = None
        self.right_child = None

    def set_condition(self, exp: lambda: bool):
        self.condition = exp

    def set_left_child(self, node):
        self.left_child = node

    def set_right_child(self, node):
        self.right_child = node
    


class RegressionTree:
    def __init__(self, max_height: int, leaf_size: int):
        pass


    def decision_path(X):
        pass

    def fit(self, X, y):
        # index of the feature
        feature_idx = 0
        # used to track which node to visit next
        node_queue = []
        # initialize the root node
        node = BST()

        while not self.stopping_criteria():
            # compute the index which has the lowest
            # sum of squares value (excluding end points)
            sample_idx = np.argmin(self.sum_of_squares(y[1:-1]))
            
            # use the index to split the data at X[idx]
            # this is also where we set the condition of the node
            condition = lambda x: x <= X[sample_idx, feature_idx]



    def sum_of_squares(self, y):
        return y.size[0] *  np.std(y)

    def stopping_criteria(self):
        """
        This function will calculate if we should stop building
        the tree based on the tree's current structure and the given
        hyperparameters.
        """
        return False


model = Node(lambda x: x <= 3)


