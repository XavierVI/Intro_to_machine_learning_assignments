import numpy as np


class Node:
    def __init__(self, condition):
        """
        @args:
        - condition: a boolean expression associated with the node
        """
        self.left_child = None
        self.right_child = None
        self.condition = condition
        pass
    
    def add_node(self, node: Node, left: bool):

        if left:
            self.left_child = node
        else:
            self.right_child = node
    

class BST:
    """
    This class is a binary search tree.
    """
    def __init__(self, condition):
        """
        @args:
        - condition: a boolean expression associated with the node
        """
        self.root = Node(condition)
    


class RegressionTree:
    def __init__(self, max_height: int, leaf_size: int):
        pass

    
    def decision_path(X):
        pass

    def fit(self, X, y):
        # used to track which node to visit next
        stack = []

        while not self.__stopping_criteria():
            # compute the index which has the lowest
            # sum of squares value (excluding end points)
            np.argmin(self.__sum_of_squares(y[1:-1]))

        pass

    def __sum_of_squares(self, y):
        return y.size[0] *  np.std(y)

    def __stopping_criteria(self):
        """
        This function will calculate if we should stop building
        the tree based on the tree's current structure and the given
        hyperparameters.
        """
        return False


model = RegressionTree(0)
model.__sum_of_squares([0])
