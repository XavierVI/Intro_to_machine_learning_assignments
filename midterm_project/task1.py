import numpy as np


    
class BST:
    """
    This class is a binary search tree.

    @args:
    - condition: a boolean expression associated with the node
    """
    def __init__(self, min_idx, max_idx, feature):
        """
        @args:
        - condition: a boolean expression associated with the node.
        This condition should be used to determine how the data is split.
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
        num_of_features = X.size[1]
        
        # initialize the root node
        node = BST()

        # used to track which node to visit next
        node_queue = []
        # add root node to the queue
        node_queue.append({
            'node': node,
            'mask': np.array([True] * X.size[0]),
            'feature': 0
        })


        while len(node_queue) > 0:
            # get a node from the queue
            node_dict = node_queue.pop(0)
            feature_idx = node_dict['feature']
            # compute the index which has the lowest
            # sum of squares value (excluding end points)
            sample_idx = np.argmin(self.sum_of_squares(y[node_dict['mask']]))
            
            # use the index to split the data at X[idx]
            # this is also where we set the condition of the node
            node.condition = lambda x: x <= X[sample_idx, feature_idx]
            
            # create the children for the node
            node.left_child = BST()
            node.right_child = BST()
            next_feature = 0 if feature_idx + 1 > num_of_features else feature_idx + 1
            
            # add them to the queue
            node_queue.append({
                'node': node.left_child,
                'mask': X[:, feature_idx] < X[sample_idx, feature_idx],
                'feature': next_feature
            })
            node_queue.append({
                'node': node.right_child,
                'mask': X[:, feature_idx] >= X[sample_idx, feature_idx],
                'feature': next_feature
            })


    def sum_of_squares(self, y):
        return y.size[0] *  np.std(y)

    def stopping_criteria(self):
        """
        This function will calculate if we should stop building
        the tree based on the tree's current structure and the given
        hyperparameters.
        """
        return False



