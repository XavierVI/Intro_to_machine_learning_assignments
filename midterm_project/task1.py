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
    def __init__(self, max_height: int=None, leaf_size: int=None):
        self.max_height = max_height
        self.leaf_size = leaf_size


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

        # iterate while the queue isn't empty
        while len(node_queue) > 0:
            # get a node from the queue
            node_dict = node_queue.pop(0)
            feature_idx = node_dict['feature']
            X_local = X['mask']
            y_local = y['mask']
            # compute the index which has the lowest
            # sum of squares value
            sample_idx = self.get_best_split(
                X_local, y_local, feature_idx
            )
            
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


    def get_boolean_masks(self, X, sample_idx, feature_idx):
        """
        This function is used to calculate the boolean masks to split 
        the given data at the data point X[sample_idx, feature_idx].

        Samples are compared only along the given feature index.

        @return: two boolean masks.
        """
        left_bool_mask = X[:, feature_idx] < X[sample_idx, feature_idx]
        right_bool_mask = X[:, feature_idx] > X[sample_idx, feature_idx]
        return left_bool_mask, right_bool_mask


    def get_best_split(self, X, y, feature_idx):
        left_bool_mask, right_bool_mask = \
            self.get_boolean_masks(X, 1, feature_idx)
        best_sample_idx = 1
        # best sum of squares error
        best_sse = self.sum_of_squares(y[left_bool_mask]) + \
                   self.sum_of_squares(y[right_bool_mask])
        for sample_idx in range(0, X.shape[0] - 1):
            left_bool_mask, right_bool_mask = self.get_boolean_masks(
                X, sample_idx, feature_idx
            )
            sse = self.sum_of_squares(y[left_bool_mask]) + \
                  self.sum_of_squares(y[right_bool_mask])

            if sse < best_sse:
                best_sample_idx = sample_idx
                best_sse = sse

        return best_sample_idx


    def sum_of_squares(self, y):
        """
        This function computes the sum of squares.

        @args:
        - y: an array of labels
        """
        return y.size[0] *  np.std(y)

    def stopping_criteria(self):
        """
        This function will calculate if we should stop building
        the tree based on the tree's current structure and the given
        hyperparameters.
        """
        return False



