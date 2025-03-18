import numpy as np

class Node:
    def __init__(
        self,
        left_child=None, right_child=None,
        condition=None, value: float=None):
        """
        @args:
        - condition: a boolean expression associated with the node.
        This condition should be used to determine how the data is split.
        """
        self.left_child = left_child
        self.right_child = right_child
        self.feature = None
        self.condition_value = condition
        self.pred_value = value
    
    def condition(self, x): 
        return x[self.feature] >= self.condition_value

    def is_leaf(self):
        return self.left_child == None and self.right_child == None

    def set_left_child(self, node):
        self.left_child = node

    def set_right_child(self, node):
        self.right_child = node
    

class BST:
    """
    This class is a binary search tree.

    @args:
    - condition: a boolean expression associated with the node
    """
    def __init__(self):
        self.root_node = Node()

    def get_height(self, node):
        if node == None:
            return 0

        return 1 + max(
            self.get_height(node.left_child),
            self.get_height(node.right_child)
        )

    def get_num_of_leaves(self):
        num_of_leaves = 0
        node_stack = []
        node_stack.append(self.root_node)

        while len(node_stack) > 0:
            node = node_stack.pop()

            if node.left_child != None:
                node_stack.append(node.left_child)

            if node.right_child != None:
                node_stack.append(node.right_child)

            if node.left_child == None and node.right_child == None:
                num_of_leaves += 1
        
        return num_of_leaves



class RegressionTree:
    def __init__(self, max_height: int=None, leaf_size: int=None):
        self.max_height = max_height
        self.leaf_size = leaf_size
        self.height = 0
        self.bst = BST()


    def decision_path(self, x):
        node = self.bst.root_node
        # iterate until we get a leaf node
        while not node.is_leaf():
            if node.condition(x):
                print(f'x[{node.feature}] >= {node.condition_value}')
                node = node.right_child

            else:
                print(f'x[{node.feature}] < {node.condition_value}')
                node = node.left_child

        print(f'x == {node.pred_value}')

    def predict(self, x):
        node = self.bst.root_node

        # iterate until we get a leaf node
        while not node.is_leaf():
            if node.condition(x):
                node = node.right_child

            else:
                node = node.left_child

        return node.pred_value

    def fit(self, X, y):
        num_of_features = X.shape[1]
        
        # start at the root node
        node = self.bst.root_node

        # calculate the impurity of the whole dataset
        sse = self.sum_of_squares(y)

        # used to track which node to visit next
        node_queue = []
        # add root node to the queue
        node_queue.append({
            'node': node,
            'depth': 0,
            'mask': np.array([True] * X.shape[0]),
            'impurity': sse,
        })
        print(f'Length of queue: {len(node_queue)}')

        # iterate while the queue isn't empty
        while len(node_queue) > 0:
            # get a node from the queue
            node_dict = node_queue.pop(0)
            curr_node = node_dict['node']
            depth = node_dict['depth']
            num_of_leaves = self.bst.get_num_of_leaves()
            impurity = node_dict['impurity']
            X_local = X[node_dict['mask']]
            y_local = y[node_dict['mask']]

            # determine if this node should be a leaf
            if X_local.shape[0] < 3 or impurity == 0 or \
                self.can_add_node(num_of_leaves, depth):
                # set the value property
                curr_node.pred_value = np.mean(y_local, axis=0)
                # continue to the next iteration
                continue

            # compute the index which has the lowest
            # sum of squares value
            best_feature_idx = 0
            best_sample_idx, best_left_sse, best_right_sse = \
                self.get_best_split(X_local, y_local, best_feature_idx)

            # iterate over all features to find the best split
            for feature_idx in range(1, num_of_features):
                sample_idx, left_sse, right_sse = \
                    self.get_best_split(X_local, y_local, best_feature_idx)
                
                if best_left_sse + best_right_sse > left_sse + right_sse:
                    best_sample_idx = sample_idx
                    best_left_sse = left_sse
                    best_right_sse = right_sse
                    best_feature_idx = feature_idx
            
            # use the index which splits the data at X[sample_idx, feature_idx]
            # to set the condition property of the node
            curr_node.condition_value = X[best_sample_idx, best_feature_idx]
            curr_node.feature = best_feature_idx

            if self.leaf_size == None or (self.leaf_size != None and num_of_leaves <= self.leaf_size):
                # create a node and set it as the left child of curr_node
                curr_node.left_child = Node()
                # add left_child to the queue
                node_queue.append({
                    'node': curr_node.left_child,
                    'depth': depth + 1,
                    'mask': (X[:, best_feature_idx] < X_local[best_sample_idx, best_feature_idx])
                            & node_dict['mask'],
                    'impurity': best_left_sse,
                })

            if self.leaf_size == None or (self.leaf_size != None and num_of_leaves + 1 <= self.leaf_size):
                # create a node and set it as the right child of curr_node
                curr_node.right_child = Node()
                # add right_child to the queue
                node_queue.append({
                    'node': curr_node.right_child,
                    'depth': depth + 1,
                    'mask': (X[:, best_feature_idx] > X_local[best_sample_idx, best_feature_idx]) 
                            & node_dict['mask'],
                    'impurity': best_right_sse,
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
        best_left_sse = self.sum_of_squares(y[left_bool_mask])
        best_right_sse = self.sum_of_squares(y[right_bool_mask])

        for sample_idx in range(2, X.shape[0] - 1):
            left_bool_mask, right_bool_mask = self.get_boolean_masks(
                X, sample_idx, feature_idx
            )
            left_sse = self.sum_of_squares(y[left_bool_mask])
            right_sse = self.sum_of_squares(y[right_bool_mask])

            if left_sse + right_sse < best_left_sse + best_right_sse:
                best_sample_idx = sample_idx
                best_left_sse = left_sse
                best_right_sse = right_sse

        return (best_sample_idx, best_left_sse, best_right_sse)

    def can_add_node(self, num_of_leaves, node_depth):
        """
        This function will be used to help determine if a node should
        be a leaf node. It specifically determines if the BST will not
        exceed the hyperparameters of the BST, if any were given.

        @args:
        - num_of_leaves: the number of leaves in the tree.
        - node_depth: the depth of a node in the BST.
        """
        return (
            (self.leaf_size != None and num_of_leaves >= self.leaf_size)
            or 
            (self.max_height != None and self.max_height == node_depth)
        )


    def sum_of_squares(self, y):
        """
        This function computes the sum of squares.

        @args:
        - y: an array of labels
        """
        return y.shape[0] *  np.var(y)



