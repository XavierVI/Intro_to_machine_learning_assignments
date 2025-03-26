import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Node:
    def __init__(self, left=None, right=None):
        """
        @args:
        - condition_value: the
        """
        self.left = left
        self.right = right
        self.feature = None
        self.condition_value = None
        self.pred_value = None

    def eval_condition(self, x):
        return x[self.feature] >= self.condition_value

    def is_leaf(self):
        return self.left == None and self.right == None

    def set_left_child(self, node):
        self.left = node

    def set_right_child(self, node):
        self.right = node


class BST:
    """
    This class is a binary search tree.
    """

    def __init__(self):
        self.root_node = Node()

    def get_height(self, node):
        if node == None:
            return 0

        return 1 + max(
            self.get_height(node.left),
            self.get_height(node.right)
        )

    def get_num_of_leaves(self):
        num_of_leaves = 0
        node_stack = []
        node_stack.append(self.root_node)

        while len(node_stack) > 0:
            node = node_stack.pop()

            if node.left != None:
                node_stack.append(node.left)

            if node.right != None:
                node_stack.append(node.right)

            if node.left == None and node.right == None:
                num_of_leaves += 1

        return num_of_leaves

    def plot_tree(self):
        """
        """
        fig, ax = plt.subplots(figsize=(18, 18   ))
        ax.set_axis_off()

        node_positions = {}
        self._assign_positions(self.root_node, 0, 0, node_positions, level_spacing= 0.5)

        self._draw_tree(ax, self.root_node, node_positions)

        plt.title("Regression Tree", fontsize=14)
        plt.show()

    def _assign_positions(self, node, x, y, node_positions, level_spacing):
        """"""
        if node:
            node_positions[node] = (x, y)
            if node.left:
                self._assign_positions(
                    node.left, x - level_spacing, y - 1, node_positions, level_spacing * 0.6)
            if node.right:
                self._assign_positions(
                    node.right, x + level_spacing, y - 1, node_positions, level_spacing * 0.6)

    def _draw_tree(self, ax, node, node_positions):
        """"""
        if node:
            x, y = node_positions[node]
            if node.pred_value != None:
                ax.text(x, y, str(f'y == {node.pred_value:0.2f}'), ha="center", va="center", fontsize=10,
                        bbox=dict(facecolor='lightblue', edgecolor='black', boxstyle='round,pad=1.0'))

            elif node.condition_value != None:
                ax.text(x, y, str(f'x[{node.feature}] >= {node.condition_value:0.2f}'), ha="center", va="center",
                        fontsize=10,
                        bbox=dict(facecolor='lightblue', edgecolor='black', boxstyle='round,pad=1.0'))

            else:
                ax.text(x, y, str('None'), ha="center", va="center", fontsize=10,
                        bbox=dict(facecolor='lightblue', edgecolor='black', boxstyle='round,pad=1.0'))

            if node.left:
                lx, ly = node_positions[node.left]
                ax.plot([x, lx], [y, ly], 'k-', lw=1.0)
                self._draw_tree(ax, node.left, node_positions)

            if node.right:
                rx, ry = node_positions[node.right]
                ax.plot([x, rx], [y, ry], 'k-', lw=1.0)
                self._draw_tree(ax, node.right, node_positions)


class RegressionTree:
    def __init__(self, max_height: int = None, leaf_size: int = None):
        self.max_height = max_height
        self.leaf_size = leaf_size
        self.height = 0
        self.bst = BST()

    def decision_path(self, x):
        node = self.bst.root_node
        # iterate until we get a leaf node
        while not node.is_leaf():
            if node.eval_condition(x):
                print(f'x[{node.feature}] >= {node.condition_value}')
                node = node.right

            else:
                print(f'x[{node.feature}] < {node.condition_value}')
                node = node.left

        print(f'x == {node.pred_value}')

    def predict(self, x):
        node = self.bst.root_node

        # iterate until we get a leaf node
        while not node.is_leaf():
            if node.eval_condition(x):
                node = node.right

            else:
                node = node.left

        return node.pred_value

    def fit(self, X, y):
        if len(X.shape) > 1:
            num_of_features = X.shape[1]
        else:
            num_of_features = 1
            X = X.reshape(-1, 1)

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
            if X_local.shape[0] == 1 or impurity == 0 or \
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
                    self.get_best_split(X_local, y_local, feature_idx)

                if best_left_sse + best_right_sse > left_sse + right_sse:
                    best_sample_idx = sample_idx
                    best_left_sse = left_sse
                    best_right_sse = right_sse
                    best_feature_idx = feature_idx

            # use the index which splits the data at X[sample_idx, feature_idx]
            # to set the condition property of the node
            curr_node.condition_value = X_local[best_sample_idx, best_feature_idx]
            curr_node.feature = best_feature_idx

            # create a node and set it as the left child of curr_node
            curr_node.left = Node()
            left_split = (X[:, best_feature_idx] < \
                          X_local[best_sample_idx, best_feature_idx]) \
                         & node_dict['mask']

            # if the dataset will not be empty
            if np.any(left_split):
                # add left to the queue
                node_queue.append({
                    'node': curr_node.left,
                    'depth': depth + 1,
                    'mask': left_split,
                    'impurity': best_left_sse,
                })

            else:
                curr_node.left.pred_value = np.mean(y_local, axis=0)

            # create a node and set it as the right child of curr_node
            curr_node.right = Node()
            right_split = (X[:, best_feature_idx] > \
                           X_local[best_sample_idx, best_feature_idx]) \
                          & node_dict['mask']

            # if the dataset will not be empty
            if np.any(right_split):
                # add right to the queue
                node_queue.append({
                    'node': curr_node.right,
                    'depth': depth + 1,
                    'mask': right_split,
                    'impurity': best_right_sse,
                })

            else:
                curr_node.right.pred_value = np.mean(y_local, axis=0)

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
        return y.shape[0] * np.var(y)

df = pd.read_csv('Task2Data.csv', header=None, encoding='utf-8')

X = df.iloc[1:,1:3].values
X = df.drop(df.columns[1], axis=0)
X = np.array(X)
y = df.iloc[1:,1:3].values
y_train = []
x_train = []
y_test = []
x_test = []

for row in y:
    y_value = row[0]  # y value (column index 1)
    binary_value = row[1]  # binary indicator (column index 2)
    if int(binary_value) == 0:
        y_train.append(y_value)
    else:
        y_test.append(y_value)

for row in X:
    x_value = row[0]  # x value (column index 1)
    binary_value = row[2]  # binary indicator (column index 2)
    if int(binary_value) == 0:
        x_train.append(x_value)
    else:
        x_test.append(x_value)

y_train = np.array(y_train)
x_train = np.array(x_train)

RT = RegressionTree(max_height= None, leaf_size= None)

RegressionTree.fit(RT, X = x_train, y = y_train)

BST.plot_tree(RT.bst)
