from scipy.sparse.csgraph import minimum_spanning_tree, breadth_first_order
from scipy.special import logsumexp
import numpy as np
import itertools
import csv
import os

dir = 'nltcs'
train_file = 'nltcs.train.data'


def load_dataset(dir, filename):

    path = os.path.join(dir,filename)

    with open(path , "r") as file :
        reader = csv.reader( file , delimiter=',')
        dataset = np.array(list(reader)).astype(float)
    
    return dataset


class BinaryCLT:
    def __init__(self, data, root: int = None, alpha: float = 0.01):

        self.data = data
        self.alpha = alpha
        # N: samples x D: random variables 
        self.N, self.D = data.shape
        if root is None:
            root = np.random.randint(self.D)
        self.root = root

        # Mutual information
        self.mi = np.zeros((self.D, self.D))
        self.margins = self.margin_prob()
        
        for X, Y in itertools.combinations(range(self.D), 2):
            mi_val = self.mutual_information(X, Y)
            self.mi[X, Y] = mi_val
            self.mi[Y, X] = mi_val

        mst = minimum_spanning_tree(-self.mi)
        #edges = np.array(mst.nonzero()).T.tolist() + np.array(mst.T.nonzero()).T.tolist()
        
        undirected = mst + mst.T
        _, predecessors = breadth_first_order(undirected, self.root,
                                       directed=False,
                                       return_predecessors=True)

        self.tree = predecessors.tolist()
        self.tree[self.root] = -1

    
    def margin_prob(self):
        # calculate P(X=x) marginal Probabilities for all the RVs of the dataset 
        count_ones = np.sum(self.data, axis=0) + 2*self.alpha
        sample_size = self.N + 4*self.alpha
        probs_one = count_ones/sample_size
        probs_zero = 1. - probs_one

        return np.vstack((probs_zero, probs_one))

    def joint_prob(self, X, Y):
        # calculate P(X=x, Y=y) joint probability
        joint = np.zeros((2,2))

        for x_val in [0,1]:
            for y_val in [0,1]:
                count = np.sum((self.data[:, X] == x_val) & (self.data[:, Y] == y_val))
                joint[ y_val, x_val] = (count + self.alpha) / (self.N + 4*self.alpha)

        return joint
    
    def mutual_information(self, X, Y):
        joint = self.joint_prob(X, Y)
        p_x = self.margins[:, X]
        p_y = self.margins[:, Y]
        mi = 0.0

        for x in [0, 1]:
            for y in [0, 1]:
                p_xy = joint[y, x]
                if p_xy > 0:
                    mi += p_xy * np.log(p_xy / (p_x[x] * p_y[y]))
        return mi

    def get_tree(self):
        self.tree

    def get_log_params(self):
        pass

    def log_prob(self, x, exhaustive: bool = False):
        pass

    def sample(self, n_samples: int):
        pass
     

train_data = load_dataset(dir, train_file)

clt = BinaryCLT(data=train_data, root=10)

print(clt.tree)

