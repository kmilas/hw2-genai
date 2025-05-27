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
        # N: samples  D: random variables 
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
        
        self.undirected = mst + mst.T
        self.order, predecessors = breadth_first_order(self.undirected, self.root,
                                       directed=False,
                                       return_predecessors=True)

        self.tree = predecessors.tolist() 
        self.tree[self.root] = -1

        self.log_params = np.zeros((self.D,2,2))

        for i in range(self.D):
            parent = self.tree[i]

            if parent<0:
                # For the root only marginal probability
                self.log_params[i,0,:] = np.log(self.margins[:, i])
                self.log_params[i,1,:] = np.log(self.margins[:, i])

            else:
                for j in [0,1]:
                    for k in [0,1]:
                        # Conditional Probabilities
                        self.log_params[i,j,k] = np.log(self.joint_prob(parent, i)[j, k]) - np.log(self.margins[j, parent])
        
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
                joint[x_val, y_val] = (count + self.alpha) / (self.N + 4*self.alpha)

        return joint
    
    
    def mutual_information(self, X, Y):
        joint = self.joint_prob(X, Y)
        px = self.margins[:, X]
        py = self.margins[:, Y]
        mi = 0.0

        for x in [0, 1]:
            for y in [0, 1]:
                p_xy = joint[x, y]
                if p_xy > 0:
                    mi += p_xy * (np.log(p_xy) - np.log(px[x]) - np.log(py[y]))
        return mi

    def get_tree(self):
        return self.tree

    def get_log_params(self):
        return self.log_params

    def log_prob(self, x, exhaustive: bool = False):
        n = x.shape[0]
        lg = np.zeros((n, 1))

        if exhaustive:
            # Whole joint distribution 
            n_options = 2 ** self.D
            joint_distrib = np.zeros((n_options, self.D+1))
            for i in range(n_options):
                x_i = np.binary_repr(i, width=self.D)
                value = 0.
                for j, x_i_j in enumerate(x_i):
                    parent = self.tree[j]
                    if parent < 0:  # root
                        value += self.log_params[j, 0, int(x_i_j)]
                    else:
                        value += self.log_params[j, int(x_i[parent]), int(x_i_j)]
                joint_distrib[i] = np.array(list(map(int, x_i)) + [value])

            # Sanity check:
            # print(logsumexp(joint_distrib[:, -1]))

            for i in range(n):
                to_sum = []
                factors = []

                # Convert nan values to all possible values (0 and 1)
                n_fill_combinations = 2 ** np.sum(np.isnan(x[i]))
                for j in range(n_fill_combinations):
                    x_i = np.copy(x[i])
                    idx = 0
                    for k in range(len(x_i)):
                        if np.isnan(x_i[k]):
                            x_i[k] = (j >> idx) & 1
                            idx += 1
                    to_sum.append(x_i)

                for j in range(n_options):
                    for k in to_sum:
                        if np.array_equal(joint_distrib[j, :-1], k):
                            factors.append(joint_distrib[j, -1])
                lg[i] = logsumexp(factors)
        else:
            pass

        return lg

    def sample(self, n_samples: int):
        samples = np.zeros((n_samples,self.D), dtype=int)

        for t in range(n_samples):
            # sample root
            probs = np.exp(self.log_params[self.root,0,:])
            x = np.zeros(self.D, dtype=int)
            x[self.root] = np.random.choice([0,1], p=probs)

            #order = breadth_first_order(None, self.root, directed=False)[0]
            for i in self.order:
                if i==self.root: continue
                parent = self.tree[i]
                p = np.exp(self.log_params[i, x[parent], :])

                x[i] = np.random.choice([0,1], p=p)
            samples[t] = x
        
        return samples
     

train_data = load_dataset(dir, train_file)

clt = BinaryCLT(data=train_data, root=0)

print(clt.tree)
print(clt.get_log_params())
print(clt.sample(n_samples=3))
x = np.array([
    [np.nan, 0,0,1,0,1,1,1,1,1,0,1,1,0,0,1],
    [0,0,np.nan,1,0,np.nan,1,1,np.nan,1,0,1,1,0,0,1]
])
print(clt.log_prob(x, exhaustive=True))