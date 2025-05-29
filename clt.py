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
            joint_distrib = np.zeros((2 ** self.D, self.D+1))
            for i in range(2 ** self.D):
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

            data_slice = joint_distrib[:, :-1]
            values_to_extract = joint_distrib[:, -1]
            
            for i in range(n):
                nan_mask_in_x = np.isnan(x[i])
                current_factors = np.array([])
                non_nan_mask_in_x = ~nan_mask_in_x
                num_nans = int(np.sum(nan_mask_in_x))

                if np.any(non_nan_mask_in_x):
                    x_defined_values = x[i][non_nan_mask_in_x]
                    data_slice_defined_parts = data_slice[:, non_nan_mask_in_x]
                
                    initial_match_mask = np.all(data_slice_defined_parts == x_defined_values, axis=1)
                else:
                    initial_match_mask = np.full(data_slice.shape[0], True, dtype=bool)

                candidate_row_indices = np.where(initial_match_mask)[0]

                if num_nans == 0:
                    current_factors = values_to_extract[candidate_row_indices]
                else:
                    data_slice_at_x_nan_positions = data_slice[candidate_row_indices][:, nan_mask_in_x]
                    valid_fill_values_mask = np.all(
                        (data_slice_at_x_nan_positions == 0) | (data_slice_at_x_nan_positions == 1),
                        axis=1
                    )
                    final_matching_indices_in_data_slice = candidate_row_indices[valid_fill_values_mask]
                    
                    if final_matching_indices_in_data_slice.size > 0:
                        current_factors = values_to_extract[final_matching_indices_in_data_slice]

                lg[i] = logsumexp(current_factors)
        else:
            for i in range(n):
                messages = np.full((self.D, 2), np.nan)
                for j in reversed(self.order):
                    parent = self.tree[j]
                    if parent >= 0:
                        if np.isnan(x[i, parent]):
                            parent_values = [0, 1]
                        else:
                            parent_values = [int(x[i, parent])]
                        
                        if np.isnan(x[i, j]):
                            children_values = [0, 1]
                        else:
                            children_values = [int(x[i, j])]
                        for k in parent_values:
                            values = []
                            for l in children_values:
                                if np.isnan(messages[j, l]):
                                    values.append(self.log_params[j, k, l])
                                else:
                                    values.append(messages[j, l] + self.log_params[j, k, l])
                            if np.isnan(messages[parent, k]):
                                messages[parent, k] = logsumexp(values)
                            else:
                                messages[parent, k] += logsumexp(values)
                    else:
                        if np.isnan(x[i, j]):
                            possible_values = [0, 1]
                        else:
                            possible_values = [int(x[i, j])]
                        for k in possible_values:
                            if np.isnan(messages[j, k]):
                                messages[j, k] = self.log_params[j, 0, k]
                            else:
                                messages[j, k] += self.log_params[j, 0, k]

                # Sum over all messages for the root (removing the nan values)
                lg[i] = logsumexp(messages[self.root, ~np.isnan(messages[self.root, :])])

        return lg

    def sample_old(self, n_samples: int):
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
    
    def sample(self, n_samples: int) -> np.ndarray:
        D = self.D
        out = np.zeros((n_samples, D), dtype=int)

        children = [[] for _ in range(D)]
        for v, p in enumerate(self.tree):
            if p != -1:
                children[p].append(v)

        for s in range(n_samples):
            root_probs = np.exp(self.log_params[self.root, 0, :])
            out[s, self.root] = np.random.choice([0, 1], p=root_probs)

            frontier = [self.root]
            while frontier:
                u = frontier.pop()
                xu = out[s, u]
                for v in children[u]:
                    probs = np.exp(self.log_params[v, xu, :])
                    out[s, v] = np.random.choice([0, 1], p=probs)
                    frontier.append(v)

        return out
     

train_data = load_dataset(dir, train_file)

clt = BinaryCLT(data=train_data, root=0)
print(clt.tree)
print(clt.get_log_params())
print(clt.sample(n_samples=3))

x = np.array([
    [0,np.nan,0,1,0,1,1,1,1,1,0,1,1,0,0,1],
    [0,0,np.nan,1,0,np.nan,1,1,np.nan,1,0,1,1,0,0,1],
    [0,0,np.nan,1,0,1,1,1,np.nan,1,0,1,1,0,0,1]
])
print(clt.log_prob(x, exhaustive=False))
print(clt.log_prob(x, exhaustive=True))