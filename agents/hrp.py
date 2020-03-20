import numpy as np 
import pandas as pd 
import scipy.cluster.hierarchy as sch 
import matplotlib.pyplot as plt 
from poptim.agents.base import Agent 
from collections import deque 

class HRP:

    def __init__(self, sigma, corr_mat):

        self.sigma = sigma 
        self.corr_mat = corr_mat 

    def tree_clustering(self):
        dist = ((1. - self.corr_mat)*0.5)**.5
        return sch.linkage(dist, 'single')

    def get_quasi_diag(self, link):
        link = link.astype(int)
        sort_ix = pd.Series([link[-1,0], link[-1,1]])
        num_items = link[-1, 3]
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0]*2, 2)
            df_0 = sort_ix[sort_ix >= num_items]
            i = df_0.index
            j = df_0.values - num_items
            sort_ix[i] = link[j,0]
            df_0 = pd.Series(link[j, 1], index=i+1)
            sort_ix = sort_ix.append(df_0)
            sort_ix = sort_ix.sort_index()
            sort_ix.index = range(sort_ix.shape[0])
        return sort_ix.tolist()

    def get_recursive_bisection(self, sort_ix):
        w = pd.Series(1, index=sort_ix)
        c_items = [sort_ix]
        while len(c_items)>0:
            c_items = [i[j:k] for i in c_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
            for i in range(0, len(c_items), 2):
                c_items_0 = c_items[i]
                c_items_1 = c_items[i+1]
                c_var_0 = self.get_cluster_var(c_items_0)
                c_var_1 = self.get_cluster_var(c_items_1)
                alpha = 1. - (c_var_0 / (c_var_0 + c_var_1))
                w[c_items_0] *= alpha 
                w[c_items_1] *= 1-alpha 
        return w
    
    def get_cluster_var(self,c_items):
        cov_ = self.sigma.loc[c_items, c_items]
        w_ = self.get_ivp(cov_).reshape(-1,1)
        c_var = np.dot(np.dot(w_.T, cov_), w_)[0,0]
        return c_var 

    def get_ivp(self, cov):
        ivp = 1. / np.diag(cov)
        ivp /= ivp.sum()
        return ivp 
    
    def get_weights(self):

        link = self.tree_clustering()
        sort_ix = self.get_quasi_diag(link)
        sort_ix = self.corr_mat.index[sort_ix].tolist()
        w = self.get_recursive_bisection(sort_ix)
        return w.values
        #return w.values.reshape(-1,1)

class HRPAgent(Agent):

    _id = 'hrp'

    def __init__(self, action_space, window=50):
        
        self.action_space = action_space 
        self.memory = deque(maxlen=window)
        self.w = self.action_space.sample()

    def observe(self, observation, action, reward, done, next_observation):

        self.memory.append(observation['returns'].values)
    
    def act(self, observation):

        memory = np.array(self.memory)

        M = len(observation)

        if len(self.memory) != self.memory.maxlen:
            return self.action_space.sample()
        else:
            sigma = pd.DataFrame(np.cov(memory.T))
            corr = pd.DataFrame(np.corrcoef(memory.T))
        
        hrp_algo = HRP(sigma, corr)

        w = hrp_algo.get_weights()

        if np.any(w < 0):
            w = w + np.abs(w.min())
        
        self.w = w / w.sum()

        return self.w 

def plot_corr_matrix(corr, labels=None, save=False, path=None):
    if labels is None: 
        labels = []
    plt.pcolor(corr)
    plt.colorbar()
    plt.yticks(np.arange(.5, corr.shape[0] + .5), labels)
    plt.xticks(np.arange(.5, corr.shape[0] + .5), labels)
    if save:
        plt.savefig(path)
    plt.show()

def generate_data(n_obs, size_0, size_1, sigma_1):
    import random
    random.seed(12345)
    np.random.seed(seed=12345)
    x = np.random.normal(0,1,size=(n_obs, size_0))
    cols = [random.randint(0, size_0 - 1) for i in range(size_1)]
    y = x[:, cols] + np.random.normal(0, sigma_1, size=(n_obs, len(cols)))
    x = np.append(x, y, axis=1)
    x = pd.DataFrame(x, columns = range(1, x.shape[1]+ 1))
    return x, cols

def main():
    n_obs, size_0, size_1, sigma_1 = 10000, 5, 5, .25
    x, cols = generate_data(n_obs, size_0, size_1, sigma_1)
    print([(j+1, size_0+i) for i, j in enumerate(cols,1)])
    cov, corr = x.cov(), x.corr()
    plot_corr_matrix(corr, labels=corr.columns)
    hrp = HRP(cov, corr)
    w = hrp.get_weights()
    print(w)
    return w

if __name__ == '__main__':

    main()
