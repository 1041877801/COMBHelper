import torch
from torch_geometric.data import Data, InMemoryDataset
import os
import random
import networkx as nx
import numpy as np
import pickle


class BA_Train(InMemoryDataset):
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None
        ):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return ['']
    
    @property
    def processed_file_names(self):
        return 'BA1k_MVC_1024.pt' # for MVC
        # return 'BA1k_MIS_1024.pt' # for MIS
    
    def download(self):
        pass
    
    def process(self):
        if os.path.exists(os.path.join(self.raw_dir, 'BA1k.G')):
            print('G exists')
            G = pickle.load(open(os.path.join(self.raw_dir, 'BA1k.G'), 'rb'))
        else:
            G = nx.barabasi_albert_graph(n=1000, m=4, seed=42)
            
            f = open(os.path.join(self.raw_dir, 'BA1k.G'), 'wb')
            pickle.dump(G, f)
        
        edge_index = []
        for source, target in nx.edges(G):
            edge_index.append([source, target])
            edge_index.append([target, source])
        edge_index = torch.LongTensor(edge_index)
        
        num_nodes = nx.number_of_nodes(G)
        x = np.zeros((num_nodes, 1))
        for i in range(num_nodes):
            x[i] = nx.degree(G, i)
            
        FFM_matrix = torch.load(os.path.join(self.raw_dir, 'FFM-1->1024.pt'))
        x_proj = (2. * np.pi * x) @ FFM_matrix.T
        x = np.concatenate([np.sin(x_proj), np.cos(x_proj)], axis=1)
        x = torch.FloatTensor(x)
        
        y = [0] * num_nodes
        solution_file = open(os.path.join(self.raw_dir, 'BA1k_MVC.LP_solution'), 'rb') # for MVC
        # solution_file = open(os.path.join(self.raw_dir, 'BA1k_MIS.LP_solution'), 'rb') # for MIS
        solution_set = pickle.load(solution_file)
        for node in solution_set:
            y[node] = 1
        y = torch.LongTensor(y)
        
        data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y, num_nodes=num_nodes)
        
        train_mask = [False] * num_nodes
        val_mask = [False] * num_nodes
        weight = torch.ones(num_nodes)

        for idx in range(num_nodes):
            rand_int = random.randint(1, 100)
            if 1 <= rand_int <= 50:
                train_mask[idx] = True
                deg = nx.degree(G, idx)
                weight[idx] = deg # for MVC
                # weight[idx] = 1 / deg # for MIS
            elif 51 <= rand_int <= 100:
                val_mask[idx] = True
        
        train_mask = torch.tensor(train_mask)
        val_mask = torch.tensor(val_mask)
        
        weight = weight[train_mask]
        weight = weight / weight.sum()
        
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.weight = weight
        
        torch.save(self.collate([data]), self.processed_paths[0])
        
        
class BA_Test(InMemoryDataset):
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None
        ):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return ['']
    
    @property
    def processed_file_names(self):
        return 'BA5k_MVC_1024.pt' # for MVC
        # return 'BA5k_MIS_1024.pt' # for MIS
    
    def download(self):
        pass
    
    def process(self):
        if os.path.exists(os.path.join(self.raw_dir, 'BA5k.G')):
            print('G exists')
            G = pickle.load(open(os.path.join(self.raw_dir, 'BA5k.G'), 'rb'))
        else:
            G = nx.barabasi_albert_graph(n=5000, m=4, seed=42)
            
            f = open(os.path.join(self.raw_dir, 'BA5k.G'), 'wb')
            pickle.dump(G, f)
        
        edge_index = []
        for source, target in nx.edges(G):
            edge_index.append([source, target])
            edge_index.append([target, source])
        edge_index = torch.LongTensor(edge_index)
        
        num_nodes = nx.number_of_nodes(G)
        x = np.zeros((num_nodes, 1))
        for i in range(num_nodes):
            x[i] = nx.degree(G, i)
            
        FFM_matrix = torch.load(os.path.join(self.raw_dir, 'FFM-1->1024.pt'))
        x_proj = (2. * np.pi * x) @ FFM_matrix.T
        x = np.concatenate([np.sin(x_proj), np.cos(x_proj)], axis=1)
        x = torch.FloatTensor(x)
        
        y = [0] * num_nodes
        solution_file = open(os.path.join(self.raw_dir, 'BA5k_MVC.LP_solution'), 'rb') # for MVC
        # solution_file = open(os.path.join(self.raw_dir, 'BA5k_MIS.LP_solution'), 'rb') # for MIS
        solution_set = pickle.load(solution_file)
        for node in solution_set:
            y[node] = 1
        y = torch.LongTensor(y)
        
        data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y, num_nodes=num_nodes)
        
        torch.save(self.collate([data]), self.processed_paths[0])