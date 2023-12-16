import torch
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_gz, extract_zip
import os
import random
import networkx as nx
import numpy as np
import pickle


class Cora(InMemoryDataset):
    url = 'https://github.com/kimiyoung/planetoid/blob/master/data/ind.cora.graph'
    
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
        return ['ind.cora.graph']
    
    @property
    def processed_file_names(self):
        return 'Cora_MVC_1024.pt' # for MVC
        # return 'Cora_MIS_1024.pt' # for MIS
    
    def download(self):
        download_url(self.url, self.raw_dir)
    
    def process(self):
        if os.path.exists(os.path.join(self.raw_dir, 'Cora.G')):
            print('G exists!')
            G = pickle.load(open(os.path.join(self.raw_dir, 'Cora.G'), 'rb'))
        else:
            f = open(os.path.join(self.raw_dir, 'ind.cora.graph'), 'rb')
            G_dict = pickle.load(f)
            G = nx.from_dict_of_lists(G_dict)
            
            f = open(os.path.join(self.raw_dir, 'Cora.G'), 'wb')
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
        solution_file = open(os.path.join(self.raw_dir, 'Cora_MVC.LP_solution'), 'rb') # for MVC
        # solution_file = open(os.path.join(self.raw_dir, 'Cora_MIS.LP_solution'), 'rb') # for MIS
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
        

class Alpha(InMemoryDataset):
    url = 'https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz'
    
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
        return ['soc-sign-bitcoinalpha.csv']
    
    @property
    def processed_file_names(self):
        return 'Alpha_MVC_1024.pt' # for MVC
        # return 'Alpha_MIS_1024.pt' # for MIS
    
    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_gz(path, self.raw_dir)
        
    def process(self):
        node_map = {}
        with open(os.path.join(self.raw_dir, 'soc-sign-bitcoinalpha.csv')) as f:
            idx = 0
            for line in f:
                info = line.strip().split(',')
                for node in [info[0], info[1]]:
                    if node not in node_map:
                        node_map[node] = idx
                        idx += 1
        
        edge_index = set()
        with open(os.path.join(self.raw_dir, 'soc-sign-bitcoinalpha.csv')) as f:
            for idx, line in enumerate(f):
                info = line.strip().split(',')
                source, target = info[0], info[1]
                source, target = node_map[source], node_map[target]
                if source != target:
                    edge_index.add((source, target))
                    edge_index.add((target, source))

        G = nx.from_edgelist(edge_index)
        f = open(os.path.join(self.raw_dir, 'Alpha.G'), 'wb')
        pickle.dump(G, f)
            
        num_nodes = nx.number_of_nodes(G)
        
        x = np.zeros((num_nodes, 1))
        for i in range(num_nodes):
            x[i] = nx.degree(G, i)
        
        ffm_matrix = torch.load(os.path.join(self.raw_dir, 'FFM-1->1024.pt'))
        x_proj = (2. * np.pi * x) @ ffm_matrix.T
        x = np.concatenate([np.sin(x_proj), np.cos(x_proj)], axis=1)
        x = torch.FloatTensor(x)
        
        y = [0] * num_nodes
        f = open(os.path.join(self.raw_dir, 'Alpha_MVC.LP_solution'), 'rb') # for MVC
        # f = open(os.path.join(self.raw_dir, 'Alpha_MIS.LP_solution'), 'rb') # for MIS
        solution_set = pickle.load(f)
        for node in solution_set:
            y[node] = 1
        y = torch.LongTensor(y)
        
        edge_index = torch.LongTensor(list(edge_index))
        
        data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y, num_nodes=num_nodes)
        
        torch.save(self.collate([data]), self.processed_paths[0])
        
        
class OTC(InMemoryDataset):
    url = 'https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz'
    
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
        return ['soc-sign-bitcoinotc.csv']
    
    @property
    def processed_file_names(self):
        return 'OTC_MVC_1024.pt' # for MVC
        # return 'OTC_MIS_1024.pt' # for MIS
    
    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_gz(path, self.raw_dir)
        
    def process(self):
        node_map = {}
        with open(os.path.join(self.raw_dir, 'soc-sign-bitcoinotc.csv')) as f:
            idx = 0
            for line in f:
                info = line.strip().split(',')
                for node in [info[0], info[1]]:
                    if node not in node_map:
                        node_map[node] = idx
                        idx += 1
        
        edge_index = set()
        with open(os.path.join(self.raw_dir, 'soc-sign-bitcoinotc.csv')) as f:
            for idx, line in enumerate(f):
                info = line.strip().split(',')
                source, target = info[0], info[1]
                source, target = node_map[source], node_map[target]
                if source != target:
                    edge_index.add((source, target))
                    edge_index.add((target, source))

        G = nx.from_edgelist(edge_index)
        f = open(os.path.join(self.raw_dir, 'OTC.G'), 'wb')
        pickle.dump(G, f)
        num_nodes = nx.number_of_nodes(G)
        
        x = np.zeros((num_nodes, 1))
        for i in range(num_nodes):
            x[i] = nx.degree(G, i)
        
        ffm_matrix = torch.load(os.path.join(self.raw_dir, 'FFM-1->1024.pt'))
        x_proj = (2. * np.pi * x) @ ffm_matrix.T
        x = np.concatenate([np.sin(x_proj), np.cos(x_proj)], axis=1)
        x = torch.FloatTensor(x)
        
        y = [0] * num_nodes
        f = open(os.path.join(self.raw_dir, 'OTC_MVC.LP_solution'), 'rb') # for MVC
        # f = open(os.path.join(self.raw_dir, 'OTC_MIS.LP_solution'), 'rb') # for MIS
        solution_set = pickle.load(f)
        for node in solution_set:
            y[node] = 1
        y = torch.LongTensor(y)
        
        edge_index = torch.LongTensor(list(edge_index))
        
        data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y, num_nodes=num_nodes)
        
        torch.save(self.collate([data]), self.processed_paths[0])
        

if __name__ == '__main__':
    dataset = OTC(root='./data/OTC')
    data = dataset[0]
    print(data)