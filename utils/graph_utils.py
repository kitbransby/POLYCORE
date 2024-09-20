import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.nn.conv import MessagePassing

# Pooling from COMA: https://github.com/pixelite1201/pytorch_coma/blob/master/layers.py
class Pool(MessagePassing):
    def __init__(self):
        # source_to_target is the default value for flow, but is specified here for explicitness
        super(Pool, self).__init__(flow='source_to_target')

    def forward(self, x, pool_mat,  dtype=None):
        #print(x.shape, pool_mat.shape)
        #print('pool mat: ', pool_mat.is_sparse)
        pool_mat = pool_mat.transpose(0, 1)
        out = self.propagate(edge_index=pool_mat._indices(), x=x, norm=pool_mat._values(), size=pool_mat.size())
        return out

    def message(self, x_j, norm):
        return norm.view(1, -1, 1) * x_j

def scipy_to_torch_sparse(scp_matrix):
    values = scp_matrix.data
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = scp_matrix.shape
    sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape)) #.to_dense()
    return sparse_tensor

def IVUS_Graph(N=[8, 16, 32, 64], I=2, inter=False, include_edge_type=False, include_node_type=False):
    A_list = []
    Up_list = []
    Down_list = []
    Edge_list = []
    Edge_type_list = []
    Node_type_list = []

    for n in N:
        A_mat, A_intra, A_inter = Adj(I, n, inter)
        
        Edge = Get_Edge_List(A_intra)
        Edge = torch.from_numpy(Edge).long().to('cuda:0')
        Edge_list.append(Edge)
        
        A_mat = sp.csc_matrix(A_mat).tocoo()
        A_mat = scipy_to_torch_sparse(A_mat).to('cuda:0')
        A_list.append(A_mat)
        
        Edge_types = torch.zeros((A_mat._indices().shape[1]), dtype=torch.int32).to('cuda:0') # [num edges, edge_dim]
        
        A_intra *= 1
        A_inter *= 2
        A_class = A_intra + A_inter # 0 where no edge, 1 where intra edge, 2 where inter edge
        all_edges = np.stack(np.where(A_class != 0))
        
        for i in range(all_edges.shape[1]):
            edge = all_edges[:,i]
            edge_cls = A_class[edge[0], edge[1]]
            Edge_types[i] = edge_cls - 1
        Edge_type_list.append(Edge_types)

        node_type = torch.zeros(n*2).to('cuda:0')
        node_type[:n] = 0
        node_type[n:] = 1
        Node_type_list.append(node_type.long())
        
        up_mat = Up(I, n * 2)
        up_mat = sp.csc_matrix(up_mat).tocoo()
        up_mat = scipy_to_torch_sparse(up_mat).to('cuda:0')
        Up_list.append(up_mat)

        down_mat = Down(I, n)
        down_mat = sp.csc_matrix(down_mat).tocoo()
        down_mat = scipy_to_torch_sparse(down_mat).to('cuda:0')
        Down_list.append(down_mat)

    output = [A_list, Up_list, Down_list, Edge_list]
        
    if include_edge_type:
        output.append(Edge_type_list)
    if include_node_type:
        output.append(Node_type_list)

    return output

def Get_Edge_List(adj):
    
    edge_list = np.zeros((adj.shape[0],2))
    
    reverse = [0, edge_list.shape[0]//2 -1, edge_list.shape[0]//2, edge_list.shape[0]-1] # first and last of each ring. 

    for i in range(adj.shape[0]):
        pair = np.nonzero(adj[i,:])[0]
        if i in reverse:
            pair = pair[::-1]
        edge_list[i,:] = pair
        
    return edge_list


def Adj(I, N, inter=False):
    A_inter = np.zeros((N * I, N * I))
    A_intra = np.zeros((N * I, N * I))
    for i in range(I):
        adj_instance = Adj_Instance(N)
        A_intra[i * N:(i + 1) * N, i * N:(i + 1) * N] = adj_instance
    if inter==True:
        for i in range(N):
            A_inter[i,i+N] = 1
            A_inter[i+N,i] = 1
            
    A_mat = (A_inter + A_intra).astype(np.int64)
    return A_mat, A_intra.astype(np.int64), A_inter.astype(np.int64)

def Edge_Attr(I, N):
    F = np.zeros((N * I, N * I))
    for i in range(N):
        F[i,i+N] = 1
        F[i+N,i] = 1
    

def Adj_Instance(N):
    A = np.zeros((N, N))
    for i in range(N - 1):
        if i == 0:
            A[i, i + 1] = 1
            A[i + 1, i] = 1
            A[N - 1, i] = 1
            A[i, N - 1] = 1
        else:
            A[i, i + 1] = 1
            A[i + 1, i] = 1
            A[i - 1, i] = 1
            A[i, i - 1] = 1
    return A


def Up(I, N):
    N2 = int(np.ceil(N / 2))
    UP = np.zeros((N * I, N2 * I))

    for i in range(I):
        up_instance = Up_Instance(N)

        UP[i * N:(i + 1) * N, i * N2:(i + 1) * N2] = up_instance

    return UP

def Up_Instance(N):
    N2 = int(np.ceil(N / 2))
    UP = np.zeros([N, N2])

    for i in range(N):

        if i % 2 == 0:
            UP[i, i // 2] = 1

        elif i == (N - 1):
            UP[i, i // 2] = 1 / 2
            UP[i, 0] = 1 / 2

        else:
            UP[i, i // 2] = 1 / 2
            UP[i, i // 2 + 1] = 1 / 2

    return UP


def Down_Instance(N):
    N2 = int(np.ceil(N / 2))
    DOWN = np.zeros([N2, N])

    for i in range(N):

        if i % 2 == 0:
            DOWN[i // 2, i] = 1

    return DOWN

def Down(I, N):
    N2 = int(np.ceil(N / 2))
    DOWN = np.zeros((N2 * I, N * I))

    for i in range(I):
        down_instance = Down_Instance(N)

        DOWN[i * N2:(i + 1) * N2, i * N:(i + 1) * N] = down_instance

    return DOWN


