import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_scatter import scatter, scatter_min
from torch_geometric.nn import inits
from torch.nn.utils.weight_norm import weight_norm

import math
from math import sqrt
import dgl
from dgl.nn.pytorch import GraphConv

from torch.nn.utils.weight_norm import weight_norm

import numpy as np

from ban import BANLayer
from features import angle_emb, torsion_emb


def swish(x):
    return x * torch.sigmoid(x)


class Linear(torch.nn.Module):
    """
    input[batch_num,nodes_num,in_channels] -> output[batch_num,nodes_num,out_channels]
    """
    # Glorot初始化方法的思想是根据输入和输出的连接数量，以及所使用的激活函数的特性，来初始化权重矩阵。这样可以避免在网络训练初期产生梯度消失或梯度爆炸的问题，有助于加快模型的收敛速度。
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 bias=True,
                 weight_initializer='glorot', 
                 bias_initializer='zeros'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

        assert in_channels > 0
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.in_channels > 0:
            if self.weight_initializer == 'glorot':
                inits.glorot(self.weight)
            elif self.weight_initializer == 'glorot_orthogonal':
                inits.glorot_orthogonal(self.weight, scale=2.0)
            elif self.weight_initializer == 'uniform':
                bound = 1.0 / math.sqrt(self.weight.size(-1))
                torch.nn.init.uniform_(self.weight.data, -bound, bound)
            elif self.weight_initializer == 'kaiming_uniform':
                inits.kaiming_uniform(self.weight, fan=self.in_channels,
                                      a=math.sqrt(5))
            elif self.weight_initializer == 'zeros':
                inits.zeros(self.weight)
            elif self.weight_initializer is None:
                inits.kaiming_uniform(self.weight, fan=self.in_channels,
                                      a=math.sqrt(5))
            else:
                raise RuntimeError(
                    f"Linear layer weight initializer "
                    f"'{self.weight_initializer}' is not supported")

        if self.in_channels > 0 and self.bias is not None:
            if self.bias_initializer == 'zeros':
                inits.zeros(self.bias)
            elif self.bias_initializer is None:
                inits.uniform(self.in_channels, self.bias)
            else:
                raise RuntimeError(
                    f"Linear layer bias initializer "
                    f"'{self.bias_initializer}' is not supported")

    def forward(self, x):
        """"""
        return F.linear(x, self.weight, self.bias)


class TwoLayerLinear(torch.nn.Module):
    """
    input[batch_num,nodes_num,in_channels] 
    -> [batch_num,nodes_num,middle_channels] 
    -> output[batch_num,nodes_num,out_channels]
    """
    def __init__(
            self,
            in_channels,
            middle_channels,
            out_channels,
            bias=False,
            act_bool=False,
    ):
        super(TwoLayerLinear, self).__init__()
        self.lin1 = Linear(in_channels, middle_channels, bias=bias)
        self.lin2 = Linear(middle_channels, out_channels, bias=bias)
        self.act_bool = act_bool

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x):
        x = self.lin1(x)
        if self.act_bool:
            x = swish(x)
        x = self.lin2(x)
        if self.act_bool:
            x = swish(x)
        return x

    
class Interaction(nn.Module):
    """
    This layer combines node and edge features in message passing and updates node
    representations.

    Parameters
    ----------
    node_feats : int
        Size for the input and output node features.
    edge_in_feats : int
        Size for the input edge features.
    hidden_feats : int
        Size for hidden representations.
    """
    def __init__(self, 
                 num_radial, 
                 num_spherical,
                 middle_channels, 
                 hidden_channels, 
                 output_channels, 
                 num_lin, 
                 act=swish):
        super(Interaction, self).__init__()
        
        self.act = act
        # input: node_feature
        self.lin = Linear(hidden_channels, hidden_channels)
        # Transformations of Bessel and spherical basis representations.
        self.lin_feature1 = TwoLayerLinear(num_radial * num_spherical ** 2, 
                                           middle_channels, 
                                           hidden_channels)
        self.lin_feature2 = TwoLayerLinear(num_radial * num_spherical, 
                                           middle_channels, 
                                           hidden_channels)
        # input: x, feature1
        self.conv1 = GraphConv(hidden_channels, 
                               hidden_channels, 
                               norm='none', 
                               weight=True, 
                               bias=True, 
                               activation=None, 
                               allow_zero_in_degree=True)
        # input: x, feature2
        self.conv2 = GraphConv(hidden_channels, 
                               hidden_channels, 
                               norm='none', 
                               weight=True, 
                               bias=True, 
                               activation=None, 
                               allow_zero_in_degree=True)
        # input:h1
        self.lin1 = Linear(hidden_channels, hidden_channels)
        # input:h2
        self.lin2 = Linear(hidden_channels, hidden_channels)
        # input: [h1, h2]
        self.lin_cat = Linear(2 * hidden_channels, hidden_channels)
        # 
        self.lins = nn.ModuleList()
        for _ in range(num_lin):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        # batch normalization
        self.bn_layer = nn.BatchNorm1d(hidden_channels)
        self.final = Linear(hidden_channels, output_channels)
        
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.lin.reset_parameters()
        self.lin_feature1.reset_parameters()
        self.lin_feature2.reset_parameters()
        
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin_cat.reset_parameters()

        for lin in self.lins:
            lin.reset_parameters()

        self.bn_layer.reset_parameters()
        
        self.final.reset_parameters()
        

    def forward(self, g, node_feats, edge_feats1, edge_feats2):
        """Performs message passing and updates node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feats)
            Input node features, V for the number of nodes.
        edge_feats1 : float32 tensor of shape (E, edge_in_feats)
            Input edge features 1, E for the number of edges.
        
        edge_feats2 : float32 tensor of shape (E, edge_in_feats)
            Input edge features 2, E for the number of edges.

        Returns
        -------
        float32 tensor of shape (V, node_feats)
            Updated node representations.
        """
        node_feats = self.act(self.lin(node_feats))
        edge_feats1 = self.lin_feature1(edge_feats1)
        edge_feats2 = self.lin_feature2(edge_feats2)
        
        h1 = self.conv1(g, node_feats, edge_weight=edge_feats1)
        h1 = self.lin1(h1)
        h1 = self.act(h1)
        
        h2 = self.conv2(g, node_feats, edge_weight=edge_feats2)
        h2 = self.lin2(h2)
        h2 = self.act(h2)
        
        h = self.lin_cat(torch.cat([h1, h2], 1))
        
        # 残差网络
        h = h + node_feats
        for lin in self.lins:
            h = self.act(lin(h)) + h
        # batch normalization
        h = self.bn_layer(h)
        h = self.final(h)
        
        return h
        

class FlexibleMol(nn.Module):
    """FlexibleMol.

    FlexibleMol is introduced here.

    Parameters
    ----------
    node_feats : int
        Size for node representations to learn. Default to 64.
    hidden_feats : list of int
        ``hidden_feats[i]`` gives the size of hidden representations for the i-th interaction
        layer. ``len(hidden_feats)`` equals the number of interaction layers.
        Default to ``[64, 64, 64]``.
    num_node_types : int
        Number of node types to embed. Default to 100.
    cutoff : float
        Largest center in RBF expansion. Default to 30.
    gap : float
        Difference between two adjacent centers in RBF expansion. Default to 0.1.
    """
    def __init__(self, 
                 node_channels,
                 num_radial, 
                 num_spherical,
                 cutoff,
                 hidden_channels,
                 middle_channels,
                 num_gnn, 
                 num_lin,
                 num_res, 
                 act):
        super(FlexibleMol, self).__init__()
        
        self.act = act
        self.lin_x = Linear(node_channels, hidden_channels)
        
        self.feature1 = torsion_emb(num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff)
        self.feature2 = angle_emb(num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff)
        
        self.gnn_layers = nn.ModuleList()
        for i in range(num_gnn):
            self.gnn_layers.append(
                Interaction(num_radial, 
                            num_spherical, 
                            middle_channels, 
                            hidden_channels, 
                            hidden_channels, 
                            num_lin, 
                            act))
        
        # self.lins = nn.ModuleList()
        # for _ in range(num_res):
        #     self.lins.append(nn.Linear(hidden_channels, hidden_channels))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.lin_x.reset_parameters()
    
        for layer in self.gnn_layers:
            layer.reset_parameters()
            
        # for lin in self.lins:
        #     lin.reset_parameters()

    def forward(self, g, t):
        """Performs message passing and updates node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_types : int64 tensor of shape (V)
            Node types to embed, V for the number of nodes.
        edge_dists : float32 tensor of shape (E, 1)
            Distances between end nodes of edges, E for the number of edges.

        Returns
        -------
        node_feats : float32 tensor of shape (V, node_feats)
            Updated node representations.
        """
        j, i = g.edges()
        i = i.long()
        j = j.long()
        vecs = g.ndata['pos'][j] -g.ndata['pos'][i]
        dist = vecs.norm(dim=-1)

        num_nodes = g.ndata['pos'].shape[0]
        
        _, argmin0 = scatter_min(dist, i, dim_size=num_nodes)
        argmin0[argmin0 >= len(i)] = 0
        n0 = j[argmin0]
        add = torch.zeros_like(dist).to(dist.device)
        add[argmin0] = 3.0 # self.cutoff
        dist1 = dist + add

        _, argmin1 = scatter_min(dist1, i, dim_size=num_nodes)
        argmin1[argmin1 >= len(i)] = 0
        n1 = j[argmin1]
        # --------------------------------------------------------

        _, argmin0_j = scatter_min(dist, j, dim_size=num_nodes)
        argmin0_j[argmin0_j >= len(j)] = 0
        n0_j = i[argmin0_j]

        add_j = torch.zeros_like(dist).to(dist.device)
        add_j[argmin0_j] = 5.0 # self.cutoff
        dist1_j = dist + add_j

        # i[argmin] = range(0, num_nodes)
        _, argmin1_j = scatter_min(dist1_j, j, dim_size=num_nodes)
        argmin1_j[argmin1_j >= len(j)] = 0
        n1_j = i[argmin1_j]

        # ----------------------------------------------------------

        # n0, n1 for i
        n0 = n0[i]
        n1 = n1[i]

        # n0, n1 for j
        n0_j = n0_j[j]
        n1_j = n1_j[j]

        # tau: (iref, i, j, jref)
        # when compute tau, do not use n0, n0_j as ref for i and j,
        # because if n0 = j, or n0_j = i, the computed tau is zero
        # so if n0 = j, we choose iref = n1
        # if n0_j = i, we choose jref = n1_j
        mask_iref = n0 == j
        iref = torch.clone(n0)
        iref[mask_iref] = n1[mask_iref]
        idx_iref = argmin0[i]
        idx_iref[mask_iref] = argmin1[i][mask_iref]

        mask_jref = n0_j == i
        jref = torch.clone(n0_j)
        jref[mask_jref] = n1_j[mask_jref]
        idx_jref = argmin0_j[j]
        idx_jref[mask_jref] = argmin1_j[j][mask_jref]

        pos_ji, pos_in0, pos_in1, pos_iref, pos_jref_j = (
            vecs,
            vecs[argmin0][i],
            vecs[argmin1][i],
            vecs[idx_iref],
            vecs[idx_jref]
        )

        # Calculate angles.
        a = ((-pos_ji) * pos_in0).sum(dim=-1)
        b = torch.cross(-pos_ji, pos_in0).norm(dim=-1)
        theta = torch.atan2(b, a)
        theta[theta < 0] = theta[theta < 0] + math.pi

        # Calculate torsions.
        dist_ji = pos_ji.pow(2).sum(dim=-1).sqrt()
        plane1 = torch.cross(-pos_ji, pos_in0)
        plane2 = torch.cross(-pos_ji, pos_in1)
        a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
        b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
        phi = torch.atan2(b, a)
        phi[phi < 0] = phi[phi < 0] + math.pi

        # Calculate right torsions.
        plane1 = torch.cross(pos_ji, pos_jref_j)
        plane2 = torch.cross(pos_ji, pos_iref)
        a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
        b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
        tau = torch.atan2(b, a)
        tau[tau < 0] = tau[tau < 0] + math.pi
        
        # get node_features, edge_features
        node_feats = g.ndata['h']
        node_feats = self.act(self.lin_x(node_feats))
        
        edge_feats1 = self.feature1(dist, theta, phi)
        edge_feats2 = self.feature2(dist, tau)
        
        # gnn block
        for gnn in self.gnn_layers:
            node_feats = gnn(g, node_feats, edge_feats1, edge_feats2)
        
        g.ndata['h'] = node_feats
        
        # bondrotable transform
        max_length = 200
        feats = []
        for i, graph in enumerate(dgl.unbatch(g)):
            feat = torch.matmul(t[i], graph.ndata['h'])
            # 这里需要补一个normaliziton
            
            # padding到max_length
            feat = torch.cat([feat, torch.zeros([max_length - feat.size()[0], feat.size()[-1]], dtype=torch.float32)])
            feats.append(feat)
        
        # get substructure_representation
        feats = torch.cat(feats, dim=0)
        
        # for lin in self.lins:
        #     feats = self.act(lin(feats))
            
        feats = feats.view(g.batch_size, -1, feats.size()[-1])
        
        return feats

def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss


def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss


def entropy_logits(linear_output):
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent


class DrugBAN(nn.Module):
    def __init__(self, **config):
        super(DrugBAN, self).__init__()
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        num_filters = config["PROTEIN"]["NUM_FILTERS"]
        kernel_size = config["PROTEIN"]["KERNEL_SIZE"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        drug_padding = config["DRUG"]["PADDING"]
        protein_padding = config["PROTEIN"]["PADDING"]
        out_binary = config["DECODER"]["BINARY"]
        ban_heads = config["BCN"]["HEADS"]
        self.drug_extractor = FlexibleMol(node_channels=drug_in_feats-1, 
                                          num_radial=6, 
                                          num_spherical=3, 
                                          cutoff=3.0, 
                                          hidden_channels=drug_embedding,
                                          middle_channels=256,
                                          num_gnn=3,
                                          num_lin=3, 
                                          num_res=4, 
                                          act=swish)
        self.protein_extractor = ProteinCNN(protein_emb_dim, num_filters, kernel_size, protein_padding)

        self.bcn = weight_norm(
            BANLayer(v_dim=drug_hidden_feats[-1], q_dim=num_filters[-1], h_dim=mlp_in_dim, h_out=ban_heads),
            name='h_mat', dim=None)
        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)

    def forward(self, bg_d, t, v_p, mode="train"):
        v_d = self.drug_extractor(bg_d, t)
        v_p = self.protein_extractor(v_p)
        f, att = self.bcn(v_d, v_p)
        score = self.mlp_classifier(f)
        if mode == "train":
            return v_d, v_p, f, score
        elif mode == "eval":
            return v_d, v_p, score, att


class ProteinCNN(nn.Module):
    def __init__(self, embedding_dim, num_filters, kernel_size, padding=True):
        super(ProteinCNN, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]
        kernels = kernel_size
        self.conv1 = nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=kernels[0])
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=kernels[1])
        self.bn2 = nn.BatchNorm1d(in_ch[2])
        self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernels[2])
        self.bn3 = nn.BatchNorm1d(in_ch[3])

    def forward(self, v):
        v = self.embedding(v.long())
        v = v.transpose(2, 1)
        v = self.bn1(F.relu(self.conv1(v)))
        v = self.bn2(F.relu(self.conv2(v)))
        v = self.bn3(F.relu(self.conv3(v)))
        v = v.view(v.size(0), v.size(2), -1)
        return v


class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits


class RandomLayer(nn.Module):
    def __init__(self, input_dim_list, output_dim=256):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]
