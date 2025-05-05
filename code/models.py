import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.nn import Linear as Linear_pyg
from torch_scatter import scatter
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network, register_layer


class GatedGCNLayer(pyg_nn.conv.MessagePassing):
    """
        GatedGCN layer
        Residual Gated Graph ConvNets
        https://arxiv.org/pdf/1711.07553.pdf
    """
    def __init__(self, in_dim, out_dim, dropout, residual,
                 equivstable_pe=False, **kwargs):
        super().__init__(**kwargs)
        self.A = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.B = pyg_nn.Linear(in_dim, out_dim, bias=True)
        # self.C = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.D = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.E = pyg_nn.Linear(in_dim, out_dim, bias=True)

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        self.EquivStablePE = equivstable_pe
        if self.EquivStablePE:
            self.mlp_r_ij = nn.Sequential(
                nn.Linear(1, out_dim), nn.ReLU(),
                nn.Linear(out_dim, 1),
                nn.Sigmoid())

        self.bn_node_x = nn.BatchNorm1d(out_dim)
        self.bn_edge_e = nn.BatchNorm1d(out_dim)
        self.dropout = dropout
        self.residual = residual
        self.e = None

    def forward(self, batch):
        x, e, edge_index = batch.x, batch.edge_attr, batch.edge_index
        # e_ij = Dx_i + Ex_j + Ce

        """
        x               : [n_nodes, in_dim]
        e               : [n_edges, in_dim]
        edge_index      : [2, n_edges]
        """
        if self.residual:
            x_in = x
            # e_in = e

        Ax = self.A(x)
        Bx = self.B(x)
        # Ce = self.C(e)
        Dx = self.D(x)
        Ex = self.E(x)

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        pe_LapPE = batch.pe_EquivStableLapPE if self.EquivStablePE else None

        x, e = self.propagate(edge_index,
                              Bx=Bx, Dx=Dx, Ex=Ex,
                              e=e, Ax=Ax,
                              PE=pe_LapPE)

        x = self.bn_node_x(x)
        # e = self.bn_edge_e(e)

        x = F.relu(x)
        # e = F.relu(e)

        x = F.dropout(x, self.dropout, training=self.training)
        # e = F.dropout(e, self.dropout, training=self.training)

        if self.residual:
            x = x_in + x
            # e = e_in + e

        batch.x = x
        batch.edge_attr = e

        return batch

    def message(self, Dx_i, Ex_j, PE_i, PE_j):
        """
        {}x_i           : [n_edges, out_dim]
        {}x_j           : [n_edges, out_dim]
        {}e             : [n_edges, out_dim]
        """
        # e_ij = Dx_i + Ex_j + Ce
        e_ij = Dx_i + Ex_j
        sigma_ij = torch.sigmoid(e_ij)

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        if self.EquivStablePE:
            r_ij = ((PE_i - PE_j) ** 2).sum(dim=-1, keepdim=True)
            r_ij = self.mlp_r_ij(r_ij)  # the MLP is 1 dim --> hidden_dim --> 1 dim
            sigma_ij = sigma_ij * r_ij

        self.e = e_ij
        return sigma_ij

    def aggregate(self, sigma_ij, index, Bx_j, Bx):
        """
        sigma_ij        : [n_edges, out_dim]  ; is the output from message() function
        index           : [n_edges]
        {}x_j           : [n_edges, out_dim]
        """
        dim_size = Bx.shape[0]  # or None ??   <--- Double check this

        sum_sigma_x = sigma_ij * Bx_j
        numerator_eta_xj = scatter(sum_sigma_x, index, 0, None, dim_size,
                                   reduce='sum')

        sum_sigma = sigma_ij
        denominator_eta_xj = scatter(sum_sigma, index, 0, None, dim_size,
                                     reduce='sum')

        out = numerator_eta_xj / (denominator_eta_xj + 1e-6)
        return out

    def update(self, aggr_out, Ax):
        """
        aggr_out        : [n_nodes, out_dim] ; is the output from aggregate() function after the aggregation
        {}x             : [n_nodes, out_dim]
        """
        x = Ax + aggr_out
        e_out = self.e
        del self.e
        return x, e_out


class GatedGCNGraphGymLayer(nn.Module):
    """GatedGCN layer.
    Residual Gated Graph ConvNets
    https://arxiv.org/pdf/1711.07553.pdf
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = GatedGCNLayer(in_dim=layer_config.dim_in,
                                   out_dim=layer_config.dim_out,
                                   dropout=0.,  # Dropout is handled by GraphGym's `GeneralLayer` wrapper
                                   residual=False,  # Residual connections are handled by GraphGym's `GNNStackStage` wrapper
                                   **kwargs)

    def forward(self, batch):
        return self.model(batch)



class GCN2ConvLayer(nn.Module):
    """GCNII Layer from https://arxiv.org/abs/2007.02133.
    """
    def __init__(self, dim_in, dim_out, dropout, residual):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout
        self.residual = residual

        self.model = pyg_nn.GCN2Conv(self.dim_in, alpha=0.2)
        # alpha value is set using results from the GCNII paper

    def forward(self, batch):
        x_in = batch.x

        batch.x = self.model(batch.x, batch.x0, batch.edge_index)

        batch.x = F.relu(batch.x)
        batch.x = F.dropout(batch.x, p=self.dropout, training=self.training)

        if self.residual:
            batch.x = x_in + batch.x  # residual connection

        return batch
    
class GINEConvESLapPE(pyg_nn.conv.MessagePassing):
    """GINEConv Layer with EquivStableLapPE implementation.

    Modified torch_geometric.nn.conv.GINEConv layer to perform message scaling
    according to EquivStable LapPE:
        ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
    """
    def __init__(self, nn, eps=0., train_eps=False, edge_dim=None, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        if edge_dim is not None:
            if hasattr(self.nn[0], 'in_features'):
                in_channels = self.nn[0].in_features
            else:
                in_channels = self.nn[0].in_channels
            self.lin = pyg_nn.Linear(edge_dim, in_channels)
        else:
            self.lin = None
        self.reset_parameters()

        if hasattr(self.nn[0], 'in_features'):
            out_dim = self.nn[0].out_features
        else:
            out_dim = self.nn[0].out_channels

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        self.mlp_r_ij = torch.nn.Sequential(
            torch.nn.Linear(1, out_dim), torch.nn.ReLU(),
            torch.nn.Linear(out_dim, 1),
            torch.nn.Sigmoid())

    def reset_parameters(self):
        pyg_nn.inits.reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        if self.lin is not None:
            self.lin.reset_parameters()
        pyg_nn.inits.reset(self.mlp_r_ij)

    def forward(self, x, edge_index, edge_attr=None, pe_LapPE=None, size=None):
        # if isinstance(x, Tensor):
        #     x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr,
                             PE=pe_LapPE, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j, edge_attr, PE_i, PE_j):
        if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError("Node and edge feature dimensionalities do not "
                             "match. Consider setting the 'edge_dim' "
                             "attribute of 'GINEConv'")

        if self.lin is not None:
            edge_attr = self.lin(edge_attr)

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        r_ij = ((PE_i - PE_j) ** 2).sum(dim=-1, keepdim=True)
        r_ij = self.mlp_r_ij(r_ij)  # the MLP is 1 dim --> hidden_dim --> 1 dim

        return ((x_j + edge_attr).relu()) * r_ij

    def __repr__(self):
        return f'{self.__class__.__name__}(nn={self.nn})'


class GINEConvLayer(nn.Module):
    """Graph Isomorphism Network with Edge features (GINE) layer.
    """
    def __init__(self, dim_in, dim_out, dropout, residual):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout
        self.residual = residual

        gin_nn = nn.Sequential(
            pyg_nn.Linear(dim_in, dim_out), nn.ReLU(),
            pyg_nn.Linear(dim_out, dim_out))
        self.model = pyg_nn.GINEConv(gin_nn)

    def forward(self, batch):
        x_in = batch.x

        batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr)

        batch.x = F.relu(batch.x)
        batch.x = F.dropout(batch.x, p=self.dropout, training=self.training)

        if self.residual:
            batch.x = x_in + batch.x  # residual connection

        return batch


class GINEConvGraphGymLayer(nn.Module):
    """Graph Isomorphism Network with Edge features (GINE) layer.
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        gin_nn = nn.Sequential(
            Linear_pyg(layer_config.dim_in, layer_config.dim_out), nn.ReLU(),
            Linear_pyg(layer_config.dim_out, layer_config.dim_out))
        self.model = pyg_nn.GINEConv(gin_nn)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr)
        return batch
    
class CustomGNN(nn.Module):
    """
    A custom GNN model that uses your config dictionary directly
    instead of relying on torch_geometric.graphgym.config.cfg
    """

    def __init__(self, dim_in, dim_edge, dim_out, model_cfg):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        # Optional pre-message passing layers
        self.has_pre_mp = model_cfg.get('layers_pre_mp', 0) > 0
        if self.has_pre_mp:
            from torch_geometric.graphgym.models.gnn import GNNPreMP
            self.pre_mp = GNNPreMP(
                dim_in,
                model_cfg['dim_inner'],
                model_cfg['layers_pre_mp']
            )
            dim_in = model_cfg['dim_inner']

        # assert model_cfg['dim_inner'] == dim_in, \
        #     "Mismatch between model config 'dim_inner' and layer dimensions."

        conv_model = self.build_conv_model(model_cfg['name'])
        self.model_type = model_cfg['name']
        self.layers = nn.ModuleList()
        print("dim_in:", dim_in)

        for _ in range(model_cfg['layers_mp']):
            self.layers.append(conv_model(
                in_dim=dim_in,
                out_dim=dim_in,
                dropout=model_cfg['dropout'],
                residual=model_cfg['residual'],
                equivstable_pe=model_cfg.get('equivstable_pe', False)
            ))

        self.post_mp = nn.Sequential(
            pyg_nn.Linear(dim_in, dim_in),
            nn.ReLU(),
            pyg_nn.Linear(dim_in, dim_out)
        )

    def build_conv_model(self, model_name):
        if model_name == 'gatedgcn':
            return GatedGCNLayer
        else:
            raise ValueError(f"Model type '{model_name}' not recognized")

    def forward(self, batch):
        # 🔧 Ensure inputs are float
        batch.x = batch.x.float()
        if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
            batch.edge_attr = batch.edge_attr.float()

        # print("Input batch x shape:", batch)

        # Apply encoder
        batch = self.encoder(batch)
        # print("Encoded batch x shape:", batch)

        # Pre-message-passing layers (if any)
        if self.has_pre_mp:
            batch = self.pre_mp(batch)

        # print("has_pre_mp batch x shape:", batch)

        # # Message passing layers
        # i = 1
        for conv in self.layers:
            if self.model_type == 'gcniiconv':
                batch.x0 = batch.x  # GCNII requires x0
            # print(f"Before conv {i} batch x shape:", batch.x.shape)
            # print(conv)
            # i+=1
            batch = conv(batch)

        # Final head
        # out = batch
        # out.x = self.post_mp(out.x)# .mean(dim=0)

        graph_repr = pyg_nn.global_mean_pool(batch.x, batch.batch)
        out = self.post_mp(graph_repr) 
        # print("Output batch x shape:", out)
        return out