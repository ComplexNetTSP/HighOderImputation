import torch
import torch.nn as nn
from einops import rearrange
from .high_order_conv import SCConvLayer
from ..utils.ops import reverse_tensor
from .high_order_rnn import HighOrderRecurrentCell

class SpatialDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear = nn.Conv1d(input_size, hidden_size, kernel_size=1)
        self.conv = SCConvLayer(hidden_size, hidden_size)
        self.read_out = nn.Conv1d(2*hidden_size, output_size, kernel_size=1)

    def forward(self, x, m, h, u, B1,
                                B1T,
                                B2,
                                B2T,
                                L0,
                                L1,
                                L2, ):
        x_in = [x, m, h] if u is None else [x, m, u, h]
        x_in = torch.cat(x_in, 1)
        x_in = self.linear(x_in)
        out = self.conv(x_in,B1=B1,B1T=B1T,B2=B2,B2T=B2T, L0=L0,L1=L1,L2=L2)

        # activation ?
        out = torch.cat([out, h], 1)

        return self.read_out(out), out


class HighOrderImputationLayer(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 u_size=None,
                 n_layers=1,
                 dropout=0.,
                 n_nodes=None,
                 layer_norm=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.u_size = u_size
        self.n_nodes = n_nodes
        self.layer_norm = layer_norm
        self.n_layers = n_layers
        self.dropout = dropout

        rnn_input_size = 2 * self.input_size + self.u_size  # input + mask + (eventually) exogenous
        self.cells = nn.ModuleList()
        self.norms = nn.ModuleList()


        for i in range(self.n_layers):
            self.cells.append(HighOrderRecurrentCell(input_size=rnn_input_size if i == 0 else self.hidden_size,
                                                     hidden_size = self.hidden_size))
            if layer_norm:
                # Or others ? Like BatchNorm, NodeNorm, PairNorm ?
                self.norms.append(nn.GroupNorm(num_groups=1, num_channels=self.hidden_size))
            else:
                self.norms.append(nn.Identity())

        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        # Fist stage readout
        self.first_stage = nn.Conv1d(in_channels=self.hidden_size, out_channels=self.input_size, kernel_size=1)

        self.spatial_decoder = SpatialDecoder(input_size=rnn_input_size + self.hidden_size,
                                              hidden_size=self.hidden_size,
                                              output_size=self.input_size)

        # Hidden state initialization embedding
        if n_nodes is not None:
            self.h0 = self.init_hidden_states(n_nodes)
        else:
            self.register_parameter('h0', None)

    def init_hidden_states(self, n_nodes):
        h0 = []
        for l in range(self.n_layers):
            std = 1. / torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float))
            vals = torch.distributions.Normal(0, std).sample((self.hidden_size, n_nodes))
            h0.append(nn.Parameter(vals))
        return nn.ParameterList(h0)

    def get_h0(self, x):
        if self.h0 is not None:
            return [h.expand(x.shape[0], -1, -1) for h in self.h0]
        return [torch.zeros(size=(x.shape[0], self.hidden_size, x.shape[2])).to(x.device)] * self.n_layers

    def update_state(self, x, h,
                            B1,
                            B1T,
                            B2,
                            B2T,
                            L0,
                            L1,
                            L2):
        rnn_in = x
        for layer, (cell, norm) in enumerate(zip(self.cells, self.norms)):
            rnn_in = h[layer] = norm(cell(rnn_in, h[layer], B1,
                                                            B1T,
                                                            B2,
                                                            B2T,
                                                            L0,
                                                            L1,
                                                            L2))
            if self.dropout is not None and layer < (self.n_layers - 1):
                rnn_in = self.dropout(rnn_in)

        return h

    def forward(self, x,
                    B1,
                    B1T,
                    B2,
                    B2T,
                    L0,
                    L1,
                    L2, mask=None, u=None, h=None):

        # x:[batch, features, nodes, steps]
        *_, steps = x.size()
        # infer all valid if mask is None
        if mask is None:
            mask = torch.ones_like(x, dtype=torch.uint8)

        # init hidden state using node embedding or the empty state
        if h is None:
            h = self.get_h0(x)
        elif not isinstance(h, list):
            h = [*h]

        predictions, imputations, states = [], [], []
        representations = []

        for step in range(steps):
            x_s = x[..., step]
            m_s = mask[..., step]
            h_s = h[-1]
            u_s = u[..., step] if u is not None else None
            # firstly impute missing values with predictions from state
            xs_hat_1 = self.first_stage(h_s)
            # fill missing values in input with prediction
            x_s = torch.where(m_s, x_s, xs_hat_1)
            # prepare inputs
            # retrieve maximum information from neighbors
            xs_hat_2, repr_s = self.spatial_decoder(x=x_s, m=m_s, h=h_s, u=u_s,
                                                    B1 = B1, B1T = B1T, B2 = B2, B2T = B2T,
                                                    L0 = L0, L1 = L1, L2 = L2) # receive messages from neighbors (no self-loop!)
            # readout of imputation state + mask to retrieve imputations
            # prepare inputs
            x_s = torch.where(m_s, x_s, xs_hat_2)
            inputs = [x_s, m_s]
            if u_s is not None:
                inputs.append(u_s)
            inputs = torch.cat(inputs, dim=1)  # x_hat_2 + mask + exogenous
            # update state with original sequence filled using imputations
            h = self.update_state(inputs, h, B1, B1T, B2, B2T, L0, L1, L2)
            # store imputations and states
            imputations.append(xs_hat_2)
            predictions.append(xs_hat_1)
            states.append(torch.stack(h, dim=0))
            representations.append(repr_s)

        # Aggregate outputs -> [batch, features, nodes, steps]
        imputations = torch.stack(imputations, dim=-1)
        predictions = torch.stack(predictions, dim=-1)
        states = torch.stack(states, dim=-1)
        representations = torch.stack(representations, dim=-1)

        return imputations, predictions, representations, states

class BiHighOrderImputationLayer(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 ff_size,
                 ff_dropout,
                 n_layers=1,
                 dropout=0.,
                 n_nodes=None,
                 u_size=0,
                 embedding_size=0,
                 layer_norm=False,
                 merge='mlp'
                 ):
        super().__init__()

        self.fwd_high_order_impute_layer = HighOrderImputationLayer(input_size=input_size,
                                                                    hidden_size=hidden_size,
                                                                    u_size=u_size,
                                                                    n_layers=n_layers,
                                                                    dropout=dropout,
                                                                    n_nodes=n_nodes,
                                                                    layer_norm=layer_norm)
        self.bwd_high_order_impute_layer = HighOrderImputationLayer(input_size=input_size,
                                                                    hidden_size=hidden_size,
                                                                    u_size=u_size,
                                                                    n_layers=n_layers,
                                                                    dropout=dropout,
                                                                    n_nodes=n_nodes,
                                                                    layer_norm=layer_norm)

        if n_nodes is None:
            embedding_size = 0

        if embedding_size > 0:
            self.emb = nn.Parameter(torch.empty(embedding_size, n_nodes))
            nn.init.kaiming_normal_(self.emb, nonlinearity='relu')
        else:
            self.register_parameter('emb', None)

        if merge == 'mlp':
            self._impute_from_states = True
            self.out = nn.Sequential(
                nn.Conv2d(in_channels=4 * hidden_size + input_size + embedding_size,
                          out_channels=ff_size, kernel_size=1),
                nn.ReLU(),
                nn.Dropout(ff_dropout),
                nn.Conv2d(in_channels=ff_size, out_channels=input_size, kernel_size=1)
            )

        elif merge in ['mean', 'sum', 'min', 'max']:
            self._impute_from_states = False
            self.out = getattr(torch, merge)
        else:
            raise ValueError("Merge option %s not allowed." % merge)


    def forward(self, x,
                B1,
                B1T,
                B2,
                B2T,
                L0,
                L1,
                L2,
               mask=None,
               u=None,):
        # Forward
        fwd_out, fwd_pred, fwd_repr, _ = self.fwd_high_order_impute_layer(x, B1=B1, B1T=B1T,B2=B2, B2T=B2T,L0=L0, L1=L1, L2=L2, mask=mask, u=u)
        # Backward
        rev_x, rev_mask, rev_u = [reverse_tensor(tens) for tens in (x, mask, u)]

        *bwd_res, _ = self.bwd_high_order_impute_layer(rev_x,  B1=B1, B1T=B1T, B2=B2, B2T=B2T, L0=L0, L1=L1, L2=L2, mask=rev_mask, u=rev_u)

        bwd_out, bwd_pred, bwd_repr = [reverse_tensor(res) for res in bwd_res]


        if self._impute_from_states:
            inputs = [fwd_repr, bwd_repr, mask]
            if self.emb is not None:
                b, *_, s = fwd_repr.shape  # fwd_h: [batches, channels, nodes, steps]
                inputs += [self.emb.view(1, *self.emb.shape, 1).expand(b, -1, -1, s)]  # stack emb for batches and steps
            imputation = torch.cat(inputs, dim=1)
            imputation = self.out(imputation)
        else:
            imputation = torch.stack([fwd_out, bwd_out], dim=1)
            imputation = self.out(imputation, dim=1)

        predictions = torch.stack([fwd_out, bwd_out, fwd_pred, bwd_pred], dim=0)

        return imputation, predictions








