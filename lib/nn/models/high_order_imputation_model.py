import torch
from einops import rearrange
from torch import nn
from ..layers import HighOrderImputationLayer, BiHighOrderImputationLayer
from ...utils.parser_utils import str_to_bool

class HighOrderImputationModel(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 d_ff,
                 ff_dropout,
                 B1,
                 B1T,
                 B2,
                 B2T,
                 L0,
                 L1,
                 L2,
                 n_layers = 1,
                 d_u=0,
                 d_emb=0,
                 layer_norm=False,
                 merge='mlp',
                 impute_only_holes=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.d_ff = d_ff
        self.ff_dropout = ff_dropout

        self.register_buffer('B1', torch.tensor(B1).float())
        self.register_buffer('B1T',torch.tensor(B1T).float())
        self.register_buffer('B2',torch.tensor(B2).float())
        self.register_buffer('B2T', torch.tensor(B2T).float())
        self.register_buffer('L0', torch.tensor(L0).float())
        self.register_buffer('L1', torch.tensor(L1).float())
        self.register_buffer('L2', torch.tensor(L2).float())
        self.n_layers = n_layers
        self.d_u = d_u
        self.d_emb = d_emb
        self.layer_norm = layer_norm
        self.merge = merge
        self.impute_only_holes = impute_only_holes

        self.impute_layer = BiHighOrderImputationLayer(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            ff_size=self.d_ff,
            ff_dropout=self.ff_dropout,
            u_size = self.d_u,
            embedding_size = self.d_emb,
            layer_norm=self.layer_norm,
            merge=self.merge
        )



    def forward(self, x, mask = None,u = None):
        # x: [batches, steps, nodes, channels] -> [batches, channels, nodes, steps]
        x = rearrange(x, 'b s n c -> b c n s')
        if mask is not None:
            mask = rearrange(mask, 'b s n c -> b c n s')

        if u is not None:
            u = rearrange(u, 'b s n c -> b c n s')

        # imputation: [batches, channels, nodes, steps] prediction: [4, batches, channels, nodes, steps]
        imputation, prediction = self.impute_layer(x,
                                                   B1 = self.B1, B1T = self.B1T, B2 = self.B2, B2T = self.B2T,
                                                   L0 = self.L0, L1 = self.L1, L2 = self.L2,
                                                   mask=mask, u=u)

        # In evaluation stage impute only missing values
        if self.impute_only_holes and not self.training:
            imputation = torch.where(mask, x, imputation)

        # out: [batches, channels, nodes, steps] -> [batches, steps, nodes, channels]
        imputation = torch.transpose(imputation, -3, -1)
        prediction = torch.transpose(prediction, -3, -1)
        if self.training:
            return imputation, prediction
        return imputation

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--hidden-size', type=int, default=64)
        parser.add_argument('--d-ff', type=int, default=64)
        parser.add_argument('--ff-dropout', type=int, default=0.3)
        parser.add_argument('--n-layers', type=int, default=1)
        parser.add_argument('--d-u', type=int, default=0)
        parser.add_argument('--d-emb', type=int, default=8)
        parser.add_argument('--layer-norm', type=str_to_bool, nargs='?', const=True, default=False)
        parser.add_argument('--merge', type=str, default='mlp')
        parser.add_argument('--impute-only-holes', type=str_to_bool, nargs='?', const=True, default=True)
        return parser
