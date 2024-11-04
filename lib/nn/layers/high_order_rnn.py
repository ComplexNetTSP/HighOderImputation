import torch
import torch.nn as nn
from einops import rearrange
from .high_order_conv import  SCConvLayer
from ..utils.ops import reverse_tensor

class HighOrderRecurrentCell(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 activation='relu'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation_fn = getattr(torch, activation)
        self.forget_gate = SCConvLayer(input_size + hidden_size, hidden_size)
        self.update_gate = SCConvLayer(input_size + hidden_size, hidden_size)
        self.c_gate = SCConvLayer(input_size + hidden_size, hidden_size)

    def forward(self, x, h,
                B1,
                B1T,
                B2,
                B2T,
                L0,
                L1,
                L2):
        x_gates = torch.cat([x, h], dim=1)
        r = torch.sigmoid(self.forget_gate(x_gates,
                                            B1,
                                            B1T,
                                            B2,
                                            B2T,
                                            L0,
                                            L1,
                                            L2))
        u = torch.sigmoid(self.update_gate(x_gates,
                                            B1,
                                            B1T,
                                            B2,
                                            B2T,
                                            L0,
                                            L1,
                                            L2))
        x_c = torch.cat([x, r * h], dim=1)

        c = self.c_gate(x_c,
                        B1,
                        B1T,
                        B2,
                        B2T,
                        L0,
                        L1,
                        L2)
        c = self.activation_fn(c)

        return u * h + (1. - u) * c

class HighOrderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,n_layers):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.rnn_cells = nn.ModuleList()
        for i in range(self.n_layers):
            self.rnn_cells.append(HighOrderRecurrentCell(
                input_size=self.input_size if i == 0 else self.hidden_size,
                hidden_size=self.hidden_size))

        self.output_layer = nn.Conv2d(self.hidden_size, self.output_size, kernel_size=1)

    def init_hidden_states(self, x):
        return [torch.zeros(size=(x.shape[0], self.hidden_size, x.shape[2])).to(x.device) for _ in range(self.n_layers)]

    def single_pass(self, x, h,
                    B1,
                    B1T,
                    B2,
                    B2T,
                    L0,
                    L1,
                    L2):
        out = x
        for l, layer in enumerate(self.rnn_cells):
            out = h[l] = layer(out, h[l],
                                B1,
                                B1T,
                                B2,
                                B2T,
                                L0,
                                L1,
                                L2)
        return out, h

    def forward(self, x,
                    B1,
                    B1T,
                    B2,
                    B2T,
                    L0,
                    L1,
                    L2,
                    h=None):
        # x:[batch, features, nodes, steps]
        *_, steps = x.size()
        if h is None:
            h = self.init_hidden_states(x)
        for step in range(steps):
            out, h = self.single_pass(x[..., step], h,
                                                    B1,
                                                    B1T,
                                                    B2,
                                                    B2T,
                                                    L0,
                                                    L1,
                                                    L2)

        return self.output_layer(out[..., None])

if __name__ == '__main__':
    pass