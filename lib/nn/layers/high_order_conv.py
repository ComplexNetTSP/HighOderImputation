import torch
from topomodelx.base.aggregation import Aggregation
from topomodelx.base.conv import Conv
from einops import rearrange
from ..utils.ops import block_diag


class SCConvLayer(torch.nn.Module):

    def __init__(self, node_channels , hidden_channels) -> None:
        super().__init__()

        self.node_channels = node_channels
        # self.edge_channels = edge_channels
        # self.face_channels = face_channels
        self.hidden_channels = hidden_channels

        # self.ln = torch.nn.Linear(self.node_channels, hidden_channels)

        self.conv_0_to_0 = Conv(
            in_channels=node_channels,
            out_channels=hidden_channels,
            update_func="relu",
        )
        self.conv_0_to_1 = Conv(
            in_channels=node_channels,
            out_channels=hidden_channels,
            update_func="relu",
        )

        self.conv_1_to_1 = Conv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            update_func="relu",
        )
        self.conv_1_to_0 = Conv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            update_func="relu",
        )

        self.conv_1_to_2 = Conv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            update_func="relu",
        )

        self.conv_2_to_1 = Conv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            update_func="relu",
        )

        self.conv_2_to_2 = Conv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            update_func="relu",
        )

        self.conv_HSN_0_to_0 = Conv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            update_func="relu",
        )

        self.conv_HSN_1_to_0 = Conv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            update_func="relu",
        )
        # self.bn1 = torch.nn.BatchNorm1d(self.node_channels)
        # self.bn2 = torch.nn.BatchNorm1d(self.edge_channels)
        # self.bn3 = torch.nn.BatchNorm1d(self.edge_channels)
        # self.bn4 = torch.nn.BatchNorm1d(self.node_channels)
        # self.bn5 = torch.nn.BatchNorm1d(self.face_channels)
        # self.bn6 = torch.nn.BatchNorm1d(self.edge_channels)
        # self.bn7 = torch.nn.BatchNorm1d(self.face_channels)

        self.aggr_on_nodes = Aggregation(aggr_func="sum", update_func="relu")
        self.aggr_on_edges = Aggregation(aggr_func="sum", update_func="relu")
        self.aggr_on_faces = Aggregation(aggr_func="sum", update_func="relu")

        self.aggr_on_nodes_HSN = Aggregation(aggr_func="sum", update_func="relu")


    def forward(
            self,
            x_0,
            B1,
            B1T,
            B2,
            B2T,
            L0,
            L1,
            L2
    ):
        # print(x_0.shape)
        # h_0 = self.ln(x_0)
        # X_0 -- > [batch, features, nodes]
        batch_size, _, nodes = x_0.shape
        # arrange X_0 -- > [batch, nodes, features]
        x_0 = rearrange(x_0, 'b f n -> (b n) f')

        supra_L0 = block_diag(*[L0 for _ in range(batch_size)])
        supra_L1 = block_diag(*[L1 for _ in range(batch_size)])
        supra_L2 = block_diag(*[L2 for _ in range(batch_size)])
        supra_B1 = block_diag(*[B1 for _ in range(batch_size)])
        supra_B1T = block_diag(*[B1T for _ in range(batch_size)])
        supra_B2 = block_diag(*[B2 for _ in range(batch_size)])
        supra_B2T = block_diag(*[B2T for _ in range(batch_size)])


        # print("shit")
        # print("x_0 shape", x_0.shape)
        # print("supra_L0 shape", supra_L0.shape)
        # print("shit")

        x0_0_0 = self.conv_0_to_0(x_0, supra_L0)
        x0_0_1 = self.conv_0_to_1(x_0, supra_B1T)
        # x0_1_2 = self.conv_1_to_2(x0_0_1, supra_B2T)

        x1_1_0 = self.conv_1_to_0(x0_0_1, supra_B1)
        x1_1_1 = self.conv_1_to_1(x0_0_1, supra_L1)
        x1_1_2 = self.conv_1_to_2(x0_0_1, supra_B2T)

        # adj_norm = adjacency_down_1_norm.add(adjacency_up_1_norm)

        x2_2_1 = self.conv_2_to_1(x1_1_2, supra_B2)
        # x2_1_2 = self.conv_1_to_2(x0_1_2, B2T)
        x2_2_2 = self.conv_2_to_2(x1_1_2, supra_L2)

        # x0_0_0 = self.bn1()
        # x1_1_0 = self.bn2()
        # x0_0_1 = self.bn3()
        # x1_1_1 = self.bn4()
        # x2_2_1 = self.bn5()
        # x1_1_2 = self.bn6()
        # x2_2_2 = self.bn7()

        x0_level1_out = self.aggr_on_nodes([x0_0_0, x1_1_0])
        x1_level1_out = self.aggr_on_edges([x0_0_1, x1_1_1, x2_2_1])
        x2_level1_out = self.aggr_on_faces([x1_1_2, x2_2_2])

        # HSN
        x0_HSN_0_0 = self.conv_HSN_0_to_0(x0_level1_out, supra_L0)
        x0_HSN_1_0 = self.conv_HSN_1_to_0(x1_level1_out, supra_B1)
        x0_HSN_out = self.aggr_on_nodes_HSN([x0_HSN_0_0, x0_HSN_1_0])

        x0_HSN_out = rearrange(x0_HSN_out, '(b n) f-> b f n',b = batch_size, n = nodes)

        return x0_HSN_out


if __name__ == '__main__':
    pass


