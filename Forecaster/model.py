import torch
import torch.nn as nn
from layers.Multi_Expert_Block import MEB
from layers.RevIN import RevIN

class QuadraFormer(nn.Module):
    def __init__(self, **kwargs):
        params = {**kwargs}
        super(QuadraFormer, self).__init__()
        self.layer_nums = 1
        # self.layer_nums = params["num_layers"]  # --layer_nums 3
        self.num_nodes = params["input_dim"]
        self.pre_len = params["prediction_length"]
        self.seq_len = params["window_size"]
        self.k = 2  # --k 2
        self.num_experts_list = [4]
        self.patch_size_list = [[16, 8, 4, 2]]
        self.d_model = 16  # --d_model 16
        self.d_ff = 64  # --d_ff 64
        self.residual_connection = True  # --residual_connection 0
        self.batch_norm = False  # --batch_norm 0
        self.revin = True  # --revin 1
        self.drop = 0.1  # --drop 0.1

        if self.revin:
            self.revin_layer = RevIN(
                num_features=self.num_nodes,
                affine=False,  # 默认无affine变换
                subtract_last=False
            )

        self.start_fc = nn.Linear(in_features=1, out_features=self.d_model)
        self.AMS_lists = nn.ModuleList()
        self.device = torch.device('cuda:0')  # 默认使用gpu 0

        for num in range(self.layer_nums):
            self.AMS_lists.append(
                MEB(self.seq_len, self.seq_len, self.num_experts_list[num], self.device, k=self.k,
                    num_nodes=self.num_nodes, patch_size=self.patch_size_list[num], noisy_gating=True,
                    d_model=self.d_model, d_ff=self.d_ff, layer_number= num + 1,
                    residual_connection=self.residual_connection, batch_norm=self.batch_norm))
        self.projections = nn.Sequential(
            nn.Linear(self.seq_len * self.d_model, self.pre_len)
        )

    def forward(self, x):
        balance_loss = torch.tensor(0.0, device=x.device)
        if self.revin:
            x = self.revin_layer(x, 'norm')
        out = self.start_fc(x.unsqueeze(-1))
        batch_size = x.shape[0]
        for layer in self.AMS_lists:
            out, aux_loss = layer(out)
            balance_loss = balance_loss + aux_loss.to(x.device)
        out = out.permute(0, 2, 1, 3).reshape(batch_size, self.num_nodes, -1)
        res = out
        out = self.projections(out).transpose(2, 1)
        out += res.mean(dim=2).unsqueeze(1)
        if self.revin:
            out = self.revin_layer(out, 'denorm')
        return out, balance_loss


