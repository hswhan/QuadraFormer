import math
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from Forecaster.layers.Embedding import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat


class CrossDimensionAttention(nn.Module):
    def __init__(self, seg_num, factor, d_model, n_heads=2, window_size=5, d_ff=None, dropout=0.1):
        super(CrossDimensionAttention, self).__init__()
        d_ff = d_ff or (4 * d_model)
        self.n_heads = 2  # 修改此处
        self.head_dim = d_model // self.n_heads  # head_dim = d_model / n_heads
        self.d_model = d_model
        self.factor = factor
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        # 对 router、x 投影的线性层
        self.linear_router = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_query = nn.Linear(d_model, d_model)

        # 使用稀疏注意力模块代替标准的 ScaledDotProductAttention
        self.dim_sender = SparseScaledDotProductAttention(d_model, n_heads, window_size=window_size,
                                                          attn_dropout=dropout, res_attention=False, lsa=False)
        self.dim_receiver = SparseScaledDotProductAttention(d_model, n_heads, window_size=window_size,
                                                            attn_dropout=dropout, res_attention=False, lsa=False)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.MLP2 = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def _create_sparse_mask(self, seg_num):
        mask = build_sparse_mask(seg_num, seg_num, self.window_size, device='cpu')
        return mask

    def forward(self, x):
        """
        x: [B, T, L, d_model]
        返回: [B, T, L, d_model]
        """
        B, T, L, D = x.shape  # D == d_model
        n_heads = self.n_heads
        head_dim = self.head_dim  # 每个头的维度

        # 1. 将 x 的形状由 [B, T, L, D] 转换为 [B, L, T, D]，再合并 B 和 L: [B*L, T, D]
        x_perm = rearrange(x, 'b t l d -> b l t d')
        x_dim = x_perm.reshape(B * L, T, D)

        # 2. 初步 MLP1 + 残差 + Norm
        x_proj = x_dim + self.dropout(self.MLP1(x_dim))
        x_proj = self.norm1(x_proj)  # [B*L, T, D]

        # 3. 对 x_proj 生成 keys 和 values，用于 sender 注意力
        keys = self.linear_k(x_proj)  # [B*L, T, D]
        values = self.linear_v(x_proj)  # [B*L, T, D]
        # 拆分多头：reshape为 [B*L, T, n_heads, head_dim]，再 permute 得到 [B*L, n_heads, T, head_dim]
        keys = keys.view(B * L, T, n_heads, head_dim).permute(0, 2, 1, 3)
        values = values.view(B * L, T, n_heads, head_dim).permute(0, 2, 1, 3)

        # 4. 准备 router:
        # 原 router 的 shape: [L, factor, D]. 扩展到每个样本: [B, L, factor, D], 合并 B 与 L: [B*L, factor, D]
        router = self.router.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * L, self.factor, D)
        # 对 router 进行投影：
        router_proj = self.linear_router(router)  # [B*L, factor, D]
        # 拆分多头： reshape为 [B*L, factor, n_heads, head_dim] → permute得到 [B*L, n_heads, factor, head_dim]
        router_proj = router_proj.view(B * L, self.factor, n_heads, head_dim).permute(0, 2, 1, 3)

        # 5. Sender Attention: 用 router_proj 作为 query, keys 和 values 作为键和值，计算 sender 注意力
        # router_proj: [B*L, n_heads, factor, head_dim]
        # keys: [B*L, n_heads, T, head_dim]
        sender_output, _ = self.dim_sender(router_proj, keys, values)
        # sender_output: [B*L, n_heads, factor, head_dim]（buffer）
        buffer = sender_output

        # 6. Receiver Attention: 对 x_proj 使用 linear_query 生成 query，并拆分多头
        query = self.linear_query(x_proj)  # [B*L, T, D]
        query = query.view(B * L, T, n_heads, head_dim).permute(0, 2, 1, 3)  # [B*L, n_heads, T, head_dim]
        # 使用 buffer 作为 key 和 value：buffer的 shape [B*L, n_heads, factor, head_dim]
        receiver_output, _ = self.dim_receiver(query, buffer, buffer)
        # receiver_output: [B*L, n_heads, T, head_dim]
        # 合并多头：permute并 reshape 回 [B*L, T, D]
        receiver_output = receiver_output.permute(0, 2, 1, 3).contiguous().view(B * L, T, D)

        # 7. 残差 + MLP2 + Norm
        x_updated = x_proj + self.dropout(self.MLP2(receiver_output))
        x_updated = self.norm2(x_updated)  # [B*L, T, D]

        # 8. 恢复原始形状：reshape回 [B, L, T, D]，再 rearrange为 [B, T, L, D]
        final_out = x_updated.view(B, L, T, D).permute(0, 2, 1, 3)
        final_out = self.norm3(final_out)
        return final_out


# -----------------------------
# 主模块：QuadraLayer
# -----------------------------
class QuadraLayer(nn.Module):
    def __init__(self, device, d_model, d_ff, num_nodes, patch_nums, patch_size,
                 dynamic, factorized, layer_number, batch_norm,
                 # 参数用于 intra_cross_attention，提供默认值
                 cross_factor=8,
                 cross_d_ff=None,
                 cross_dropout=0.1,
                 cross_nheads=2,
                 ):
        super(QuadraLayer, self).__init__()
        if cross_d_ff is None:
            cross_d_ff = 4 * d_model

        self.device = device
        self.d_model = d_model
        self.num_nodes = num_nodes
        self.dynamic = dynamic
        self.patch_nums = patch_nums  # 总 patch 数目（要求 temporal_length / patch_size）
        self.patch_size = patch_size  # 每个 patch 的时间步数
        self.layer_number = layer_number
        self.batch_norm = batch_norm

        # --------------------------
        # Intra Patch 模块：局部 patch 内注意力计算（intra-dim，不含跨-dim交互）
        # --------------------------
        self.intra_embedding_shared = nn.Parameter(torch.rand(1, 1, self.num_nodes, 16), requires_grad=True)

        self.embedding_generator_shared = nn.Sequential(
            nn.Linear(16, d_model)
        )

        self.intra_patch_attention = Intra_Patch_Attention(d_model, factorized=factorized)
        self.weights_generator_distinct = WeightGenerator(d_model, d_model, mem_dim=16,
                                                          num_nodes=num_nodes, factorized=factorized,
                                                          number_of_weights=2)
        self.weights_generator_shared = WeightGenerator(d_model, d_model, mem_dim=None,
                                                        num_nodes=num_nodes, factorized=False, number_of_weights=2)
        # 用于将多个 patch 输出线性转换后对齐（原代码中采用 permute + Linear）
        self.intra_Linear = nn.Linear(self.patch_nums, self.patch_nums * patch_size)
        # --------------------------
        # intra_cross_attention：对每个 patch 内输出执行跨-dim TSA 操作
        self.intra_cross_attention = CrossDimensionAttention(seg_num=num_nodes, factor=cross_factor,
                                                             d_model=d_model, n_heads=cross_nheads,
                                                             d_ff=cross_d_ff, dropout=cross_dropout)
        # --------------------------
        # Inter Patch 模块：利用 unfold 将整个序列划分为 patch 后，进行 patch 之间的局部注意力计算
        self.emb_linear = nn.Linear(d_model * patch_size, d_model * patch_size)
        self.W_pos = positional_encoding(pe='sincos', learn_pe=True, q_len=patch_nums, d_model=d_model * patch_size)
        self.dropout = nn.Dropout(0.1)
        n_heads = d_model  # 这里简单设置 n_heads = d_model，具体可调整
        d_k = (d_model * patch_size) // n_heads
        d_v = (d_model * patch_size) // n_heads
        self.inter_patch_attention = Inter_Patch_Attention(d_model * patch_size, d_model * patch_size,
                                                           n_heads, d_k, d_v, attn_dropout=0,
                                                           proj_dropout=0.1, res_attention=False)
        # --------------------------
        # inter_cross_attention：对 inter branch 的输出执行跨-dim TSA 操作
        self.inter_cross_attention = CrossDimensionAttention(seg_num=num_nodes, factor=cross_factor,
                                                             d_model=d_model, n_heads=cross_nheads,
                                                             d_ff=cross_d_ff, dropout=cross_dropout)
        # --------------------------
        # Final FFN 以及（可选）归一化层、dropout，进行最终融合处理
        self.final_ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )
        if batch_norm:
            self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
            self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_attn = None
            self.norm_ffn = None

    def forward(self, x):
        """
        输入 x: [batch, temporal_length, num_nodes, d_model]
        要求 temporal_length 可被 patch_size 整除，且 patch_nums = temporal_length / patch_size
        """
        batch_size = x.size(0)
        #############################################
        # Stage 1 & 2: Intra-Patch（局部）处理，结合 Intra Attention 与跨维度（TSA）处理
        #############################################
        intra_processed = None
        # 获取权重（由 WeightGenerator 生成）
        weights_shared, biases_shared = self.weights_generator_shared()
        weights_distinct, biases_distinct = self.weights_generator_distinct()
        for i in range(self.patch_nums):
            # 切出当前 patch，形状 [batch, patch_size, num_nodes, d_model]
            t = x[:, i * self.patch_size: (i + 1) * self.patch_size, :, :]
            # 获取对应的 learnable embedding，形状 [batch, 1, num_nodes, embed_dim]
            intra_emb = self.embedding_generator_shared(self.intra_embedding_shared).expand(batch_size, -1, -1, -1)
            # 拼接 embedding 与 patch 数据 → [batch, (1 + patch_size), num_nodes, ?]
            t_cat = torch.cat([intra_emb, t], dim=1)
            # Intra-Patch Attention计算（局部时序与 dim 内信息交互）
            out_intra, attn_intra = self.intra_patch_attention(
                intra_emb, t_cat, t_cat,
                weights_distinct, biases_distinct,
                weights_shared, biases_shared
            )
            # 立即对当前 patch 输出执行跨-dim（TSA）处理
            out_intra_cross = self.intra_cross_attention(out_intra)  # 输出形状：[batch, new_patch_size, num_nodes, d_model]
            if intra_processed is None:
                intra_processed = out_intra_cross
            else:
                intra_processed = torch.cat([intra_processed, out_intra_cross], dim=1)
        # 对拼接后的 intra 输出做线性变换对齐（类似原实现）
        intra_processed = intra_processed.permute(0, 3, 2, 1)  # [batch, d_model, num_nodes, temporal_intra]
        intra_processed = self.intra_Linear(intra_processed)
        intra_processed = intra_processed.permute(0, 3, 2, 1)  # 恢复为 [batch, temporal_intra, num_nodes, d_model]

        #############################################
        # Stage 3: Inter-Patch Intra-Dim Attention（patch 之间的局部交互）
        #############################################
        # 利用 unfold 将原始 x 划分为 patch，得到形状 [batch, patch_num, num_nodes, d_model, patch_size]
        x_unfold = x.unfold(dimension=1, size=self.patch_size, step=self.patch_size)
        x_unfold = x_unfold.permute(0, 2, 1, 3, 4)  # [batch, num_nodes, patch_num, d_model, patch_size]
        b, nvar, patch_num, d, patch_len = x_unfold.shape
        # 将数据合并，得到 [b*nvar, patch_num, d * patch_size]
        x_inter = x_unfold.reshape(b * nvar, patch_num, d * patch_len)
        x_inter = self.emb_linear(x_inter)
        x_inter = self.dropout(x_inter + self.W_pos)
        # 调用 Inter-Patch Attention，输出形状 [b*nvar, patch_num, d_model]
        inter_out, attn_inter = self.inter_patch_attention(Q=x_inter, K=x_inter, V=x_inter)
        # 恢复形状：[b, nvar, patch_num, d_model]，然后扩展 patch 维还原为时间轴
        inter_out = inter_out.reshape(b, nvar, patch_num, self.patch_size, self.d_model)
        # 然后将 patch_num 和 patch_size 这两个时间轴维度合并为一个
        inter_out = inter_out.reshape(b, nvar, patch_num * self.patch_size, self.d_model)
        # 最后转置回 [batch, temporal_inter, num_nodes, d_model]
        inter_out = inter_out.permute(0, 2, 1, 3)

        #############################################
        # Stage 4: Inter-Patch Cross-Dimension Attention（patch 间跨-dim 信息交互）
        #############################################
        inter_processed = self.inter_cross_attention(inter_out)  # 输出形状：[batch, temporal_inter, num_nodes, d_model]

        #############################################
        # 融合与最终处理
        #############################################
        # 这里简单采用逐元素相加融合（假设时间步数对齐），可根据需要设计其它融合策略
        combined = x + intra_processed + inter_processed
        if self.norm_attn is not None:
            b, T, nvar, dm = combined.shape
            combined = self.norm_attn(combined.reshape(b * nvar, T, dm)).reshape(b, T, nvar, dm)
        combined = self.dropout(combined)
        combined = self.final_ffn(combined) + combined
        if self.norm_ffn is not None:
            b, T, nvar, dm = combined.shape
            combined = self.norm_ffn(combined.reshape(b * nvar, T, dm)).reshape(b, T, nvar, dm)

        return combined, (attn_intra, attn_inter)

# class QuadraLayer(nn.Module):
#     def __init__(self, device, d_model, d_ff, num_nodes, patch_nums, patch_size,
#                  dynamic, factorized, layer_number, batch_norm,
#                  cross_factor=4, cross_d_ff=None, cross_dropout=0.1, cross_nheads=2):
#         super(QuadraLayer, self).__init__()
#         if cross_d_ff is None:
#             cross_d_ff = 4 * d_model
#
#         self.device = device
#         self.d_model = d_model
#         self.num_nodes = num_nodes
#         self.dynamic = dynamic
#         self.patch_nums = patch_nums
#         self.patch_size = patch_size
#         self.layer_number = layer_number
#         self.batch_norm = batch_norm
#
#         # Embedding（优化为共享）
#         self.intra_embedding_shared = nn.Parameter(torch.rand(1, 1, num_nodes, 16), requires_grad=True)
#         self.embedding_generator_shared = nn.Sequential(nn.Linear(16, d_model))
#
#         self.intra_patch_attention = Intra_Patch_Attention(d_model, factorized=factorized)
#         self.weights_generator_distinct = WeightGenerator(d_model, d_model, mem_dim=16,
#                                                           num_nodes=num_nodes, factorized=factorized,
#                                                           number_of_weights=2)
#         self.weights_generator_shared = WeightGenerator(d_model, d_model, mem_dim=None,
#                                                         num_nodes=num_nodes, factorized=False, number_of_weights=2)
#
#         self.intra_Linear = nn.Linear(patch_nums, patch_nums * patch_size)
#
#         self.intra_cross_attention = CrossDimensionAttention(seg_num=num_nodes, factor=cross_factor,
#                                                              d_model=d_model, n_heads=cross_nheads,
#                                                              d_ff=cross_d_ff, dropout=cross_dropout)
#
#         self.emb_linear = nn.Linear(d_model * patch_size, d_model * patch_size)
#         self.W_pos = positional_encoding(pe='zeros', learn_pe=True, q_len=patch_nums, d_model=d_model * patch_size)
#         self.dropout = nn.Dropout(0.1)
#         self.inter_patch_attention = Inter_Patch_Attention(d_model * patch_size, d_model * patch_size,
#                                                            n_heads=d_model,
#                                                            d_k=(d_model * patch_size) // d_model,
#                                                            d_v=(d_model * patch_size) // d_model,
#                                                            attn_dropout=0., proj_dropout=0.1)
#
#         self.inter_cross_attention = CrossDimensionAttention(seg_num=num_nodes, factor=cross_factor,
#                                                              d_model=d_model, n_heads=cross_nheads,
#                                                              d_ff=cross_d_ff, dropout=cross_dropout)
#
#         self.final_ffn = nn.Sequential(
#             nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(0.1), nn.Linear(d_model, d_model)
#         )
#
#         if batch_norm:
#             self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
#             self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
#         else:
#             self.norm_attn = None
#             self.norm_ffn = None
#
#     def forward(self, x):
#         batch_size, total_len, num_nodes, d_model = x.size()
#         weights_shared, biases_shared = self.weights_generator_shared()
#         weights_distinct, biases_distinct = self.weights_generator_distinct()
#
#         intra_outputs = []
#         for i in range(self.patch_nums):
#             t = x[:, i * self.patch_size: (i + 1) * self.patch_size, :, :]  # [B, patch, N, D]
#             intra_emb = self.embedding_generator_shared(self.intra_embedding_shared).expand(batch_size, -1, -1, -1)
#             t_cat = torch.cat([intra_emb, t], dim=1)
#             out_intra, attn_intra = self.intra_patch_attention(
#                 intra_emb, t_cat, t_cat,
#                 weights_distinct, biases_distinct,
#                 weights_shared, biases_shared
#             )
#             out_intra_cross = self.intra_cross_attention(out_intra)
#             intra_outputs.append(out_intra_cross)
#
#         intra_processed = torch.cat(intra_outputs, dim=1)  # [B, T, N, D]
#         intra_processed = intra_processed.permute(0, 3, 2, 1)  # [B, D, N, T]
#         intra_processed = self.intra_Linear(intra_processed)  # Linear over time axis
#         intra_processed = intra_processed.permute(0, 3, 2, 1)  # [B, T, N, D]
#
#         x_unfold = x.unfold(dimension=1, size=self.patch_size, step=self.patch_size)  # [B, patch_num, N, D, patch]
#         x_unfold = x_unfold.permute(0, 2, 1, 3, 4).reshape(batch_size * num_nodes, self.patch_nums, d_model * self.patch_size)
#         x_inter = self.emb_linear(x_unfold) + self.W_pos
#         x_inter = self.dropout(x_inter)
#         inter_out, attn_inter = self.inter_patch_attention(x_inter, x_inter, x_inter)
#         inter_out = inter_out.reshape(batch_size, num_nodes, self.patch_nums, self.patch_size, d_model)
#         inter_out = inter_out.permute(0, 2, 3, 1, 4).reshape(batch_size, self.patch_nums * self.patch_size, num_nodes, d_model)
#
#         inter_processed = self.inter_cross_attention(inter_out)
#
#         combined = x + intra_processed + inter_processed
#         if self.norm_attn:
#             b, T, nvar, dm = combined.shape
#             combined = self.norm_attn(combined.reshape(b * nvar, T, dm)).reshape(b, T, nvar, dm)
#
#         combined = self.dropout(combined)
#         combined = self.final_ffn(combined) + combined
#
#         if self.norm_ffn:
#             b, T, nvar, dm = combined.shape
#             combined = self.norm_ffn(combined.reshape(b * nvar, T, dm)).reshape(b, T, nvar, dm)
#
#         return combined, (attn_intra, attn_inter)




class CustomLinear(nn.Module):
    def __init__(self, factorized):
        super(CustomLinear, self).__init__()
        self.factorized = factorized

    def forward(self, input, weights, biases):
        if self.factorized:
            return torch.matmul(input.unsqueeze(3), weights).squeeze(3) + biases
        else:
            return torch.matmul(input, weights) + biases


class Intra_Patch_Attention(nn.Module):
    def __init__(self, d_model, factorized):
        super(Intra_Patch_Attention, self).__init__()
        self.head = 2

        if d_model % self.head != 0:
            raise Exception('Hidden size is not divisible by the number of attention heads')

        self.head_size = int(d_model // self.head)
        self.custom_linear = CustomLinear(factorized)

    def forward(self, query, key, value, weights_distinct, biases_distinct, weights_shared, biases_shared):
        batch_size = query.shape[0]

        key = self.custom_linear(key, weights_distinct[0], biases_distinct[0])
        value = self.custom_linear(value, weights_distinct[1], biases_distinct[1])
        query = torch.cat(torch.split(query, self.head_size, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_size, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_size, dim=-1), dim=0)

        query = query.permute((0, 2, 1, 3))
        key = key.permute((0, 2, 3, 1))
        value = value.permute((0, 2, 1, 3))



        attention = torch.matmul(query, key)
        attention /= (self.head_size ** 0.5)

        attention = torch.softmax(attention, dim=-1)

        x = torch.matmul(attention, value)
        x = x.permute((0, 2, 1, 3))
        x = torch.cat(torch.split(x, batch_size, dim=0), dim=-1)

        if x.shape[0] == 0:
            x = x.repeat(1, 1, 1, int(weights_shared[0].shape[-1] / x.shape[-1]))

        x = self.custom_linear(x, weights_shared[0], biases_shared[0])
        x = torch.relu(x)
        x = self.custom_linear(x, weights_shared[1], biases_shared[1])
        return x, attention


class Inter_Patch_Attention(nn.Module):
    def __init__(self, d_model, out_dim, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0.,
                 proj_dropout=0., qkv_bias=True, lsa=False):
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = SparseScaledDotProductAttention(d_model, n_heads, window_size=10,attn_dropout=attn_dropout,
                                                  res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, out_dim), nn.Dropout(proj_dropout))


    def forward(self, Q, K=None, V=None, prev=None, key_padding_mask=None, attn_mask=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, Q.shape[1], self.n_heads, self.d_k).transpose(1,
                                                                                 2)  # q_s    : [bs x n_heads x q_len x d_k]  此处的q_len为patch_num
        # k_s = self.W_K(K).view(bs, K.shape[1], self.n_heads, self.d_k).permute(0, 2, 3,
        #                                                                        1)  # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        k_s = self.W_K(K).view(bs, K.shape[1], self.n_heads, self.d_k).transpose(1, 2)

        v_s = self.W_V(V).view(bs, V.shape[1], self.n_heads, self.d_v).transpose(1,
                                                                                 2)  # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev,
                                                              key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        output = output.transpose(1, 2).contiguous().view(bs, Q.shape[1],
                                                          self.n_heads * self.d_v)  # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        return output, attn_weights


class ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need, 2017)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        # 如果 lsa 为 True，则 scale 为可训练参数；否则固定。
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q, k, v, prev=None, key_padding_mask=None, attn_mask=None):
        # q, k, v: shape [batch, n_heads, seq_len, d_model]
        attn_scores = torch.matmul(q, k) * self.scale  # 例如输出 shape: [batch, n_heads, seq_len_q, seq_len_k]
        if prev is not None:
            attn_scores = attn_scores + prev
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -float('inf'))
            else:
                attn_scores += attn_mask
        if key_padding_mask is not None:
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -float('inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights


class WeightGenerator(nn.Module):
    def __init__(self, in_dim, out_dim, mem_dim, num_nodes, factorized, number_of_weights=4):
        super(WeightGenerator, self).__init__()
        #print('FACTORIZED {}'.format(factorized))
        self.number_of_weights = number_of_weights
        self.mem_dim = mem_dim
        self.num_nodes = num_nodes
        self.factorized = factorized
        self.out_dim = out_dim
        if self.factorized:
            # self.memory = nn.Parameter(torch.randn(num_nodes, mem_dim), requires_grad=True).to('cpu')
            self.memory = nn.Parameter(torch.randn(num_nodes, mem_dim), requires_grad=True).to('cuda:0')
            self.generator = self.generator = nn.Sequential(*[
                nn.Linear(mem_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 100)
            ])

            self.mem_dim = 10
            self.P = nn.ParameterList(
                [nn.Parameter(torch.Tensor(in_dim, self.mem_dim), requires_grad=True) for _ in
                 range(number_of_weights)])
            self.Q = nn.ParameterList(
                [nn.Parameter(torch.Tensor(self.mem_dim, out_dim), requires_grad=True) for _ in
                 range(number_of_weights)])
            self.B = nn.ParameterList(
                [nn.Parameter(torch.Tensor(self.mem_dim ** 2, out_dim), requires_grad=True) for _ in
                 range(number_of_weights)])
        else:
            self.P = nn.ParameterList(
                [nn.Parameter(torch.Tensor(in_dim, out_dim), requires_grad=True) for _ in range(number_of_weights)])
            self.B = nn.ParameterList(
                [nn.Parameter(torch.Tensor(1, out_dim), requires_grad=True) for _ in range(number_of_weights)])
        self.reset_parameters()

    def reset_parameters(self):
        list_params = [self.P, self.Q, self.B] if self.factorized else [self.P]
        for weight_list in list_params:
            for weight in weight_list:
                init.kaiming_uniform_(weight, a=math.sqrt(5))

        if not self.factorized:
            for i in range(self.number_of_weights):
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.P[i])
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(self.B[i], -bound, bound)

    def forward(self):
        if self.factorized:
            memory = self.generator(self.memory.unsqueeze(1))
            bias = [torch.matmul(memory, self.B[i]).squeeze(1) for i in range(self.number_of_weights)]
            memory = memory.view(self.num_nodes, self.mem_dim, self.mem_dim)
            weights = [torch.matmul(torch.matmul(self.P[i], memory), self.Q[i]) for i in range(self.number_of_weights)]
            return weights, bias
        else:
            return self.P, self.B



class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

def build_sparse_mask(q_len, k_len, window_size, device):
    mask = torch.full((q_len, k_len), float('-inf'), device=device)
    for i in range(q_len):
        # 不能让 start ≥ k_len
        if i < k_len:
            start = max(0, i - window_size)
            end = min(k_len, i + window_size + 1)
            mask[i, start:end] = 0
        else:
            # q 超出 k 的索引范围，无效 attention，直接给 0 让它不 crash（也可以直接跳过）
            mask[i, :] = 0
    return mask.unsqueeze(0).unsqueeze(0)

class SparseScaledDotProductAttention(nn.Module):
    r"""Sparse Scaled Dot-Product Attention with sliding window mask.
    只计算每个位置在其局部窗口内的注意力。

    参考论文:
      Longformer: The Long-Document Transformer (Beltagy et al., 2020)

    参数:
      d_model: 输入特征维度
      n_heads: 注意力头数
      window_size: 每个查询位置左右各允许的最大位置数（注意力窗口半径）
      attn_dropout: 注意力分数的 dropout 概率
      res_attention: 是否使用残差注意力（同标准实现）
      lsa: 是否让 scale 可训练（Locality Self-Attention）
    """
    def __init__(self, d_model, n_heads, window_size, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        self.window_size = window_size
        head_dim = d_model // n_heads
        # 如果 lsa 为 True，则 scale 为可训练参数；否则固定。
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa



    def forward(self, q, k, v, prev=None, key_padding_mask=None, attn_mask=None):
        batch, n_heads, q_len, _ = q.size()
        k_len = k.size(-2)

        # 防止 nan
        if torch.isnan(q).any() or torch.isnan(k).any() or torch.isnan(v).any():
            print("Input contains NaNs")

        # 注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_scores = torch.clamp(attn_scores, min=-1e4, max=1e4)

        # 稀疏 mask
        device = q.device
        # mask = torch.full((q_len, k_len), float('-inf'), device=device)
        # for i in range(q_len):
        #     start = max(0, i - self.window_size)
        #     end = min(k_len, i + self.window_size + 1)
        #     mask[i, start:end] = 0
        # mask = mask.unsqueeze(0).unsqueeze(0)

        mask = build_sparse_mask(q_len, k_len, self.window_size, device)
        attn_scores = attn_scores + mask


        if prev is not None:
            attn_scores += prev
        if attn_mask is not None:
            attn_scores += attn_mask if attn_mask.dtype != torch.bool else attn_mask.masked_fill(attn_mask,
                                                                                                 -float('inf'))
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), -float('inf'))

        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=1e4, neginf=-1e4)
        attn_weights = self.attn_dropout(attn_weights)

        # Output
        output = torch.matmul(attn_weights, v)
        output = torch.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4)

        return output, attn_weights
