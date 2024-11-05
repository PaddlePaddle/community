> 1. 原则上请使用中文。
> 2. 侧重阐述设计思路而不只是实现方案细节，体现对方案选型的利弊考量（必要的需要有预调研实验数据支撑），特别是对框架和用户两个维度的影响。
> 3. 多利用图表来阐述设计思路。

# 标题

|任务名称 | Hackathon 7th PPSCI No.2 | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | LilaKen | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2024-10-28 | 
|版本号 | 此设计文档的版本号，如V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | 如无特殊情况，都应基于develop版本开发 | 
|文件名 | 提交的markdown设计文档文件名称，如：20241028_design_transolver.md<br> | 

# 一、概述
## 1、相关背景
求解偏微分方程（PDE）是科学与工程共有的基础性问题，对材料分析、工业设计、气象预报等重大需求至关重要。

## 2、功能目标
复现以下基于Transolver方法的PDE数据训练推理代码：
Standard benchmarks: see ./PDE-Solving-StandardBenchmark
Car design task: see ./Car-Design-ShapeNetCar
Airfoil design task: see ./Airfoil-Design-AirfRANS
精度与论文中对齐

## 3、意义
在实际应用中PDE通常被离散化到大规模网格上，再使用经典的数值方法进行求解，但是往往需要数小时甚至数天才能完成一个复杂几何结构的仿真模拟。近期，深度模型在PDE高效求解上展现了巨大潜力。得益于强大的非线性拟合能力，它可以从数据中学习从几何结构到物理场的映射，并以极快的速度（秒级）完成推理仿真。
Transolver抛弃冗余并且流于表面的网格，我们提出学习几何结构背后内在的物理状态，并在物理状态间应用注意力机制，天然具备线性复杂度和几何结构通用性；在6个标准PDE数据集上平均比之前的SOTA误差降低22%，在大规模工业仿真场景中表现最优；展现了优秀的计算效率，可扩展性（Scalability）以及分布外泛化能力（OOD Generalizability）。

# 二、飞桨现状
飞桨框架目前支持Transolver复现，已基于paddle框架复现跑通Standard benchmarks中AIRFOIL、PIPE、DARCY、ELASTICITY四个数据集ing。


# 三、业内方案调研
源代码基于Pytorch框架构建Physics Attention
```
import torch.nn as nn
import torch
from einops import rearrange, repeat


class Physics_Attention_Irregular_Mesh(nn.Module):
    ## for irregular meshes in 1D, 2D or 3D space

    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # B N C
        B, N, C = x.shape

        ### (1) Slice
        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)


class Physics_Attention_Structured_Mesh_2D(nn.Module):
    ## for structured mesh in 2D space

    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64, H=101, W=31, kernel=3):  # kernel=3):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.H = H
        self.W = W

        self.in_project_x = nn.Conv2d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_fx = nn.Conv2d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # B N C
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).contiguous().permute(0, 3, 1, 2).contiguous()  # B C H W

        ### (1) Slice
        fx_mid = self.in_project_fx(x).permute(0, 2, 3, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid = self.in_project_x(x).permute(0, 2, 3, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N G
        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5))  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)


class Physics_Attention_Structured_Mesh_3D(nn.Module):
    ## for structured mesh in 3D space

    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=32, H=32, W=32, D=32, kernel=3):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.H = H
        self.W = W
        self.D = D

        self.in_project_x = nn.Conv3d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_fx = nn.Conv3d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # B N C
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, self.D, C).contiguous().permute(0, 4, 1, 2, 3).contiguous()  # B C H W

        ### (1) Slice
        fx_mid = self.in_project_fx(x).permute(0, 2, 3, 4, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid = self.in_project_x(x).permute(0, 2, 3, 4, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N G
        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5))  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)
```
源代码基于Pytorch框架构建Transolver 3D Shape-Net-Car数据集
```
import torch
import numpy as np
import torch.nn as nn
from timm.models.layers import trunc_normal_
from einops import rearrange, repeat

ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}


class Physics_Attention_Irregular_Mesh(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # B N C
        B, N, C = x.shape

        ### (1) Slice
        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class Transolver_block(nn.Module):
    """Transformer encoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            slice_num=32,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_Irregular_Mesh(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                                     dropout=dropout, slice_num=slice_num)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class Model(nn.Module):
    def __init__(self,
                 space_dim=1,
                 n_layers=5,
                 n_hidden=256,
                 dropout=0,
                 n_head=8,
                 act='gelu',
                 mlp_ratio=1,
                 fun_dim=1,
                 out_dim=1,
                 slice_num=32,
                 ref=8,
                 unified_pos=False
                 ):
        super(Model, self).__init__()
        self.__name__ = 'UniPDE_3D'
        self.ref = ref
        self.unified_pos = unified_pos
        if self.unified_pos:
            self.preprocess = MLP(fun_dim + self.ref * self.ref * self.ref, n_hidden * 2, n_hidden, n_layers=0,
                                  res=False, act=act)
        else:
            self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)

        self.n_hidden = n_hidden
        self.space_dim = space_dim

        self.blocks = nn.ModuleList([Transolver_block(num_heads=n_head, hidden_dim=n_hidden,
                                                      dropout=dropout,
                                                      act=act,
                                                      mlp_ratio=mlp_ratio,
                                                      out_dim=out_dim,
                                                      slice_num=slice_num,
                                                      last_layer=(_ == n_layers - 1))
                                     for _ in range(n_layers)])
        self.initialize_weights()
        self.placeholder = nn.Parameter((1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float))

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_grid(self, my_pos):
        # my_pos 1 N 3
        batchsize = my_pos.shape[0]

        gridx = torch.tensor(np.linspace(-1.5, 1.5, self.ref), dtype=torch.float)
        gridx = gridx.reshape(1, self.ref, 1, 1, 1).repeat([batchsize, 1, self.ref, self.ref, 1])
        gridy = torch.tensor(np.linspace(0, 2, self.ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.ref, 1, 1).repeat([batchsize, self.ref, 1, self.ref, 1])
        gridz = torch.tensor(np.linspace(-4, 4, self.ref), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, self.ref, 1).repeat([batchsize, self.ref, self.ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy, gridz), dim=-1).cuda().reshape(batchsize, self.ref ** 3, 3)  # B 4 4 4 3

        pos = torch.sqrt(
            torch.sum((my_pos[:, :, None, :] - grid_ref[:, None, :, :]) ** 2,
                      dim=-1)). \
            reshape(batchsize, my_pos.shape[1], self.ref * self.ref * self.ref).contiguous()
        return pos

    def forward(self, data):
        cfd_data, geom_data = data
        x, fx, T = cfd_data.x, None, None
        x = x[None, :, :]
        if self.unified_pos:
            new_pos = self.get_grid(cfd_data.pos[None, :, :])
            x = torch.cat((x, new_pos), dim=-1)

        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
            fx = fx + self.placeholder[None, None, :]

        for block in self.blocks:
            fx = block(fx)

        return fx[0]

```

基于Paddle框架复现代码构建Physics Attention

```
import sys
sys.path.append('/ssd1/ken/Transolver-paddle-convert-main/utils')
import paddle_aux
import paddle
from einops import rearrange, repeat


class Physics_Attention_Irregular_Mesh(paddle.nn.Layer):

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = paddle.nn.Softmax(axis=-1)
        self.dropout = paddle.nn.Dropout(p=dropout)
        self.temperature = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.ones(shape=[1, heads, 1, 1]) * 0.5)
        self.in_project_x = paddle.nn.Linear(in_features=dim, out_features=
            inner_dim)
        self.in_project_fx = paddle.nn.Linear(in_features=dim, out_features
            =inner_dim)
        self.in_project_slice = paddle.nn.Linear(in_features=dim_head,
            out_features=slice_num)
        for l in [self.in_project_slice]:
            init_Orthogonal = paddle.nn.initializer.Orthogonal()
            init_Orthogonal(l.weight)
        self.to_q = paddle.nn.Linear(in_features=dim_head, out_features=
            dim_head, bias_attr=False)
        self.to_k = paddle.nn.Linear(in_features=dim_head, out_features=
            dim_head, bias_attr=False)
        self.to_v = paddle.nn.Linear(in_features=dim_head, out_features=
            dim_head, bias_attr=False)
        self.to_out = paddle.nn.Sequential(paddle.nn.Linear(in_features=
            inner_dim, out_features=dim), paddle.nn.Dropout(p=dropout))

    def forward(self, x):
        B, N, C = tuple(x.shape)
        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head
            ).transpose(perm=[0, 2, 1, 3]).contiguous()
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head
            ).transpose(perm=[0, 2, 1, 3]).contiguous()
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.
            temperature)
        slice_norm = slice_weights.sum(axis=2)
        slice_token = paddle.einsum('bhnc,bhng->bhgc', fx_mid, slice_weights)
        slice_token = slice_token / (slice_norm + 1e-05)[:, :, :, None].tile(
            repeat_times=[1, 1, 1, self.dim_head])
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = paddle.matmul(x=q_slice_token, y=k_slice_token.transpose(
            perm=paddle_aux.transpose_aux_func(k_slice_token.ndim, -1, -2))
            ) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = paddle.matmul(x=attn, y=v_slice_token)
        out_x = paddle.einsum('bhgc,bhng->bhnc', out_slice_token, slice_weights
            )
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)


class Physics_Attention_Structured_Mesh_2D(paddle.nn.Layer):

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64,
        H=101, W=31, kernel=3):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = paddle.nn.Softmax(axis=-1)
        self.dropout = paddle.nn.Dropout(p=dropout)
        self.temperature = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.ones(shape=[1, heads, 1, 1]) * 0.5)
        self.H = H
        self.W = W
        self.in_project_x = paddle.nn.Conv2D(in_channels=dim, out_channels=
            inner_dim, kernel_size=kernel, stride=1, padding=kernel // 2)
        self.in_project_fx = paddle.nn.Conv2D(in_channels=dim, out_channels
            =inner_dim, kernel_size=kernel, stride=1, padding=kernel // 2)
        self.in_project_slice = paddle.nn.Linear(in_features=dim_head,
            out_features=slice_num)
        for l in [self.in_project_slice]:
            init_Orthogonal = paddle.nn.initializer.Orthogonal()
            init_Orthogonal(l.weight)
        self.to_q = paddle.nn.Linear(in_features=dim_head, out_features=
            dim_head, bias_attr=False)
        self.to_k = paddle.nn.Linear(in_features=dim_head, out_features=
            dim_head, bias_attr=False)
        self.to_v = paddle.nn.Linear(in_features=dim_head, out_features=
            dim_head, bias_attr=False)
        self.to_out = paddle.nn.Sequential(paddle.nn.Linear(in_features=
            inner_dim, out_features=dim), paddle.nn.Dropout(p=dropout))

    def forward(self, x):
        B, N, C = tuple(x.shape)
        x = x.reshape(B, self.H, self.W, C).contiguous().transpose(perm=[0,
            3, 1, 2]).contiguous()
        fx_mid = self.in_project_fx(x).transpose(perm=[0, 2, 3, 1]).contiguous(
            ).reshape(B, N, self.heads, self.dim_head).transpose(perm=[0, 2,
            1, 3]).contiguous()
        x_mid = self.in_project_x(x).transpose(perm=[0, 2, 3, 1]).contiguous(
            ).reshape(B, N, self.heads, self.dim_head).transpose(perm=[0, 2,
            1, 3]).contiguous()
        slice_weights = self.softmax(self.in_project_slice(x_mid) / paddle.
            clip(x=self.temperature, min=0.1, max=5))
        slice_norm = slice_weights.sum(axis=2)
        slice_token = paddle.einsum('bhnc,bhng->bhgc', fx_mid, slice_weights)
        slice_token = slice_token / (slice_norm + 1e-05)[:, :, :, None].tile(
            repeat_times=[1, 1, 1, self.dim_head])
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = paddle.matmul(x=q_slice_token, y=k_slice_token.transpose(
            perm=paddle_aux.transpose_aux_func(k_slice_token.ndim, -1, -2))
            ) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = paddle.matmul(x=attn, y=v_slice_token)
        out_x = paddle.einsum('bhgc,bhng->bhnc', out_slice_token, slice_weights
            )
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)


class Physics_Attention_Structured_Mesh_3D(paddle.nn.Layer):

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=32,
        H=32, W=32, D=32, kernel=3):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = paddle.nn.Softmax(axis=-1)
        self.dropout = paddle.nn.Dropout(p=dropout)
        self.temperature = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.ones(shape=[1, heads, 1, 1]) * 0.5)
        self.H = H
        self.W = W
        self.D = D
        self.in_project_x = paddle.nn.Conv3D(in_channels=dim, out_channels=
            inner_dim, kernel_size=kernel, stride=1, padding=kernel // 2)
        self.in_project_fx = paddle.nn.Conv3D(in_channels=dim, out_channels
            =inner_dim, kernel_size=kernel, stride=1, padding=kernel // 2)
        self.in_project_slice = paddle.nn.Linear(in_features=dim_head,
            out_features=slice_num)
        for l in [self.in_project_slice]:
            init_Orthogonal = paddle.nn.initializer.Orthogonal()
            init_Orthogonal(l.weight)
        self.to_q = paddle.nn.Linear(in_features=dim_head, out_features=
            dim_head, bias_attr=False)
        self.to_k = paddle.nn.Linear(in_features=dim_head, out_features=
            dim_head, bias_attr=False)
        self.to_v = paddle.nn.Linear(in_features=dim_head, out_features=
            dim_head, bias_attr=False)
        self.to_out = paddle.nn.Sequential(paddle.nn.Linear(in_features=
            inner_dim, out_features=dim), paddle.nn.Dropout(p=dropout))

    def forward(self, x):
        B, N, C = tuple(x.shape)
        x = x.reshape(B, self.H, self.W, self.D, C).contiguous().transpose(perm
            =[0, 4, 1, 2, 3]).contiguous()
        fx_mid = self.in_project_fx(x).transpose(perm=[0, 2, 3, 4, 1]
            ).contiguous().reshape(B, N, self.heads, self.dim_head).transpose(
            perm=[0, 2, 1, 3]).contiguous()
        x_mid = self.in_project_x(x).transpose(perm=[0, 2, 3, 4, 1]
            ).contiguous().reshape(B, N, self.heads, self.dim_head).transpose(
            perm=[0, 2, 1, 3]).contiguous()
        slice_weights = self.softmax(self.in_project_slice(x_mid) / paddle.
            clip(x=self.temperature, min=0.1, max=5))
        slice_norm = slice_weights.sum(axis=2)
        slice_token = paddle.einsum('bhnc,bhng->bhgc', fx_mid, slice_weights)
        slice_token = slice_token / (slice_norm + 1e-05)[:, :, :, None].tile(
            repeat_times=[1, 1, 1, self.dim_head])
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = paddle.matmul(x=q_slice_token, y=k_slice_token.transpose(
            perm=paddle_aux.transpose_aux_func(k_slice_token.ndim, -1, -2))
            ) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = paddle.matmul(x=attn, y=v_slice_token)
        out_x = paddle.einsum('bhgc,bhng->bhnc', out_slice_token, slice_weights
            )
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)

```

基于Paddle框架复现代码构建Transolver 3D Shape-Net-Car数据集

```
import sys
sys.path.append('/ssd1/ken/Transolver-paddle-convert-main/utils')
import paddle_aux
import paddle
import numpy as np
from paddle.nn.initializer import TruncatedNormal, Constant
from einops import rearrange, repeat
ACTIVATION = {'gelu': paddle.nn.GELU, 'tanh': paddle.nn.Tanh, 'sigmoid':
    paddle.nn.Sigmoid, 'relu': paddle.nn.ReLU, 'leaky_relu': paddle.nn.
    LeakyReLU(negative_slope=0.1), 'softplus': paddle.nn.Softplus, 'ELU':
    paddle.nn.ELU, 'silu': paddle.nn.Silu}


class Physics_Attention_Irregular_Mesh(paddle.nn.Layer):

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = paddle.nn.Softmax(axis=-1)
        self.dropout = paddle.nn.Dropout(p=dropout)
        self.temperature = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.ones(shape=[1, heads, 1, 1]) * 0.5)
        self.in_project_x = paddle.nn.Linear(in_features=dim, out_features=
            inner_dim)
        self.in_project_fx = paddle.nn.Linear(in_features=dim, out_features
            =inner_dim)
        self.in_project_slice = paddle.nn.Linear(in_features=dim_head,
            out_features=slice_num)
        for l in [self.in_project_slice]:
            init_Orthogonal = paddle.nn.initializer.Orthogonal()
            init_Orthogonal(l.weight)
        self.to_q = paddle.nn.Linear(in_features=dim_head, out_features=
            dim_head, bias_attr=False)
        self.to_k = paddle.nn.Linear(in_features=dim_head, out_features=
            dim_head, bias_attr=False)
        self.to_v = paddle.nn.Linear(in_features=dim_head, out_features=
            dim_head, bias_attr=False)
        self.to_out = paddle.nn.Sequential(paddle.nn.Linear(in_features=
            inner_dim, out_features=dim), paddle.nn.Dropout(p=dropout))

    def forward(self, x):
        B, N, C = tuple(x.shape)
        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head
            ).transpose(perm=[0, 2, 1, 3]).contiguous()
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head
            ).transpose(perm=[0, 2, 1, 3]).contiguous()
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.
            temperature)
        slice_norm = slice_weights.sum(axis=2)
        slice_token = paddle.einsum('bhnc,bhng->bhgc', fx_mid, slice_weights)
        slice_token = slice_token / (slice_norm + 1e-05)[:, :, :, None].tile(
            repeat_times=[1, 1, 1, self.dim_head])
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = paddle.matmul(x=q_slice_token, y=k_slice_token.transpose(
            perm=paddle_aux.transpose_aux_func(k_slice_token.ndim, -1, -2))
            ) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = paddle.matmul(x=attn, y=v_slice_token)
        out_x = paddle.einsum('bhgc,bhng->bhnc', out_slice_token, slice_weights
            )
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)


class MLP(paddle.nn.Layer):

    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu',
        res=True):
        super(MLP, self).__init__()
        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = paddle.nn.Sequential(paddle.nn.Linear(in_features
            =n_input, out_features=n_hidden), act())
        self.linear_post = paddle.nn.Linear(in_features=n_hidden,
            out_features=n_output)
        self.linears = paddle.nn.LayerList(sublayers=[paddle.nn.Sequential(
            paddle.nn.Linear(in_features=n_hidden, out_features=n_hidden),
            act()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class Transolver_block(paddle.nn.Layer):
    """Transformer encoder block."""

    def __init__(self, num_heads: int, hidden_dim: int, dropout: float, act
        ='gelu', mlp_ratio=4, last_layer=False, out_dim=1, slice_num=32):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = paddle.nn.LayerNorm(normalized_shape=hidden_dim)
        self.Attn = Physics_Attention_Irregular_Mesh(hidden_dim, heads=
            num_heads, dim_head=hidden_dim // num_heads, dropout=dropout,
            slice_num=slice_num)
        self.ln_2 = paddle.nn.LayerNorm(normalized_shape=hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim,
            n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = paddle.nn.LayerNorm(normalized_shape=hidden_dim)
            self.mlp2 = paddle.nn.Linear(in_features=hidden_dim,
                out_features=out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class Model(paddle.nn.Layer):

    def __init__(self, space_dim=1, n_layers=5, n_hidden=256, dropout=0,
        n_head=8, act='gelu', mlp_ratio=1, fun_dim=1, out_dim=1, slice_num=
        32, ref=8, unified_pos=False):
        super(Model, self).__init__()
        self.__name__ = 'UniPDE_3D'
        self.ref = ref
        self.unified_pos = unified_pos
        if self.unified_pos:
            self.preprocess = MLP(fun_dim + self.ref * self.ref * self.ref,
                n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
        else:
            self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2,
                n_hidden, n_layers=0, res=False, act=act)
        self.n_hidden = n_hidden
        self.space_dim = space_dim
        self.blocks = paddle.nn.LayerList(sublayers=[Transolver_block(
            num_heads=n_head, hidden_dim=n_hidden, dropout=dropout, act=act,
            mlp_ratio=mlp_ratio, out_dim=out_dim, slice_num=slice_num,
            last_layer=_ == n_layers - 1) for _ in range(n_layers)])
        self.initialize_weights()
        self.placeholder = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=1 / n_hidden * paddle.rand(shape=n_hidden, dtype='float32'))

    def initialize_weights(self):
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, paddle.nn.Linear):
            trunc_normal = TruncatedNormal(mean=0.0, std=0.02)
            trunc_normal(m.weight)
            if m.bias is not None:
                constant = Constant(value=0.0)
                constant(m.bias)
        elif isinstance(m, (paddle.nn.LayerNorm, paddle.nn.BatchNorm1D)):
            constant = Constant(value=0.0)
            constant(m.bias)
            constant = Constant(value=1.0)
            constant(m.weight)


    def get_grid(self, my_pos):
        batchsize = tuple(my_pos.shape)[0]
        gridx = paddle.to_tensor(data=np.linspace(-1.5, 1.5, self.ref),
            dtype='float32')
        gridx = gridx.reshape(1, self.ref, 1, 1, 1).tile(repeat_times=[
            batchsize, 1, self.ref, self.ref, 1])
        gridy = paddle.to_tensor(data=np.linspace(0, 2, self.ref), dtype=
            'float32')
        gridy = gridy.reshape(1, 1, self.ref, 1, 1).tile(repeat_times=[
            batchsize, self.ref, 1, self.ref, 1])
        gridz = paddle.to_tensor(data=np.linspace(-4, 4, self.ref), dtype=
            'float32')
        gridz = gridz.reshape(1, 1, 1, self.ref, 1).tile(repeat_times=[
            batchsize, self.ref, self.ref, 1, 1])
        grid_ref = paddle.concat(x=(gridx, gridy, gridz), axis=-1).cuda(
            blocking=True).reshape(batchsize, self.ref ** 3, 3)
        pos = paddle.sqrt(x=paddle.sum(x=(my_pos[:, :, None, :] - grid_ref[
            :, None, :, :]) ** 2, axis=-1)).reshape(batchsize, tuple(my_pos
            .shape)[1], self.ref * self.ref * self.ref).contiguous()
        return pos

    def forward(self, data):
        cfd_data, geom_data = data
        x, fx, T = cfd_data.x, None, None
        x = x[None, :, :]
        if self.unified_pos:
            new_pos = self.get_grid(cfd_data.pos[None, :, :])
            x = paddle.concat(x=(x, new_pos), axis=-1)
        if fx is not None:
            fx = paddle.concat(x=(x, fx), axis=-1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
            fx = fx + self.placeholder[None, None, :]
        for block in self.blocks:
            fx = block(fx)
        return fx[0]

```


# 四、对比分析
对第三部分调研的方案进行对比**评价**和**对比分析**，论述各种方案的优劣势。
好像没啥好对比的，只是不同框架

# 五、设计思路与实现方案

### 主体设计具体描述
先复现基于Pytorch版本的Transolver代码，与原论文对齐精度，使用Paconvert转换Pytorch代码为Paddle代码，对其中不支持的部分代码函数进行人工修改，再复现基于Paddle版本的Transolver代码，与原论文和基于Pytorch版本的代码对齐精度，计算相对误差，对于公式(复现指标-源论文指标)/源论文指标<10%,该公式使用的是相对误差，但在某些情况下，绝对误差可能更有意义。例如，当源论文指标非常接近零时，即使是微小的绝对误差也会导致相对误差非常大。当出现微小的绝对误差以及源论文指标数据在以小于5的数字开头的时候，与Paddle团队沟通误差事宜。

### 主体设计选型考量
基于Pytorch版本复现部分数据：
| Model      | Shape-Net-Car |         |        |        |
| ---------- | ------------- | ------- | ------ | ------ |
|            | volume     | surf          | Cd      | ρd     |
| Transolver | 0.0211        | 0.0769  | 0.0123 | 0.9909 |
| 相对误差   | 0.01932       | 0.03221 | 0.1942 | -0.003 |

| Model      | POINT CLOUND | STRUCTURED MESH | REGULAR GRID |               |       |              |
| ---------- | ------------ | --------------- | ------------ | ------------- | ----- | ------------ |
|            | ELASTICITY   | PLASTICITY      | AIRFOIL      | PIPE          | NAVIER-STOKES | DARCY |
| Transolver | 0.0072       | 0.0013          | 0.0049       | 0.0047        |       | 0.0049       |
| 相对误差   | 0.125        | 0.083333333     | -0.075471698 | 0.424242424   |       | -0.140350877 |

源论文指标：
| Model      | Shape-Net-Car |        |        |        |
| ---------- | ------------- | ------ | ------ | ------ |
|            | volume     | surf          | Cd      | ρd     |
| Transolver | 0.0207        | 0.0745 | 0.0103 | 0.9935 |

| Model      | POINT CLOUND | STRUCTURED MESH | REGULAR GRID |               |       |        |
| ---------- | ------------ | --------------- | ------------ | ------------- | ----- | ------ |
|            | ELASTICITY   | PLASTICITY      | AIRFOIL      | PIPE          | NAVIER-STOKES | DARCY |
| Transolver | 0.0064       | 0.0012          | 0.0053       | 0.0033        | 0.09  | 0.0057 |

# 六、测试和验收的考量
正确性验证：验证Transolver Pytorch复现精度与Transolver Paddle复现精度，代码能正常跑通训练、评估。

# 七、影响面
新增Transolver API，对其他模块没有影响

# 八、排期规划
2024/10 完成Transolver Pytorch复现，精度对齐原论文； 2024/11 完成Transolver Paddle复现，精度对齐原论文且完成验收；

# 名词解释

# 附件及参考资料
