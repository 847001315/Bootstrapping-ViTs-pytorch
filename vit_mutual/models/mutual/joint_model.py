import math
import logging
from typing import Any, Dict, Callable, List, Tuple
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from cv_lib.utils import MidExtractor

from vit_mutual.models.vision_transformers import ViT
from vit_mutual.models.transformer.transformer import MLP, MultiHeadSelfAttention
from vit_mutual.models.cnn.blocks import CNNBlock
from vit_mutual.models.cnn.blocks.conv import conv_2d, Conv_2d
from vit_mutual.models.layers import get_activation_fn, Norm_fn
from vit_mutual.models.cnn import get_input_proj


class SharedConv(nn.Module):
    def __init__(self, mhsa: MultiHeadSelfAttention):
        super().__init__()
        self.mhsa = mhsa

        self.kernel_size = math.ceil(math.sqrt(mhsa.num_heads))
        self.phi: torch.Tensor = None
        self.last_shape: Tuple[int, int] = None

    def get_phi(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        phi = Conv_2d.get_phi(
            shape=shape,
            device=device,
            kernel_size=(self.kernel_size, self.kernel_size),
            flatten=False
        )
        phi = phi[:self.mhsa.num_heads]
        phi = phi.permute(1, 0, 2).flatten(1)
        return phi

    def forward(self, x: torch.Tensor):
        H = self.mhsa.num_heads
        d_k = self.mhsa.head_dim

        # [Hxd_k, d] -> [H, d_k, d]
        weight_v = self.mhsa.get_weight_v().unflatten(0, (H, d_k))
        # [d, Hxd_k] -> [d, H, d_k] -> [H, d, d_k]
        weight_o = self.mhsa.get_weight_o().unflatten(1, (H, d_k)).transpose(0, 1)
        weights = torch.bmm(weight_o, weight_v)

        shape = x.shape[2:]
        if self.last_shape == shape and self.phi is not None:
            phi = self.phi
        else:
            phi = self.get_phi(shape, x.device)
            self.last_shape = shape
            self.phi = phi
        return conv_2d(x, phi=self.phi, weights=weights, bias=self.mhsa.get_bias_o())


class SharedLinearProjection(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias

    def forward(self, x: torch.Tensor):
        # reshape as [out_dim, in_dim, 1, 1]
        return F.conv2d(x, self.weight[..., None, None], self.bias)


class SharedMLP(nn.Module):
    def __init__(self, mlp: MLP):
        super().__init__()
        self.linear1 = SharedLinearProjection(mlp.linear1)
        self.linear2 = SharedLinearProjection(mlp.linear2)
        self.dropout = deepcopy(mlp.dropout)
        self.activation = deepcopy(mlp.activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return x


class MutualCNN(nn.Module): # 这里在我看来，还只是在弄CNN，看上去没什么特殊的。用于下面的联合
    def __init__(
        self,
        input_proj: nn.Module,
        conv_blocks: nn.ModuleList,
        mlp_blocks: nn.ModuleList,
        embed_dim: int,
        num_classes: int,
        norm_fn: Callable[[Dict[str, Any]], nn.Module],
        activation: str = "relu",
        dropout: float = None,
        down_sample_layers: List[int] = list(),
        pre_norm: bool = True,
    ):
        super().__init__()
        self.input_proj = input_proj
        layers = nn.ModuleList()
        for b1, b2 in zip(conv_blocks, mlp_blocks):
            block = CNNBlock(
                embed_dim=embed_dim,
                conv_block=b1,
                mlp_block=b2,
                norm=norm_fn,
                dropout=dropout,
                pre_norm=pre_norm
            )
            layers.append(block)
        self.layers = layers
        self.bn = nn.BatchNorm2d(embed_dim)
        self.activation = get_activation_fn(activation)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(embed_dim, num_classes)

        self.downsample = nn.ModuleDict()
        for layer_id in range(len(self.layers)):
            if layer_id in down_sample_layers:
                layer = nn.AvgPool2d(kernel_size=2, stride=2)
            else:
                layer = nn.Identity()
            self.downsample[f"{layer_id}"] = layer

    def forward(self, x: torch.Tensor):
        x = self.input_proj(x)
        for layer_id, layer in enumerate(self.layers):
            x = layer(x)
            x = self.downsample[f"{layer_id}"](x)
        x = self.activation(self.bn(x))
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        pred = self.linear(x)
        return pred


class JointModel(nn.Module): # 按照名字，这里应该要和VIT结合了才对
    def __init__(
        self,
        vit: ViT,
        embed_dim: int,
        input_proj_cfg: Dict[str, Any],
        norm_cfg: Dict[str, Any],
        activation: str = "relu",
        dropout: float = None,
        extract_cnn: List[str] = list(), # 这里的值具体是什么，得看JointModel初始化的值
        extract_vit: List[str] = list(), # 这里有提取出来的vit
        down_sample_layers: List[int] = list(), # 下采样层
        cnn_pre_norm: bool = True, # cnn的预备训练的归一化？
        **kwargs
    ):
        super().__init__()
        # input embedding layer 输入的embedding
        input_proj = get_input_proj(embed_dim, input_proj_cfg) # 根据输入的embedding和input的cfg，得到input_proj
        # share layers 
        mhsa = [SharedConv(mhsa) for mhsa in vit.get_mhsa()] # 把vit中的get_mhsa拿出来，放进sharedConv函数中，得到了列表mhsa
        mlp = [SharedMLP(mlp) for mlp in vit.get_mlp()] # mlp也同理，从vit中拿。
        norm_fn = Norm_fn(norm_cfg)

        cnn = MutualCNN( # 这里是直接调用上面MutualCNN（相互CNN），很好理解，和论文里面一样，让它们两个保持一致
            input_proj=input_proj,
            conv_blocks=nn.ModuleList(mhsa), # 把上面mhsa和mlp的block拿下来，当作CNN的卷积
            mlp_blocks=nn.ModuleList(mlp), # 和CNN的mlp，让他们层数一致
            embed_dim=embed_dim, 
            num_classes=vit.num_classes,
            norm_fn=norm_fn,
            activation=activation,
            dropout=dropout,
            down_sample_layers=down_sample_layers,
            pre_norm=cnn_pre_norm
        )
        self.models = nn.ModuleDict() # 
        self.extractors: Dict[str, MidExtractor] = OrderedDict() # 声明一个有序字典，其中key为str，value为MidExtractor
        self.models["cnn"] = cnn
        self.extractors["cnn"] = MidExtractor(self.models["cnn"], extract_cnn) # 这是提取出来的cnn的东西，这边的话，声明后的默认值为
        self.models["vit"] = vit
        self.extractors["vit"] = MidExtractor(self.models["vit"], extract_vit) # 这是提取出来的vit的东西，可以这么理解？

    def forward(self, x: torch.Tensor): # 我个人可以认为这个x就是输入的图片。
        preds = OrderedDict() # 声明一个有序字典
        mid_features = OrderedDict() # 也声明一个中间态的有序字典
        for name, model in self.models.items(): # 即由cnn和vit组成，其中
            preds[name] = model(x) # 当name为cnn时，得到一个preds[cnn]，当name为vit时，得到preds[vit]。所以最后结果时得到[preds[cnn],preds[vit]]
            mid_features[name] = self.extractors[name].features # 这是中间态，看样子时调用MidExtractor.features。这里为啥没将x输入进来？我以为这里应该会保存中间态的特征图数值，但是按照它写的代码，我目前感觉好像它
            # 只实现了拿出默认值的feature的效果，莫非它在执行model的时候，就实现了这个feature的效果？
        ret = {
            "preds": preds,
            "mid_features": mid_features
        } # 得到一个ret的list，其中preds由[preds[cnn],preds[vit]]组成。mid_features由 两个中间态的[MidExtractor(self.models["cnn"], extract_cnn),MidExtractor(self.models["vit"], extract_vit)]组成。
        return ret


