import abc
import logging
from collections import OrderedDict
from functools import partial
from typing import Callable, List, Mapping, Any

import numpy as np
import torch
from autologging import logged
from torch import nn
from torchvision.models.vision_transformer import EncoderBlock
from torchvision.ops import Conv3dNormActivation


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@logged
class UnetREncoder(nn.Module):
    def __init__(self,
                 seq_length: int,
                 num_layers: int,
                 num_heads: int,
                 hidden_dim: int,
                 mlp_dim: int,
                 dropout: float,
                 attention_dropout: float,
                 norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.pos_embedding = nn.Parameter(
            torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)

        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )

        self.layers = nn.ModuleDict(
            layers
        )
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3,
                      f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        input = self.dropout(input)
        intermediary_results = []
        tmp = input
        for layer_name, layer in self.layers.items():
            self.__log.debug(f"layer_name: {layer_name}")
            tmp = layer(tmp)
            intermediary_results.append(tmp)
        return self.ln(tmp), intermediary_results


def get_norm_layer(normalization: str):
    if normalization == "batch_norm":
        normal_layer = torch.nn.BatchNorm3d
    elif normalization == "instance":
        normal_layer = torch.nn.InstanceNorm3d
    else:
        raise NotImplementedError("")
    return normal_layer


@logged
class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size=(3, 3, 3),
                 normalization: str = "batch_norm"):
        super().__init__()
        norm_layer = get_norm_layer(normalization)
        self.inner = nn.Sequential(
            Conv3dNormActivation(in_channels, out_channels, padding="same",
                                 kernel_size=kernel_size, norm_layer=norm_layer),
            Conv3dNormActivation(out_channels, out_channels, padding="same",
                                 kernel_size=kernel_size, norm_layer=norm_layer),
        )

    def forward(self, input: torch.Tensor):
        return self.inner(input)


@logged
class DeConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 deconv_kernel_size=(2, 2, 2),
                 kernel_size=(3, 3, 3),
                 normalization: str = "batch_norm"):
        super().__init__()
        self.deconv = torch.nn.ConvTranspose3d(in_channels, in_channels, deconv_kernel_size,
                                               stride=(2, 2, 2))
        self.conv = torch.nn.Conv3d(in_channels, out_channels, kernel_size, padding="same")
        self.norm = get_norm_layer(normalization)(out_channels)
        self.activation = torch.nn.ReLU()

    def forward(self, input: torch.Tensor):
        return self.activation(self.norm(self.conv(self.deconv(input))))


class UnetRDecoder(nn.Module):

    def __init__(self,
                 hidden_dim=768,
                 feature_sz: int = 16,
                 normalization: str = "batch_norm"):
        super().__init__()
        self.z2_up_sample = torch.nn.ConvTranspose3d(
            hidden_dim, feature_sz * 8,
            (2, 2, 2),
            stride=(2, 2, 2),
        )
        self.z9_up_sample = DeConvBlock(hidden_dim, feature_sz * 8, normalization=normalization)
        self.z9_conv_block = nn.Sequential(
            ConvBlock(feature_sz * 16, feature_sz * 4, normalization=normalization),
            DeConvBlock(feature_sz * 4, feature_sz * 4, normalization=normalization)
        )
        self.z6_up_sample = nn.Sequential(
            DeConvBlock(hidden_dim, feature_sz * 8, normalization=normalization),
            DeConvBlock(feature_sz * 8, feature_sz * 4, normalization=normalization),

        )

        self.z6_conv_block = nn.Sequential(
            ConvBlock(feature_sz * 8, feature_sz * 2, normalization=normalization),
            DeConvBlock(feature_sz * 2, feature_sz * 2, normalization=normalization)

        )

        self.z3_up_sample = nn.Sequential(
            DeConvBlock(hidden_dim, feature_sz * 8, normalization=normalization),
            DeConvBlock(feature_sz * 8, feature_sz * 4, normalization=normalization),
            DeConvBlock(feature_sz * 4, feature_sz * 2, normalization=normalization),

        )
        self.z3_conv_block = nn.Sequential(
            ConvBlock(feature_sz * 4, feature_sz, normalization=normalization),
            DeConvBlock(feature_sz, feature_sz, normalization=normalization)

        )

    def forward(self, intermediary_results: List[torch.Tensor]):
        z3, z6, z9, z12 = (intermediary_results[2],
                           intermediary_results[5],
                           intermediary_results[8],
                           intermediary_results[11])

        z12_up_sampled = self.z2_up_sample(z12)
        z9_up_sampled = self.z9_up_sample(z9)
        z_12_9_concat = self.z9_conv_block(torch.concat([z12_up_sampled, z9_up_sampled], dim=1))
        z_6_up_sampled = self.z6_up_sample(z6)
        z_12_9_6_concat = self.z6_conv_block(torch.concat([z_12_9_concat, z_6_up_sampled], dim=1))
        z_3_up_sampled = self.z3_up_sample(z3)

        z_12_9_6_3 = self.z3_conv_block(torch.concat([z_12_9_6_concat, z_3_up_sampled], dim=1))
        return z_12_9_6_3


class AbstractPretrainableModel(abc.ABC, nn.Module):

    @abc.abstractmethod
    def filter_not_pretrainable(self, pretrained_state_dict: Mapping[str, Any]) -> Mapping[
        str, Any]:
        raise NotImplementedError()

    def load_from_pretrained(self, pretrained_w_path: str, map_location):
        pretrained_state_dict = torch.load(pretrained_w_path, map_location=map_location)
        pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if
                                 k in self.state_dict()}
        pretrained_state_dict = self.filter_not_pretrainable(pretrained_state_dict)
        logging.info(f"Loaded {len(pretrained_state_dict)} layers")
        self.load_state_dict(pretrained_state_dict, strict=False)


class UnetR(AbstractPretrainableModel):
    def filter_not_pretrainable(self, pretrained_state_dict: Mapping[str, Any]) -> Mapping[
        str, Any]:
        return {k: v for k, v in pretrained_state_dict.items() if "segmentation_head" not in k}

    def __init__(self, nb_classes: int = 1,
                 hidden_dim=768,
                 patch_size=16,
                 input_dim=96,
                 mlp_dim=3072,
                 feature_sz: int = 16,
                 normalization: str = "batch_norm"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.input_dim = input_dim

        self.projection_module = nn.Sequential(
            Conv3dNormActivation(1, hidden_dim, kernel_size=(3, 3, 3),
                                 activation_layer=None, stride=self.patch_size,
                                 norm_layer=get_norm_layer(normalization)),
        )
        logging.info(f"projection_module params: {count_parameters(self.projection_module)}")

        self.input_conv_block = ConvBlock(1, feature_sz)
        logging.info(f"input_conv_block params: {count_parameters(self.input_conv_block)}" )

        length = (input_dim // patch_size) ** 3
        self.encoder = UnetREncoder(seq_length=length,
                                    num_layers=12,
                                    num_heads=12,
                                    hidden_dim=hidden_dim,
                                    mlp_dim=mlp_dim,
                                    dropout=0,
                                    attention_dropout=0,
                                    )

        logging.info(f"encoder params: {count_parameters(self.encoder)}")
        self.decoder = UnetRDecoder(hidden_dim, feature_sz=feature_sz, normalization=normalization)
        logging.info(f"decoder params: {count_parameters(self.decoder)}")

        self.segmentation_head = nn.Sequential(
            ConvBlock(feature_sz * 2, feature_sz * 4, normalization=normalization),
            Conv3dNormActivation(feature_sz * 4, nb_classes, padding="same", kernel_size=1,
                                 norm_layer=None,
                                 activation_layer=None),
        )
        logging.info(f"segmentation_head params: {count_parameters(self.segmentation_head)}")

    def project_input(self, input: torch.Tensor):
        n, c, d, h, w = input.shape
        x = self.projection_module(input)

        n_d = d // self.patch_size
        n_w = w // self.patch_size
        n_h = h // self.patch_size
        x = x.reshape(n, self.hidden_dim, n_h * n_w * n_d)
        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)
        return x

    def forward(self, input: torch.Tensor):
        n, c, d, h, w = input.shape
        encoder_input = self.project_input(input)
        _, intermediary_results = self.encoder(encoder_input)
        intermediary_results = [x.permute(0, 2, 1).reshape((n,
                                                            self.hidden_dim,
                                                            self.input_dim // self.patch_size,
                                                            self.input_dim // self.patch_size,
                                                            self.input_dim // self.patch_size)) for
                                x in intermediary_results]
        decoded = self.decoder(intermediary_results)
        concat = torch.concat([decoded, self.input_conv_block(input)], dim=1)
        res = self.segmentation_head(concat)
        return res


if __name__ == '__main__':
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    t = torch.tensor(np.zeros(shape=(1, 1, 96, 96, 96), dtype=np.float32))
    m = UnetR(nb_classes=16, feature_sz=16, mlp_dim=1536, normalization="instance")
    r = m.forward(t)
    print(count_parameters(m))
    print(r.shape)
    # t = torch.tensor(np.zeros(shape=(1, 2, 768), dtype=np.float32))
    #
    # m = UnetREncoder(seq_length=2,
    #                  num_layers=12,
    #                  num_heads=12,
    #                  hidden_dim=768,
    #                  mlp_dim=3072,
    #                  dropout=0,
    #                  attention_dropout=0)
    # res, inter = m(t)
    #
    #
    # inter = [
    #     torch.tensor(np.zeros(shape=(1, 768, 6, 6, 6), dtype=np.float32)) for i in range(12)
    #
    # ]
    # res = UnetREncoderDecoder()(inter)
    # print(res.shape)

    # print(res.shape, len(inter))
    # m.train()
    # for i in tqdm(range(1000)):
    #     x = m(t)
    #     loss = x.flatten().sum()
    #     optimizer.step()
    #     optimizer.zero_grad()
