import numpy as np
import torch
from torch import nn
from tqdm import tqdm


class AttentionLayer(nn.Module):
    def __init__(self, d: int = 768, dh: int = 64):
        super().__init__()
        self.dh = dh
        self.U = nn.Linear(d, dh * 3, bias=False)
        self.sm = nn.Softmax(-1)
        self.scaling_factor = 1 / np.sqrt(dh)

    def forward(self, x):
        x = self.U(x)
        q, k, v = x[:, :, :self.dh], x[:, :, self.dh:2 * self.dh], x[:, :, self.dh * 2:self.dh * 3]
        A = torch.bmm(q, torch.transpose(k, 1, 2))
        A = self.sm(A) * self.scaling_factor
        out = torch.bmm(A, v)
        return out


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, nb_heads=12):
        super().__init__()
        self.heads = nn.ModuleList([])
        for i in range(nb_heads):
            self.heads.append(AttentionLayer())

    def forward(self, x):
        return torch.concat([head(x) for head in self.heads], dim=2)


class TransformEncoderBlock(nn.Module):
    def __init__(self, dims=768, dim: int = 768, mpl_hidden: int = 3072):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(dims),
            nn.Linear(in_features=dim, out_features=mpl_hidden, bias=False),
            nn.GELU(),
            nn.Linear(in_features=mpl_hidden, out_features=dim, bias=False),
            nn.GELU()
        )
        self.norm_l = nn.LayerNorm(dims)
        self.multi_attention_head = MultiHeadAttentionLayer()

    def forward(self, x):
        x = self.norm_l(x)
        ax = self.multi_attention_head(x)
        x = ax + x
        x = self.mlp(x)
        return x


class PatchEmbedder(nn.Module):
    def __init__(self, nb_feature=768, patch_size=16):
        super().__init__()
        self.embedder = nn.Sequential(
            nn.Linear(patch_size * patch_size * 3, nb_feature, bias=False),
            nn.LayerNorm(nb_feature)
        )

    def forward(self, x):
        return self.embedder(x)


class TransformEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_blocks = nn.ModuleList([])
        [self.encoder_blocks.append(TransformEncoderBlock()) for i in range(12)]

    def forward(self, x):
        for block in self.encoder_blocks:
            x = block(x)

        return x


class ClassificationHead(nn.Module):
    def __init__(self, dims=768, mpl_hidden=350):
        super().__init__()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dims),
            nn.Linear(in_features=dims, out_features=mpl_hidden, bias=False),
            nn.GELU(),
            nn.Linear(in_features=mpl_hidden, out_features=mpl_hidden, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.mlp_head(x)


class Vit(nn.Module):
    def __init__(self, patch_size: int = 16, channels=3, dim=768, nb_patches=196,
                 nb_patches_col=14):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(1, nb_patches + 1, dim))
        self.class_embedding = nn.Parameter(torch.randn(1, 1, dim))

        self.nb_patches = nb_patches
        self.patch_size = patch_size
        self.channels = channels
        self.nb_patches_col = nb_patches_col
        self.patch_embedder = PatchEmbedder()
        self.transform_encoder = TransformEncoder()
        self.class_head = ClassificationHead()

    def forward(self, x):
        # batch_size, channels, h, w
        inputs = torch.tensor(np.zeros(
            (x.shape[0], self.nb_patches, self.patch_size * self.patch_size * self.channels),
            dtype=np.float32))
        # print("inputs.shape", inputs.shape)
        for j in range(self.nb_patches_col):
            for i in range(self.nb_patches_col):
                patch_number = i + j * (self.nb_patches_col - 1)
                # print(i * self.patch_size, (i + 1) * self.patch_size)
                patch = x[:,
                        :,
                        i * self.patch_size:(i + 1) * self.patch_size,
                        j * self.patch_size:(j + 1) * self.patch_size].flatten(start_dim=1,
                                                                               end_dim=3)
                # print("i", i, "j", j, "patch_number", patch_number, patch.shape)
                inputs[:, patch_number, :] = patch
        inputs = self.patch_embedder(inputs)
        inputs = torch.concat([inputs, self.class_embedding], dim=1)
        x = inputs + self.positional_embedding
        x = self.transform_encoder(x)
        return self.class_head(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    t = torch.tensor(np.zeros(shape=(1, 3, 224, 244), dtype=np.float32))
    m = Vit(    )
    print("parameters", count_parameters(m))
    print(m(t).shape)

    optimizer= torch.optim.Adam(m.parameters())
    m.train()
    for i in tqdm(range(1000)):
        x = m(t)
        loss = x.flatten().sum()
        optimizer.step()
        optimizer.zero_grad()
