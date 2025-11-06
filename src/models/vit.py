import torch
import torch.nn as nn
import math

class PatchEmbed(nn.Module):
    def __init__(self, img_size=128, patch_size=16, in_chans=1, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # B, C, H/ps, W/ps
        x = x.flatten(2).transpose(1, 2)  # B, N, C
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=192, num_heads=3, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim*mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x):
        h = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = x + h
        h = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + h
        return x

class ViTClassifier(nn.Module):
    def __init__(self, img_size=128, patch_size=16, in_chans=1, embed_dim=192, depth=6, num_heads=3, num_classes=4):
        super().__init__()
        self.patch = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
        num_patches = (img_size//patch_size)**2
        self.pos_embed = nn.Parameter(torch.zeros(1, 1+num_patches, embed_dim))
        self.blocks = nn.ModuleList([TransformerEncoder(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.size(0)
        x = self.patch(x)  # B, N, C
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # B, 1+N, C
        x = x + self.pos_embed[:, :x.size(1), :]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x[:,0])  # CLS
        logits = self.head(x)
        return logits
