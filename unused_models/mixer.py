from einops.layers.torch import Rearrange
from torch import nn

class MlpBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.layers(x)

class MixerBlock(nn.Module):
    def __init__(self, dim, num_patches, token_dim, channel_dim, dropout=0.):
        super().__init__()

        self.token_mixer = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            MlpBlock(num_patches, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )
        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(dim),
            MlpBlock(dim, channel_dim, dropout)
        )

    def forward(self, x):
        x = x + self.token_mixer(x)
        x = x + self.channel_mixer(x)
        return x

class MlpMixer(nn.Module):
    def __init__(self, dim=512, n_layers=8, token_dim=256, channel_dim=2048, patch_size=16, image_size=224):
        super().__init__()
        
        self.num_pathes = (image_size // patch_size) ** 2

        self.make_patches = nn.Sequential(
            nn.Conv2d(1, dim, patch_size, patch_size),
            Rearrange('b c h w -> b (h w) c')
        )

        self.mixers = nn.Sequential(*[
            MixerBlock(dim, self.num_pathes, token_dim, channel_dim)
            for _ in range(n_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, 5)

    def forward(self, x):
        x = self.make_patches(x)
        x = self.layer_norm(self.mixers(x)).mean(1)
        return self.classifier(x)