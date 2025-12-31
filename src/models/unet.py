import torch
import torch.nn as nn
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, time_emb_dim, dropout=0.0):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_c),
            nn.SiLU(),
            nn.Conv2d(in_c, out_c, 3, padding=1)
        )
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_c)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_c),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_c, out_c, 3, padding=1)
        )
        self.shortcut = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x, t_emb):
        h = self.block1(x)
        # Add time embedding (broadcast over spatial dims)
        h += self.time_mlp(t_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.shortcut(x)

class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.GroupNorm(32, dim)
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        q = q.reshape(B, C, -1).permute(0, 2, 1)
        k = k.reshape(B, C, -1)
        v = v.reshape(B, C, -1).permute(0, 2, 1)
        
        attn = torch.bmm(q, k) * (C ** -0.5)
        attn = attn.softmax(dim=-1)
        
        h = torch.bmm(attn, v).permute(0, 2, 1).reshape(B, C, H, W)
        return x + self.proj(h)

class UNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        in_c = config['model']['in_channels']
        base_c = config['model']['base_channels']
        mults = config['model']['channel_mults']
        attn_res = config['model']['attention_resolutions']
        n_res = config['model']['num_res_blocks']
        drop = config['model']['dropout']

        # Time Embedding
        time_dim = base_c * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_c),
            nn.Linear(base_c, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Downsampling
        self.conv_in = nn.Conv2d(in_c, base_c, 3, padding=1)
        
        self.downs = nn.ModuleList()
        ch = base_c
        chs = [base_c]
        
        for i, mult in enumerate(mults):
            out_ch = base_c * mult
            for _ in range(n_res):
                layers = [ResidualBlock(ch, out_ch, time_dim, drop)]
                if i in attn_res:
                    layers.append(AttentionBlock(out_ch))
                self.downs.append(nn.ModuleList(layers))
                ch = out_ch
                chs.append(ch)
            if i != len(mults) - 1:
                self.downs.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                chs.append(ch)

        # Mid block
        self.mid_block1 = ResidualBlock(ch, ch, time_dim, drop)
        self.mid_attn = AttentionBlock(ch)
        self.mid_block2 = ResidualBlock(ch, ch, time_dim, drop)

        # Upsampling
        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(mults))):
            out_ch = base_c * mult
            for _ in range(n_res + 1):
                layers = [ResidualBlock(ch + chs.pop(), out_ch, time_dim, drop)]
                if i in attn_res:
                    layers.append(AttentionBlock(out_ch))
                self.ups.append(nn.ModuleList(layers))
                ch = out_ch
            if i != 0:
                self.ups.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))

        self.norm_out = nn.GroupNorm(32, ch)
        self.conv_out = nn.Conv2d(ch, config['model']['out_channels'], 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x, t):
        # Time Embedding
        t = self.time_mlp(t)
        
        # Initial Conv
        h = self.conv_in(x)
        hs = [h]
        
        # Down
        for layer in self.downs:
            if isinstance(layer, nn.ModuleList):
                for sublayer in layer:
                    if isinstance(sublayer, ResidualBlock):
                        h = sublayer(h, t)
                    else:
                        h = sublayer(h)
                hs.append(h)
            else: # Downsample conv
                h = layer(h)
                hs.append(h)
                
        # Mid
        h = self.mid_block1(h, t)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t)
        
        # Up
        for layer in self.ups:
            if isinstance(layer, nn.ModuleList):
                h_skip = hs.pop()
                h = torch.cat((h, h_skip), dim=1)
                for sublayer in layer:
                    if isinstance(sublayer, ResidualBlock):
                        h = sublayer(h, t)
                    else:
                        h = sublayer(h)
            else: # Upsample conv
                h = layer(h)
                
        h = self.act(self.norm_out(h))
        return self.conv_out(h)