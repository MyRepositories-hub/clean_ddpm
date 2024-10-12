import math

import torch
from torch import nn


class DiTBlock(nn.Module):
    def __init__(self, emb_size, n_head):
        super().__init__()

        self.emb_size = emb_size
        self.n_head = n_head

        # Conditioning
        self.gamma1 = nn.Linear(emb_size, emb_size)
        self.beta1 = nn.Linear(emb_size, emb_size)
        self.alpha1 = nn.Linear(emb_size, emb_size)
        self.gamma2 = nn.Linear(emb_size, emb_size)
        self.beta2 = nn.Linear(emb_size, emb_size)
        self.alpha2 = nn.Linear(emb_size, emb_size)

        # Layer norm
        self.ln1 = nn.LayerNorm(emb_size)
        self.ln2 = nn.LayerNorm(emb_size)

        # Multi-head self-attention
        self.wq = nn.Linear(emb_size, n_head * emb_size)
        self.wk = nn.Linear(emb_size, n_head * emb_size)
        self.wv = nn.Linear(emb_size, n_head * emb_size)
        self.lv = nn.Linear(n_head * emb_size, emb_size)

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(emb_size, emb_size * 4),
            nn.ReLU(),
            nn.Linear(emb_size * 4, emb_size)
        )

    def forward(self, x, cond):
        gamma1_val = self.gamma1(cond)
        beta1_val = self.beta1(cond)
        alpha1_val = self.alpha1(cond)
        gamma2_val = self.gamma2(cond)
        beta2_val = self.beta2(cond)
        alpha2_val = self.alpha2(cond)
        y = self.ln1(x)
        y = y * (1 + gamma1_val.unsqueeze(1)) + beta1_val.unsqueeze(1)

        # Attention
        q = self.wq(y)
        k = self.wk(y)
        v = self.wv(y)
        q = q.view(q.size(0), q.size(1), self.n_head, self.emb_size).permute(0, 2, 1, 3)
        k = k.view(k.size(0), k.size(1), self.n_head, self.emb_size).permute(0, 2, 3, 1)
        v = v.view(v.size(0), v.size(1), self.n_head, self.emb_size).permute(0, 2, 1, 3)
        attn = q @ k / math.sqrt(q.size(2))
        attn = torch.softmax(attn, dim=-1)
        y = attn @ v
        y = y.permute(0, 2, 1, 3)
        y = y.reshape(y.size(0), y.size(1), y.size(2) * y.size(3))
        y = self.lv(y)

        y = y * alpha1_val.unsqueeze(1)
        y = x + y

        z = self.ln2(y)
        z = z * (1 + gamma2_val.unsqueeze(1)) + beta2_val.unsqueeze(1)
        z = self.ff(z)
        z = z * alpha2_val.unsqueeze(1)
        return y + z


class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(TimeEmbedding, self).__init__()
        self.embedding_dim = embedding_dim

        # Create frequency bands
        self.freq_bands = torch.exp(
            torch.linspace(0., embedding_dim // 2 - 1, embedding_dim // 2) *
            (-torch.log(torch.tensor(10000.0)) / (embedding_dim // 2 - 1))
        )

    def forward(self, t):
        # Expand dimensions and compute sinusoidal embeddings
        t = t.unsqueeze(1)
        freq = self.freq_bands.unsqueeze(0).to(t.device)

        # Calculate sin and cos embeddings
        sinusoid_inp = t * freq
        emb_sin = torch.sin(sinusoid_inp)
        emb_cos = torch.cos(sinusoid_inp)

        # Concatenate sin and cos embeddings
        emb = torch.cat([emb_sin, emb_cos], dim=-1)

        return emb


class DiT(nn.Module):
    def __init__(self, img_size, patch_size, channel, emb_size, label_num, dit_num, head):
        super().__init__()

        self.patch_size = patch_size
        self.patch_count = img_size // self.patch_size
        self.channel = channel
        self.conv = nn.Conv2d(
            in_channels=channel,
            out_channels=channel * patch_size ** 2,
            kernel_size=patch_size,
            padding=0,
            stride=patch_size
        )
        self.patch_emb = nn.Linear(in_features=channel * patch_size ** 2, out_features=emb_size)
        self.patch_pos_emb = nn.Parameter(torch.rand(1, self.patch_count ** 2, emb_size))

        # Time embedding
        self.time_emb = nn.Sequential(
            TimeEmbedding(emb_size),
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size)
        )

        # Label embedding
        self.label_emb = nn.Embedding(num_embeddings=label_num, embedding_dim=emb_size)

        self.dits = nn.ModuleList()
        for _ in range(dit_num):
            self.dits.append(DiTBlock(emb_size, head))
        self.ln = nn.LayerNorm(emb_size)
        self.linear = nn.Linear(emb_size, channel * patch_size ** 2)

    def forward(self, x, t, y):
        # Label embedding
        y_emb = self.label_emb(y)
        # Time embedding
        t_emb = self.time_emb(t)

        # Condition embedding
        cond = y_emb + t_emb

        # Patch embedding
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        x = x.view(x.size(0), self.patch_count * self.patch_count, x.size(3))

        x = self.patch_emb(x)
        x = x + self.patch_pos_emb
        for dit in self.dits:
            x = dit(x, cond)
        x = self.ln(x)
        x = self.linear(x)

        # Reshape
        x = x.view(x.size(0), self.patch_count, self.patch_count, self.channel, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 2, 4, 5)
        x = x.permute(0, 1, 2, 4, 3, 5)
        x = x.reshape(x.size(0), self.channel, self.patch_count * self.patch_size, self.patch_count * self.patch_size)
        return x
