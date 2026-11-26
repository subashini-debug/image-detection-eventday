"""Hybrid Spatial-Residual + Frequency-Spectrum + Forensic MLP PyTorch model."""

from __future__ import annotations

import torch
import torch.nn as nn

from ai_image_detector.config import ModelConfig


def _norm2d(channels: int) -> nn.GroupNorm:
    """Create a GroupNorm layer independent of mini-batch size."""
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(num_groups=groups, num_channels=channels)
    return nn.GroupNorm(num_groups=1, num_channels=channels)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention block."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(4, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        weights = self.pool(x).view(b, c)
        scale = self.fc(weights).view(b, c, 1, 1)
        return x * scale


class ResidualConvBlock(nn.Module):
    """Residual 2D convolution block with BatchNorm, GELU, and SE attention."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = _norm2d(out_channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = _norm2d(out_channels)
        self.se = SEBlock(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                _norm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.se(self.bn2(self.conv2(out)))
        out = out + self.shortcut(x)
        return self.act(out)


class SpatialResidualEncoder(nn.Module):
    """Encodes 6-channel RGB + SRM high-pass residual tensor."""

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 3
        c4 = base_channels * 4

        self.stem = nn.Sequential(
            nn.Conv2d(6, c1, kernel_size=3, stride=1, padding=1, bias=False),
            _norm2d(c1),
            nn.GELU(),
        )
        self.stage1 = ResidualConvBlock(c1, c1, stride=1)
        self.stage2 = ResidualConvBlock(c1, c2, stride=2)
        self.stage3 = ResidualConvBlock(c2, c3, stride=2)
        self.stage4 = ResidualConvBlock(c3, c4, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.out_dim = c4 * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        avg_feat = self.avg_pool(x).flatten(1)
        max_feat = self.max_pool(x).flatten(1)
        return torch.cat([avg_feat, max_feat], dim=1)


class FrequencyEncoder(nn.Module):
    """Encodes 1-channel centered 2D FFT log-magnitude spectrum."""

    def __init__(self, base_channels: int = 24) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 3

        self.net = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1, bias=False),
            _norm2d(c1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
            _norm2d(c2),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1, bias=False),
            _norm2d(c3),
            nn.GELU(),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.out_dim = c3 * 2

    def forward(self, fft_map: torch.Tensor) -> torch.Tensor:
        x = self.net(fft_map)
        return torch.cat([self.avg_pool(x).flatten(1), self.max_pool(x).flatten(1)], dim=1)


class ForensicMLP(nn.Module):
    """Encodes standardized 32-dim handcrafted forensic feature vector."""

    def __init__(self, in_dim: int = 32, hidden_dim: int = 64, dropout: float = 0.25) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.out_dim = hidden_dim

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.net(feats)


class HybridAIDetector(nn.Module):
    """Multi-branch model fusing spatial/SRM, FFT spectrum, and forensic features.

    Output convention:
        * ``label = 0`` -> ``real``
        * ``label = 1`` -> ``ai_generated``
        * ``forward(...)`` returns raw logits of shape ``(B,)``.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()

        self.spatial_encoder = SpatialResidualEncoder(base_channels=self.config.base_channels)
        self.freq_encoder = FrequencyEncoder(base_channels=self.config.freq_channels)
        self.forensic_encoder = ForensicMLP(
            in_dim=self.config.forensic_dim,
            hidden_dim=self.config.forensic_hidden_dim,
            dropout=self.config.dropout,
        )

        fused_dim = (
            self.spatial_encoder.out_dim
            + self.freq_encoder.out_dim
            + self.forensic_encoder.out_dim
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(self.config.dropout),
            nn.Linear(fused_dim, self.config.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout * 0.5),
            nn.Linear(self.config.head_hidden_dim, 1),
        )
        # Learnable / calibrated log-temperature for post-hoc logit calibration.
        self.register_buffer("temperature", torch.tensor(1.0, dtype=torch.float32))

    def forward_features(
        self,
        spatial_input: torch.Tensor,
        fft_input: torch.Tensor,
        forensic_input: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        spatial_emb = self.spatial_encoder(spatial_input)
        freq_emb = self.freq_encoder(fft_input)
        forensic_emb = self.forensic_encoder(forensic_input)
        fused = torch.cat([spatial_emb, freq_emb, forensic_emb], dim=1)
        logit = self.classifier(fused).squeeze(-1)
        return {
            "logit": logit,
            "spatial_emb": spatial_emb,
            "freq_emb": freq_emb,
            "forensic_emb": forensic_emb,
        }

    def forward(
        self,
        spatial_input: torch.Tensor,
        fft_input: torch.Tensor,
        forensic_input: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_features(spatial_input, fft_input, forensic_input)["logit"]

    def predict_proba(
        self,
        spatial_input: torch.Tensor,
        fft_input: torch.Tensor,
        forensic_input: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.forward(spatial_input, fft_input, forensic_input)
        temp = torch.clamp(self.temperature, min=0.1, max=10.0)
        return torch.sigmoid(logits / temp)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
