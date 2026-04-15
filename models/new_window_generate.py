import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
import torch.distributions as dist


def get_activation(activ: str):
    if activ == "gelu":
        return nn.GELU()
    elif activ == "sigmoid":
        return nn.Sigmoid()
    elif activ == "tanh":
        return nn.Tanh()
    elif activ == "relu":
        return nn.ReLU()
    raise RuntimeError("activation should not be {}".format(activ))


class LearnableWindowGenerator(nn.Module):
    """可学习的窗口大小生成器（保持原逻辑）"""

    def __init__(self, min_size=2, max_size=20, num_scales=5, hidden_dim=64, temperature=0.1):
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.num_scales = num_scales
        self.hidden_dim = hidden_dim
        self.temperature = temperature

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        self.window_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, num_scales * (max_size - min_size + 1))
        )

        self.bias = nn.Parameter(torch.zeros(num_scales, max_size - min_size + 1))

    def forward(self, x):
        B, C, L = x.shape
        x_feat = x[:, 0:1, :] if C > 1 else x
        features = self.feature_extractor(x_feat)

        logits = self.window_predictor(features)
        logits = logits.view(B, self.num_scales, -1)
        logits = logits + self.bias.unsqueeze(0)
        logits = logits / self.temperature

        if self.training:
            window_dist = dist.RelaxedOneHotCategorical(temperature=self.temperature, logits=logits)
            window_samples = window_dist.rsample()
            window_indices = torch.argmax(window_samples, dim=-1)
        else:
            window_probs = F.softmax(logits, dim=-1)
            window_indices = torch.argmax(window_probs, dim=-1)

        window_sizes = window_indices + self.min_size
        window_sizes = torch.clamp(window_sizes, min=self.min_size, max=min(self.max_size, L))

        all_windows = window_sizes.view(-1)
        unique_windows, counts = torch.unique(all_windows, return_counts=True)
        if len(unique_windows) <= self.num_scales:
            selected_windows = unique_windows.tolist()
        else:
            sorted_indices = torch.argsort(counts, descending=True)
            selected_windows = unique_windows[sorted_indices[:self.num_scales]].tolist()
        selected_windows = sorted(list(set(selected_windows))) if selected_windows else [min(3, L)]

        return selected_windows


def learnable_multi_scale_mean(x, window_generator):
    """生成每个通道的多尺度均值（保持原逻辑）"""
    B, C, L = x.shape
    sizes = window_generator(x)
    num_sizes = len(sizes)
    output = torch.zeros(B, C, num_sizes, L, device=x.device, dtype=x.dtype)
    mask = torch.ones_like(x)

    for i, k in enumerate(sizes):
        if k <= 0:
            raise ValueError("窗口大小必须为正整数")

        left_pad = (k - 1) // 2
        right_pad = k - 1 - left_pad

        weight = torch.ones(C, 1, k, device=x.device, dtype=x.dtype)
        conv_x = F.conv1d(
            F.pad(x, (left_pad, right_pad), mode='constant', value=0),
            weight,
            padding=0,
            groups=C
        )
        conv_mask = F.conv1d(
            F.pad(mask, (left_pad, right_pad), mode='constant', value=0),
            weight,
            padding=0,
            groups=C
        )

        mean = conv_x / torch.clamp(conv_mask, min=1e-8)
        output[:, :, i, :] = mean

    return output


class MultiScaleConvFusion(nn.Module):
    """多尺度卷积融合模块：修复1D卷积输入维度，批量处理所有通道"""

    def __init__(self, in_dim, num_scales, kernel_sizes=None):
        super().__init__()
        self.num_scales = num_scales

        # 为每个尺度定义1D卷积核（默认不同尺度用不同核大小）
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7, 9, 11][:num_scales]
        assert len(kernel_sizes) == num_scales, "卷积核数量必须与尺度数一致"

        # 尺度内卷积：每个尺度独立卷积，保持时序长度L
        self.scale_convs = nn.ModuleList()
        for k in kernel_sizes:
            pad = (k - 1) // 2  # 保证卷积后L不变
            self.scale_convs.append(
                nn.Conv1d(in_dim, in_dim, kernel_size=k, padding=pad, groups=in_dim)
            )

        # 跨尺度融合：将S个尺度的特征压缩为1个
        self.cross_scale_conv = nn.Conv1d(in_dim * num_scales, in_dim, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, x):
        """
        输入: x [B×C×S, in_dim, L] （展平尺度为批量维度，适配1D卷积）
        输出: 融合后特征 [B×C, in_dim, L]
        """
        BxCxS, in_dim, L = x.shape
        BxC = BxCxS // self.num_scales  # 恢复B×C（批量×通道）

        # 每个尺度单独卷积：按尺度分组处理
        scale_features = []
        for s in range(self.num_scales):
            # 提取第s个尺度的所有通道：[B×C, in_dim, L]
            scale_feat = x[s * BxC: (s + 1) * BxC, :, :]
            conv_feat = self.scale_convs[s](scale_feat)  # 尺度内卷积
            scale_features.append(conv_feat)

        # 拼接所有尺度：[B×C, in_dim×S, L]
        combined = torch.cat(scale_features, dim=1)
        # 跨尺度融合：[B×C, in_dim, L]
        fused = self.cross_scale_conv(combined)
        return self.activation(fused)


class LearnablePerChannelProcessor(nn.Module):
    """完整修复：1D卷积输入维度 + 多尺度融合逻辑"""

    def __init__(self, in_len, hid_dim, activ, drop=0.0, num_layers=3,
                 min_size=1, max_size=20, num_scales=5, hidden_dim=64, conv_kernel_sizes=None):
        super().__init__()
        self.window_generator = LearnableWindowGenerator(
            min_size=min_size, max_size=max_size, num_scales=num_scales, hidden_dim=hidden_dim
        )
        self.num_scales = num_scales  # 设定的最大尺度数
        self.hid_dim = hid_dim  # 多尺度卷积的特征维度
        self.in_len = in_len  # 原始时序长度L

        # 1. 通道映射：单通道→高维通道（适配多尺度卷积）
        # 输入：[B×C×S, 1, L]（单通道），输出：[B×C×S, hid_dim, L]
        self.channel_proj = nn.Conv1d(1, hid_dim, kernel_size=1)

        # 2. 多尺度卷积融合模块
        self.conv_fusion = MultiScaleConvFusion(
            in_dim=hid_dim,
            num_scales=num_scales,
            kernel_sizes=conv_kernel_sizes
        )

        # 3. 特征增强网络：输入维度=hid_dim×L（融合后的特征展平）
        layers = []
        current_dim = hid_dim * in_len
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hid_dim),
                get_activation(activ),
                nn.Dropout(drop),
                nn.BatchNorm1d(hid_dim)
            ])
            current_dim = hid_dim
        layers.append(nn.Linear(hid_dim, in_len))  # 输出恢复为原始时序长度
        self.net = nn.Sequential(*layers)

        self.residual = nn.Identity()

    def forward(self, x):
        """输入: [B, C, L] → 输出: [B, C, L]"""
        B, C, L = x.shape

        # Step 1：生成多尺度均值特征 → [B, C, S, L]（S=实际尺度数）
        multi_scale_x = learnable_multi_scale_mean(x, self.window_generator)
        S = multi_scale_x.shape[2]
        assert S == self.num_scales, f"实际尺度数{S}需与设定值{self.num_scales}一致"

        # Step 2：适配1D卷积输入（消除4D结构）
        # 重塑：[B, C, S, L] → [B×C×S, 1, L]（展平B/C/S为批量维度，单通道）
        x_reshaped = rearrange(multi_scale_x, "b c s l -> (b c s) 1 l")
        # 通道映射：[B×C×S, 1, L] → [B×C×S, hid_dim, L]（单通道→高维通道）
        x_proj = self.channel_proj(x_reshaped)

        # Step 3：多尺度卷积融合 → [B×C, hid_dim, L]
        x_fused = self.conv_fusion(x_proj)

        # Step 4：特征增强（维度匹配）
        # 展平：[B×C, hid_dim, L] → [B×C, hid_dim×L]（适配全连接层）
        x_flat = rearrange(x_fused, "bxc d l -> bxc (d l)")
        # 残差增强：[B×C, hid_dim×L] → [B×C, L]
        processed = self.net(x_flat)
        residual = self.residual(x_flat[:, :self.in_len])  # 残差取前L个维度（与输出对齐）
        x_enhanced = (processed + residual) / 2

        # Step 5：恢复原始形状并与输入融合
        output = rearrange(x_enhanced, "(b c) l -> b c l", b=B, c=C)
        output = (output + x) / 2  # 输入-输出残差，保留原始特征

        return output


# 测试代码（确保无维度错误）
if __name__ == "__main__":
    # 输入配置：B=2（批量）, C=3（通道）, L=100（时序长度）
    x = torch.randn(2, 3, 100)
    print(f"输入形状: {x.shape}")

    # 初始化处理器：设定3个尺度，卷积核大小[3,5,7]（需与num_scales一致）
    processor = LearnablePerChannelProcessor(
        in_len=100,  # 原始时序长度L
        hid_dim=128,  # 多尺度卷积的特征维度
        activ="gelu",
        drop=0.1,
        num_layers=3,
        min_size=2,
        max_size=10,
        num_scales=3,  # 设定尺度数（需与conv_kernel_sizes长度一致）
        hidden_dim=64,
        conv_kernel_sizes=[3, 5, 7]  # 3个尺度对应3个卷积核
    )

    # 前向传播（训练模式）
    processor.train()
    output = processor(x)
    print(f"训练模式输出形状: {output.shape}")  # 预期 [2, 3, 100]

    # 推理模式测试
    processor.eval()
    with torch.no_grad():
        output_eval = processor(x)
        window_sizes = processor.window_generator(x)
    print(f"推理模式输出形状: {output_eval.shape}")  # 预期 [2, 3, 100]
    print(f"动态生成的窗口大小: {window_sizes}")  # 例如 [2,5,8]（需为3个尺度）