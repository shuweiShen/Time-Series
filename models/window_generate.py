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
    """可学习的窗口大小生成器：根据输入数据自动生成合适的窗口大小"""

    def __init__(self, min_size=2, max_size=20, num_scales=5, hidden_dim=64, temperature=0.1):
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.num_scales = num_scales
        self.hidden_dim = hidden_dim
        self.temperature = temperature

        # 特征提取网络：从单通道时序数据中提取全局特征
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1),  # 捕捉局部时序模式
            nn.ReLU(),  # 引入非线性
            nn.AdaptiveAvgPool1d(1),  # 全局平均池化：无论L多长，输出长度为1
            nn.Flatten()  # 展平为[B, hidden_dim]
        )

        # 窗口大小预测网络
        self.window_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),  # 维度扩张
            nn.ReLU(),
            nn.Dropout(0.1),  # 正则化，防止过拟合
            nn.Linear(hidden_dim * 2, num_scales * (max_size - min_size + 1))  # 输出每个尺度的窗口候选分布
        )

        # 可学习偏置
        self.bias = nn.Parameter(torch.zeros(num_scales, max_size - min_size + 1))

    def forward(self, x):
        """输入: x [B, C, L]，输出: 窗口大小列表 [s1, s2, ..., sS]（S≤num_scales）"""
        B, C, L = x.shape

        # 仅用第一个通道提取特征（窗口大小是全局适配，无需多通道冗余）
        x_feat = x[:, 0:1, :] if C > 1 else x  # [B, 1, L]

        # 提取全局特征：[B, 1, L] → [B, hidden_dim]
        features = self.feature_extractor(x_feat)

        # 预测窗口分布：[B, hidden_dim] → [B, num_scales, M]（M=max_size-min_size+1）
        logits = self.window_predictor(features)
        logits = logits.view(B, self.num_scales, -1)  # 重塑为多尺度分布格式
        logits = logits + self.bias.unsqueeze(0)  # 加可学习偏置
        logits = logits / self.temperature  # 温度调节：控制分布尖锐度

        # 训练/推理差异化窗口选择
        if self.training:
            # 训练用Gumbel-Softmax：保证可微性，支持端到端优化
            window_dist = dist.RelaxedOneHotCategorical(temperature=self.temperature, logits=logits)
            window_samples = window_dist.rsample()  # [B, num_scales, M]：软化的独热采样
            window_indices = torch.argmax(window_samples, dim=-1)  # [B, num_scales]：取概率最高的索引
        else:
            # 推理用硬选择：直接取概率最大的窗口
            window_probs = F.softmax(logits, dim=-1)
            window_indices = torch.argmax(window_probs, dim=-1)  # [B, num_scales]

        # 索引转实际窗口大小 + 范围裁剪（避免无效窗口）
        window_sizes = window_indices + self.min_size  # [B, num_scales]：索引→实际大小
        window_sizes = torch.clamp(window_sizes, min=self.min_size, max=min(self.max_size, L))  # 限制在有效范围

        # 跨批量筛选：选出现频次最高的窗口（保证全局适配）
        all_windows = window_sizes.view(-1)  # 展平所有窗口：[B×num_scales]
        unique_windows, counts = torch.unique(all_windows, return_counts=True)  # 统计频次
        # 选Top-S窗口（S≤num_scales）
        if len(unique_windows) <= self.num_scales:
            selected_windows = unique_windows.tolist()
        else:
            sorted_indices = torch.argsort(counts, descending=True)  # 按频次降序
            selected_windows = unique_windows[sorted_indices[:self.num_scales]].tolist()
        # 兜底逻辑 + 去重排序
        selected_windows = sorted(list(set(selected_windows))) if selected_windows else [min(3, L)]

        return selected_windows


def learnable_multi_scale_mean(x, window_generator):
    """可学习的多尺度窗口均值计算：输入[B, C, L]，输出多尺度增强特征[B, C, S, L]（S为尺度数）"""
    B, C, L = x.shape
    sizes = window_generator(x)  # 从生成器获取窗口大小列表
    num_sizes = len(sizes)
    output = torch.zeros(B, C, num_sizes, L, device=x.device, dtype=x.dtype)  # 初始化输出
    mask = torch.ones_like(x)  # 掩码：解决边界填充的均值偏差问题

    for i, k in enumerate(sizes):
        if k <= 0:
            raise ValueError("窗口大小必须为正整数")

        # 1. 非对称填充：保证卷积后时序长度与输入一致（避免L缩短）
        left_pad = (k - 1) // 2  # 左填充长度
        right_pad = k - 1 - left_pad  # 右填充长度（总填充=k-1，确保输出长度=L）

        # 2. 分组卷积求和：每个通道独立计算局部窗口和（避免通道干扰）
        weight = torch.ones(C, 1, k, device=x.device, dtype=x.dtype)  # 卷积核：全1（求和用）
        # 输入数据卷积求和
        conv_x = F.conv1d(
            F.pad(x, (left_pad, right_pad), mode='constant', value=0),  # 填充后x：[B, C, L + k-1]
            weight,
            padding=0,
            groups=C  # 分组卷积：每个通道用独立卷积核
        )  # 输出conv_x：[B, C, L]（局部窗口和）
        # 掩码卷积求和（统计窗口内有效元素数，排除填充的0）
        conv_mask = F.conv1d(
            F.pad(mask, (left_pad, right_pad), mode='constant', value=0),
            weight,
            padding=0,
            groups=C
        )  # 输出conv_mask：[B, C, L]（有效元素数）

        # 3. 计算局部均值：避免除零（用clamp限制最小为1e-8）
        mean = conv_x / torch.clamp(conv_mask, min=1e-8)

        # 4. 存入当前尺度结果
        output[:, :, i, :] = mean

    return output


class LearnablePerChannelProcessor(nn.Module):
    """可学习多尺度通道处理器：自动生成窗口→多尺度增强→恢复[B, C, L]形状"""

    def __init__(self, in_len, hid_dim, activ, drop=0.0, num_layers=3,
                 min_size=1, max_size=20, num_scales=5, hidden_dim=64, agg_mode="mean"):
        super().__init__()
        # 1. 初始化窗口生成器
        self.window_generator = LearnableWindowGenerator(
            min_size=min_size, max_size=max_size, num_scales=num_scales, hidden_dim=hidden_dim
        )

        # 2. 多尺度特征聚合方式（可选：均值、最大池化、注意力加权）
        assert agg_mode in ["mean", "max", "attention"], "agg_mode必须是mean/max/attention"
        self.agg_mode = agg_mode
        # 注意力聚合：可学习每个尺度的权重（更灵活的信息融合）
        if self.agg_mode == "attention":
            self.scale_attention = nn.Sequential(
                nn.Linear(in_len, in_len),  # 时序维度注意力
                nn.Softmax(dim=2)  # 对尺度维度归一化权重
                #可以改1
            )

        # 3. 通道级特征增强网络（全连接+残差：细化多尺度聚合后的特征）
        layers = []
        current_dim = in_len
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hid_dim),  # 维度扩张：捕捉高阶特征
                get_activation(activ),  # 非线性激活
                nn.Dropout(drop),  # 正则化
                nn.BatchNorm1d(hid_dim)  # 批量归一化：加速收敛
            ])
            current_dim = hid_dim
        layers.append(nn.Linear(hid_dim, in_len))  # 维度恢复：输出与输入长度一致
        self.net = nn.Sequential(*layers)

        # 4. 残差连接：保留原始多尺度信息，避免特征丢失
        self.residual = nn.Identity()

    def forward(self, x):
        """输入: x [B, C, L]，输出: 增强后特征 [B, C, L]"""
        B, C, L = x.shape

        # Step 1：生成多尺度增强特征 [B, C, S, L]
        multi_scale_x = learnable_multi_scale_mean(x, self.window_generator)  # S为实际尺度数

        # Step 2：多尺度特征聚合 → 从[B, C, S, L]恢复为[B, C, L]
        if self.agg_mode == "mean":
            # 均值聚合：平衡各尺度信息（计算简单，泛化性强）
            x_agg = reduce(multi_scale_x, "b c s l -> b c l", "mean")
        elif self.agg_mode == "max":
            # 最大池化聚合：突出各尺度中的强响应（适合捕捉显著特征）
            x_agg = reduce(multi_scale_x, "b c s l -> b c l", "max")
        elif self.agg_mode == "attention":
            # 注意力聚合：可学习尺度权重（动态侧重重要尺度）
            # 重塑为[B×C, S, L] → 计算每个时序位置的尺度权重 → 加权求和
            x_reshaped = rearrange(multi_scale_x, "b c s l -> (b c) s l")  # [B×C, S, L]
            attn_weights = self.scale_attention(x_reshaped)  # [B×C, S, L]：每个位置的尺度权重
            x_agg = torch.sum(x_reshaped * attn_weights, dim=1)  # [B×C, L]：加权聚合
            x_agg = rearrange(x_agg, "(b c) l -> b c l", b=B, c=C)  # 恢复[B, C, L]

        # Step 3：通道级特征增强（全连接+残差）
        # 展平：[B, C, L] → [B×C, L]（全连接层要求2D输入）
        x_flat = rearrange(x_agg, "b c l -> (b c) l")
        # 残差加和：增强特征 = 网络输出 + 原始聚合特征（保留基础信息）
        processed = self.net(x_flat)
        residual = self.residual(x_flat)
        x_enhanced = (processed + residual)/2

        # Step 4：重塑回[B, C, L]
        output = rearrange(x_enhanced, "(b c) l -> b c l", b=B, c=C)
        output = (output + x)/2
        return output


# 示例：测试完善后的代码
if __name__ == "__main__":
    # 1. 构造输入数据（B=2批量，C=3通道，L=100时序长度）
    x = torch.randn(2, 3, 100)

    # 2. 初始化处理器（选择不同聚合方式）
    processor_mean = LearnablePerChannelProcessor(
        in_len=100, hid_dim=128, activ="gelu", drop=0.1, num_layers=3,
        min_size=2, max_size=10, num_scales=3, hidden_dim=64, agg_mode="mean"
    )
    processor_attention = LearnablePerChannelProcessor(
        in_len=100, hid_dim=128, activ="gelu", drop=0.1, num_layers=3,
        min_size=2, max_size=10, num_scales=6, hidden_dim=64, agg_mode="attention"
    )

    # 3. 前向传播测试
    processor_mean.train()
    output_mean = processor_mean(x)
    print(f"均值聚合 - 输入形状: {x.shape}, 输出形状: {output_mean.shape}")  # 应均为[2,3,100]

    processor_attention.eval()
    output_attn = processor_attention(x)
    print(f"注意力聚合 - 输入形状: {x.shape}, 输出形状: {output_attn.shape}")  # 应均为[2,3,100]

    # 4. 查看生成的窗口大小
    with torch.no_grad():
        window_sizes = processor_mean.window_generator(x)
        print(f"动态生成的窗口大小: {window_sizes}")
