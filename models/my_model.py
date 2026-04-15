import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
import torch.distributions as dist
import math


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


# def dynamic_windows(seq_len, min_size=3, max_size=None, num_scales=4):
#     return [3,5,7,13]
def dynamic_windows(seq_len, min_size=3, max_size=None, num_scales=5):
    """简化的动态窗口生成，确保不同长度有不同的窗口"""
    # 基础窗口大小
    base_windows = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]

    # 根据序列长度选择不同的窗口范围

    if seq_len <= 192:
        window_pool = base_windows[:8]
    elif seq_len <= 336:
        window_pool = base_windows[4:12]
    else:  # 720 或更长
        window_pool = base_windows[6:]

    # 从池中等间隔选择
    step = max(1, len(window_pool) // num_scales)
    selected = [window_pool[i * step] for i in range(num_scales)]

    # 确保数量正确
    while len(selected) < num_scales and len(selected) < len(window_pool):
        # 添加尚未选择的窗口
        for window in window_pool:
            if window not in selected:
                selected.append(window)
                if len(selected) >= num_scales:
                    break

    return sorted(selected[:num_scales])

def multi_scale_mean(x, windows):
    """可学习的多尺度窗口均值计算：输入[B, C, L]，输出多尺度增强特征[B, C, S, L]（S为尺度数）"""
    B, C, L = x.shape
    sizes = windows  # 从生成器获取窗口大小列表
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

class MultiScaleConv(nn.Module):
    """
    多尺度卷积类，对每个通道的[S,L]二维数据应用不同尺度的卷积核，
    并通过可学习的权重对不同分支结果进行加权求和

    参数:
        in_channels (int): 输入通道数C
        kernel_sizes (list): 不同卷积核尺寸的列表，如[(3,3), (5,5), (7,7)]
        stride (tuple, optional): 卷积步长，默认为(1,1)
        padding (str or tuple, optional): 填充方式，默认为'same'
        groups (int, optional): 分组卷积数，默认为in_channels（通道独立）
        use_softmax (bool, optional): 是否对权重应用softmax归一化，默认为True
    """

    def __init__(self, in_channels, kernel_sizes, stride=(1, 1), padding='same',
                 groups=None, use_softmax=True):
        super(MultiScaleConv, self).__init__()

        # 参数校验
        if not isinstance(kernel_sizes, list) or len(kernel_sizes) < 1:
            raise ValueError("kernel_sizes必须是至少包含一个元素的列表")

        # 设置默认分组数（通道独立）
        self.groups = groups if groups is not None else in_channels

        # 创建不同尺度的卷积分支
        self.num_scales = len(kernel_sizes)
        self.conv_branches = nn.ModuleList()

        for kernel_size in kernel_sizes:
            # 计算'same' padding的具体值
            if padding == 'same':
                pad_h = (kernel_size[0] - 1) // 2
                pad_w = (kernel_size[1] - 1) // 2
                pad = (pad_w, pad_h)
            else:
                pad = padding
            # 添加卷积层（保持输入输出通道数一致）
            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=pad,
                groups=self.groups,
                bias=False
            )
            self.conv_branches.append(conv)

        # 创建可学习的融合权重
        self.scale_weights = nn.Parameter(torch.ones(self.num_scales))
        self.use_softmax = use_softmax

    def forward(self, x):
        """
        前向传播

        参数:
            x (torch.Tensor): 输入张量，形状为(B, C, S, L)

        返回:
            torch.Tensor: 多尺度融合后的张量，形状为(B, C, S_out, L_out)
        """
        # 存储各分支的卷积结果
        branch_outputs = []

        # 每个尺度的卷积计算
        for conv in self.conv_branches:
            branch_outputs.append(conv(x))

        # 计算权重（可选softmax归一化）
        if self.use_softmax:
            weights = F.softmax(self.scale_weights, dim=0)
        else:
            weights = self.scale_weights

        # 加权求和融合不同尺度的结果
        result = 0.0
        for i in range(self.num_scales):
            result += weights[i] * branch_outputs[i]

        return result


class EnhancedMultiScaleConv(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride=(1, 1), padding='same',
                 groups=1, use_softmax=True, activation='relu'):
        super(EnhancedMultiScaleConv, self).__init__()

        self.num_scales = len(kernel_sizes)
        self.conv_branches = nn.ModuleList()

        for kernel_size in kernel_sizes:
            if padding == 'same':
                pad_h = (kernel_size[0] - 1) // 2
                pad_w = (kernel_size[1] - 1) // 2
                pad = (pad_w, pad_h)
            else:
                pad = padding

            # 增强的卷积块：卷积 + 归一化 + 激活
            conv_block = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=pad,
                    groups=groups,
                    bias=False
                ),
                nn.BatchNorm2d(in_channels),
                get_activation(activation),
                nn.Dropout(0.1)  # 添加dropout防止过拟合
            )
            self.conv_branches.append(conv_block)

        self.scale_weights = nn.Parameter(torch.ones(self.num_scales))
        self.use_softmax = use_softmax

        # 添加通道注意力机制
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1),
            get_activation(activation),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        branch_outputs = []

        for conv in self.conv_branches:
            branch_out = conv(x)
            # 应用通道注意力
            attention_weights = self.channel_attention(branch_out)
            branch_out = branch_out * attention_weights
            branch_outputs.append(branch_out)

        if self.use_softmax:
            weights = F.softmax(self.scale_weights, dim=0)
        else:
            weights = self.scale_weights

        result = 0.0
        for i in range(self.num_scales):
            result += weights[i] * branch_outputs[i]

        return result

class LearnablePerChannelProcessor(nn.Module):
    def __init__(self, in_channels, in_len, pre_len, num_scales=4, agg_mode="mean",
                 conv_kernel_sizes=[(3, 3), (5, 5), (7, 7)]):
        super().__init__()
        self.windows = dynamic_windows(pre_len)


        # 2. 多尺度卷积（已修正in_channels）
        self.multi_scale_conv = MultiScaleConv(
            in_channels=in_channels,
            kernel_sizes=conv_kernel_sizes,
            stride=(1, 1),
            padding='same'
        )

        # 3. 多尺度聚合（核心修正：注意力模块维度）
        self.agg_mode = agg_mode
        self.num_scales = num_scales  # 记录尺度数，用于注意力维度
        self.in_channels = in_channels  # 记录通道数
        self.in_len = in_len  # 记录时序长度L
        self.residual_weight = nn.Parameter(torch.tensor(0.5))



    def forward(self, x):
        B, C, L = x.shape  # B=2, C=3, L=100（示例输入）
        # 验证输入维度与初始化的in_len一致（避免时序长度不匹配）
        assert L == self.in_len, f"输入时序长度{L}与初始化in_len{self.in_len}不匹配"

        # Step 1: 多尺度特征生成（输出：[B,C,S,L] = [2,3,S,100]）
        multi_scale_x = multi_scale_mean(x, self.windows)

        # Step 2: 多尺度卷积处理（输出：[B,C,S,L] = [2,3,6,100]）
        conv_output = self.multi_scale_conv(multi_scale_x)

        # Step 3: 多尺度聚合（核心修正：注意力维度重排）
        if self.agg_mode == "mean":
            x_agg = reduce(conv_output, "b c s l -> b c l", "mean")
        elif self.agg_mode == "max":
            x_agg = reduce(conv_output, "b c s l -> b c l", "max")






        output = x_agg * self.residual_weight  + x*(1-self.residual_weight)
        return output


# 示例：测试完善后的代码
if __name__ == "__main__":
    lst = [96,192,336,720]
    for i in lst:
        print(dynamic_windows(i))