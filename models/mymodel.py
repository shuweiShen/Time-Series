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


def windows():
    return [2, 7, 14, 28]

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
                 use_softmax=False):
        super(MultiScaleConv, self).__init__()

        # 参数校验
        if not isinstance(kernel_sizes, list) or len(kernel_sizes) < 1:
            raise ValueError("kernel_sizes必须是至少包含一个元素的列表")

        # 设置默认分组数（通道独立）
        self.groups =  in_channels

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


class LearnablePerChannelProcessor(nn.Module):
    def __init__(self, in_channels, in_len, num_scales=5, agg_mode="mean",
                 conv_kernel_sizes=[(3, 3), (5, 5),(7, 7)]):
        super().__init__()
        self.windows = [3, 5, 12, 24]
        self.agg_mode = agg_mode
        self.num_scales = num_scales  # 尺度数S
        self.in_channels = in_channels
        self.in_len = in_len

        self.multi_scale_conv = MultiScaleConv(
            in_channels=in_channels,
            kernel_sizes=conv_kernel_sizes,
            stride=(1, 1),
            padding='same'
        )

        # 核心修改：注意力模块的平均权重初始化
        if self.agg_mode == "attention":
            self.attn = nn.Sequential(
                nn.Linear(self.in_channels, self.in_channels),
                nn.GELU(),
                # 最后一层全连接层：初始化输出为等权重的logits
                nn.Linear(self.in_channels, self.num_scales)
            )

            # --------------------------
            # 关键：初始化最后一层全连接层参数
            # 目标：让输入任意通道特征时，输出的logits经softmax后为平均分布
            # --------------------------
            last_fc = self.attn[-1]  # 获取最后一层全连接层
            # 1. 偏置项初始化为：让logits输出为相同值（如0）
            nn.init.constant_(last_fc.bias, 0.0)
            # 2. 权重项初始化为：让输入特征经过线性变换后，输出仍为相同值（抵消输入差异）
            # 策略：将权重矩阵所有元素初始化为极小值（如1e-6），确保输入×权重≈0
            nn.init.constant_(last_fc.weight, 1e-6)

        # 残差权重（保留原逻辑）
        self.residual_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        B, C, L = x.shape
        assert L == self.in_len, f"输入时序长度{L}与初始化in_len{self.in_len}不匹配"

        # Step 1: 多尺度均值特征生成（B,C,S,L）
        multi_scale_x = multi_scale_mean(x, self.windows)
        assert multi_scale_x.shape[2] == self.num_scales, \
            f"多尺度均值的尺度数{multi_scale_x.shape[2]}与设定num_scales{self.num_scales}不匹配"

        # Step 2: 多尺度卷积处理（B,C,S,L）
        conv_output = self.multi_scale_conv(multi_scale_x)

        # Step 3: 多尺度聚合（注意力模式保留softmax，确保初始权重平均）
        if self.agg_mode == "mean":
            x_agg = reduce(conv_output, "b c s l -> b c l", "mean")
        elif self.agg_mode == "max":
            x_agg = reduce(conv_output, "b c s l -> b c l", "max")
        elif self.agg_mode == "attention":
            # 1. 提取全局特征（压缩时序和尺度维度）
            global_feat = reduce(conv_output, "b c s l -> b c s", "mean")  # B,C,S
            channel_feat = reduce(global_feat, "b c s -> b c", "mean")    # B,C

            # 2. 生成注意力权重（初始经softmax后为平均分布）
            attn_logits = self.attn(channel_feat)  # B,C,S（未归一化的logits）
            attn_weights = F.softmax(attn_logits, dim=-1)  # B,C,S（归一化后权重和为1）

            # 3. 加权求和
            x_agg = reduce(
                conv_output * attn_weights.unsqueeze(-1),
                "b c s l -> b c l",
                "sum"
            )

        # Step 4: 残差连接
        output = x_agg * self.residual_weight + (1 - self.residual_weight) * x
        return output


# 示例：测试完善后的代码
if __name__ == "__main__":
    # 1. 构造输入数据（B=2批量，C=3通道，L=100时序长度）
    x = torch.randn(2, 3, 100)

    # 2. 初始化处理器（选择不同聚合方式）
    processor_mean = LearnablePerChannelProcessor(
        in_channels=3, in_len=100,activ="gelu", drop=0.1,
        min_size=3, max_size=30, num_scales=3, hidden_dim=64, agg_mode="mean"
    )
    processor_attention = LearnablePerChannelProcessor(
        in_channels=3,  in_len=100,activ="gelu", drop=0.1,
        min_size=3, max_size=30, num_scales=6, hidden_dim=64, agg_mode="attention"
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
