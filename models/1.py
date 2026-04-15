import numpy as np
import matplotlib.pyplot as plt


# -------------------------- 1. 生成模拟时序数据（模拟x_enc的单样本数据）
def generate_simulated_ts_data(seq_len=24, n_features=3, seed=42):
    """生成带趋势和噪声的多特征时序数据，模拟原始输入x_enc的单样本"""
    np.random.seed(seed)
    time_steps = np.arange(seq_len)  # 时间步（如24小时）

    # 生成3个不同尺度的特征（模拟温度、湿度、气压等）
    feature1 = 20 + 5 * np.sin(time_steps / 4) + np.random.randn(seq_len)  # 均值~20，波动±5
    feature2 = 60 + 10 * np.cos(time_steps / 6) + 2 * np.random.randn(seq_len)  # 均值~60，波动±10
    feature3 = 1013 + 3 * np.random.randn(seq_len)  # 均值~1013，波动±3

    # 组合成数据矩阵：shape [seq_len, n_features]（对应单样本x_enc的[L, N]）
    ts_data = np.column_stack([feature1, feature2, feature3])
    return ts_data, time_steps


# -------------------------- 2. 执行“减均值、除方差”标准化（匹配模型逻辑）
def standardize_data(data):
    """
    对时序数据执行标准化：data = (data - 均值) / 标准差
    data: 输入数据，shape [seq_len, n_features]
    返回：标准化后数据、均值、标准差
    """
    # 计算每个特征的均值（按特征维度，即axis=0），keepdims保持维度一致
    mean = np.mean(data, axis=0, keepdims=True)
    # 计算每个特征的标准差（添加1e-5避免除零）
    std = np.sqrt(np.var(data, axis=0, keepdims=True) + 1e-5)
    # 标准化
    standardized_data = (data - mean) / std
    return standardized_data, mean, std


# -------------------------- 3. 可视化标准化前后的变化
def plot_standardization_comparison(raw_data, standardized_data, time_steps):
    """对比可视化原始数据与标准化后数据"""
    n_features = raw_data.shape[1]
    feature_names = [f"特征{i + 1}" for i in range(n_features)]  # 特征名称

    # 创建2行1列的子图（上下对比）
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("时序数据标准化前后对比（减均值 + 除方差）", fontsize=14, fontweight='bold')

    # 绘制原始数据
    for i in range(n_features):
        ax1.plot(time_steps, raw_data[:, i], marker='o', linewidth=2, label=feature_names[i])
    ax1.set_title("标准化前：原始数据（不同特征尺度差异大）", fontsize=12)
    ax1.set_ylabel("原始数值", fontsize=10)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 绘制标准化后数据
    for i in range(n_features):
        ax2.plot(time_steps, standardized_data[:, i], marker='s', linewidth=2, label=feature_names[i])
    ax2.set_title("标准化后：(数据 - 均值) / 标准差（特征尺度统一）", fontsize=12)
    ax2.set_xlabel("时间步", fontsize=10)
    ax2.set_ylabel("标准化后数值", fontsize=10)
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='均值线（0）')  # 标注0均值线
    ax2.legend()

    # 调整布局并保存
    plt.tight_layout()
    plt.savefig("ts_standardization_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()


# -------------------------- 4. 主函数：串联流程
if __name__ == "__main__":
    # 步骤1：生成模拟时序数据
    seq_len = 24  # 时序长度（如24小时）
    raw_data, time_steps = generate_simulated_ts_data(seq_len=seq_len)

    # 步骤2：执行标准化
    standardized_data, mean, std = standardize_data(raw_data)

    # 打印关键统计信息（验证标准化效果）
    print("=" * 50)
    print("标准化前后统计信息对比")
    print("=" * 50)
    for i in range(raw_data.shape[1]):

        print(f"  原始数据 - 均值：{mean[0, i]:.2f}，标准差：{std[0, i]:.2f}")
        print(
            f"  标准化后 - 均值：{np.mean(standardized_data[:, i]):.6f}（≈0），标准差：{np.std(standardized_data[:, i]):.6f}（≈1）")

    # 步骤3：可视化对比
    plot_standardization_comparison(raw_data, standardized_data, time_steps)