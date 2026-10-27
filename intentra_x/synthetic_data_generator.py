"""
合成无人机飞行时序数据生成器
基于物理阈值规则打标签，不使用任何加权公式。
"""
import os
import numpy as np


class SyntheticDataGenerator:
    """生成 (N, T, 6) 飞行时序数据 + 0/1 风险标签"""

    # 6 个特征的取值范围 [min, max]
    FEATURE_RANGES = np.array([
        [0.0, 20.0],    # wind_speed       m/s
        [0.0, 25.0],    # drone_speed      m/s
        [-30.0, 30.0],  # roll_angle       deg
        [-45.0, 45.0],  # pitch_angle      deg
        [5.0, 120.0],   # altitude         m
        [0.1, 1.0],     # signal_strength
    ], dtype=np.float32)

    # 各特征随机游走步长（高斯标准差）
    WALK_SIGMA = np.array([1.5, 1.2, 3.0, 3.0, 5.0, 0.08], dtype=np.float32)

    FEATURE_NAMES = ['wind_speed', 'drone_speed', 'roll_angle',
                     'pitch_angle', 'altitude', 'signal_strength']

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    # ─── 单条样本：6 维独立随机游走 ───
    def _gen_one(self, seq_len: int) -> np.ndarray:
        lo, hi = self.FEATURE_RANGES[:, 0], self.FEATURE_RANGES[:, 1]
        x = self.rng.uniform(lo, hi).astype(np.float32)  # 起点随机
        seq = np.zeros((seq_len, 6), dtype=np.float32)
        seq[0] = x
        for t in range(1, seq_len):
            step = self.rng.normal(0.0, self.WALK_SIGMA).astype(np.float32)
            seq[t] = np.clip(seq[t - 1] + step, lo, hi)
        return seq

    # ─── 物理阈值打标签：取最后 3 帧均值判断 ───
    def _label(self, seq: np.ndarray) -> int:
        m = seq[-3:].mean(axis=0)
        wind, speed, _roll, pitch, alt, sig = m
        if wind > 12 and pitch > 25:        # 失速
            return 1
        if speed > 18 and wind > 10:        # 失控
            return 1
        if sig < 0.25:                      # 断联
            return 1
        if alt < 8 and pitch < -30:         # 撞地
            return 1
        return 0

    def generate(self, n: int = 5000, seq_len: int = 20):
        """返回 X (n, T, 6) float32, y (n,) float32"""
        X = np.zeros((n, seq_len, 6), dtype=np.float32)
        y = np.zeros(n, dtype=np.float32)
        for i in range(n):
            X[i] = self._gen_one(seq_len)
            y[i] = self._label(X[i])
        # 最后 5% 标签随机翻转模拟噪声
        flip_n = int(n * 0.05)
        flip_idx = self.rng.choice(n, flip_n, replace=False)
        y[flip_idx] = 1.0 - y[flip_idx]
        return X, y

    def save_to_npy(self, path: str):
        os.makedirs(path, exist_ok=True)
        X, y = self.generate()
        np.save(os.path.join(path, 'X.npy'), X)
        np.save(os.path.join(path, 'y.npy'), y)
        return X, y

    def get_feature_names(self):
        return list(self.FEATURE_NAMES)

    def describe_rules(self):
        return [
            "失速：风速>12m/s 且 仰角>25° → 风险=1",
            "失控：飞行速度>18m/s 且 风速>10m/s → 风险=1",
            "断联：信号强度<0.25 → 风险=1",
            "撞地：高度<8m 且 仰角<-30° → 风险=1",
        ]


if __name__ == "__main__":
    gen = SyntheticDataGenerator(seed=42)
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    X, y = gen.save_to_npy(save_dir)
    print("=" * 56)
    print(f"数据已保存到: {save_dir}")
    print(f"X 形状: {X.shape}, y 形状: {y.shape}")
    print(f"风险占比: 1类 {y.mean()*100:.2f}% / 0类 {(1-y.mean())*100:.2f}%")
    print("特征范围（min ~ max）：")
    for i, name in enumerate(gen.get_feature_names()):
        print(f"  {name:18s} {X[..., i].min():7.2f} ~ {X[..., i].max():7.2f}")
    print("打标签规则：")
    for r in gen.describe_rules():
        print(f"  - {r}")
    print("=" * 56)
