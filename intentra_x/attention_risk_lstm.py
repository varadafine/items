"""
特征级 Attention-LSTM 风险预测模型
一次前向同时输出: 风险分数 + 每个输入特征的重要性权重。

注意力设计说明:
    采用「逐特征 sigmoid 门控」而非沿特征维 softmax。
    softmax 会强制所有特征权重加和=1、互相竞争，在数据有限时极易退化成
    均匀分布（每个特征≈1/n），使注意力对不同输入几乎无区分。
    sigmoid 门控让每个特征独立取值于 [0,1]，彼此不竞争——重要特征可同时置高、
    无关特征压低，从结构上避免均匀塌缩，也让特征重要性排名更具区分度与可解释性。
"""
import torch
import torch.nn as nn


class FeatureAttentionRiskLSTM(nn.Module):
    """Input (B,T,F) -> risk (B,1) and mean feature gates (B,F)."""

    def __init__(self, n_features: int = 6, hidden: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.n_features = n_features
        # ── 逐特征 sigmoid 门控（带 bias，可学习每个特征的基线开合度）──
        # 用一个小型瓶颈 MLP 从当前帧特征生成门控值，比单层线性更能捕捉
        # 「特征间的条件关系」（例如高风速时更多保留 pitch），且瓶颈维度小、
        # 参数量可控，在小数据上不易过拟合。
        gate_hidden = max(n_features, 16)
        self.gate = nn.Sequential(
            nn.Linear(n_features, gate_hidden),
            nn.Tanh(),
            nn.Linear(gate_hidden, n_features),
            nn.Sigmoid(),
        )
        # ── LSTM 编码器 ──
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden,
            num_layers=num_layers, batch_first=True, dropout=dropout,
        )
        # ── 风险回归头：64 → 32 → 1，Sigmoid 压到 [0,1] ──
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        # ── 第1步：逐特征 sigmoid 门控（每个特征独立取 [0,1]，不强制加和=1）──
        gate = self.gate(x)                            # (B,T,F)
        # ── 第2步：按 0~1 门控保留或抑制每项输入 ──
        x_w = x * gate                                 # (B,T,F)
        # ── 第3步：LSTM 编码，取最后一步 ──
        out, _ = self.lstm(x_w)                        # (B,T,64)
        h = out[:, -1, :]                              # (B,64)
        # ── 第4步：回归头输出风险分数 ──
        risk = self.head(h)                            # (B,1)
        # ── 第5步：时间维聚合得到样本级特征重要性 ──
        importance = gate.mean(dim=1)                  # (B,F)
        return risk, importance

    @staticmethod
    def get_feature_ranking(importance: torch.Tensor, names):
        """importance (6,) 或 (B,6) → [(name, weight), ...] 降序"""
        if importance.dim() == 2:
            importance = importance.mean(dim=0)
        w = importance.detach().cpu().numpy()
        pairs = list(zip(names, w.tolist()))
        pairs.sort(key=lambda p: p[1], reverse=True)
        return pairs


if __name__ == "__main__":
    model = FeatureAttentionRiskLSTM()
    x = torch.randn(4, 20, 6)
    risk, imp = model(x)
    print(f"risk: {risk.shape}, importance: {imp.shape}")
    print("ranking:", model.get_feature_ranking(
        imp, ['wind', 'speed', 'roll', 'pitch', 'alt', 'sig']))
