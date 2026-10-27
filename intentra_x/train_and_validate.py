"""
一键训练 + 评估 + 物理检验
运行: python train_and_validate.py
输出: models/best_risk_model.pt, training_curve.png, feature_importance.png, training_report.txt
"""
import os
import io
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from synthetic_data_generator import SyntheticDataGenerator
from attention_risk_lstm import FeatureAttentionRiskLSTM

# ─── 全局配置 ───
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS, BATCH, LR, ES_PATIENCE = 200, 32, 1e-3, 15
REPORT = io.StringIO()


def log(msg=""):
    print(msg)
    REPORT.write(str(msg) + "\n")


# ───────── 第1步：数据准备 ─────────
def step1_data():
    log("=" * 60); log("第1步：生成数据并划分 70/15/15"); log("=" * 60)
    gen = SyntheticDataGenerator(seed=42)
    X, y = gen.generate(n=5000, seq_len=20)
    feat_names = gen.get_feature_names()
    n = len(X); n_tr, n_va = int(n * 0.7), int(n * 0.15)
    idx = np.random.RandomState(0).permutation(n)
    tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]

    def mk(ids, shuffle):
        ds = TensorDataset(torch.tensor(X[ids]), torch.tensor(y[ids]))
        return DataLoader(ds, batch_size=BATCH, shuffle=shuffle)

    log(f"  Train/Val/Test = {len(tr)}/{len(va)}/{len(te)}")
    log(f"  风险样本占比: {y.mean()*100:.2f}%")
    log("  特征范围:")
    for i, n_ in enumerate(feat_names):
        log(f"    {n_:18s} {X[..., i].min():7.2f} ~ {X[..., i].max():7.2f}")
    return mk(tr, True), mk(va, False), mk(te, False), feat_names, gen


# ───────── 第2步：训练 ─────────
def step2_train(tr_dl, va_dl):
    log("\n" + "=" * 60); log(f"第2步：训练（device={DEVICE}）"); log("=" * 60)
    model = FeatureAttentionRiskLSTM().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)
    crit = nn.MSELoss()
    tr_hist, va_hist, best_va, best_ep, no_imp = [], [], 1e9, 0, 0
    ckpt_path = os.path.join(MODEL_DIR, 'best_risk_model.pt')

    for ep in range(1, EPOCHS + 1):
        model.train(); tot, cnt = 0.0, 0
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred, _ = model(xb); loss = crit(pred.squeeze(-1), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * xb.size(0); cnt += xb.size(0)
        tr_loss = tot / cnt

        model.eval(); tot, cnt = 0.0, 0
        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred, _ = model(xb); tot += crit(pred.squeeze(-1), yb).item() * xb.size(0); cnt += xb.size(0)
        va_loss = tot / cnt
        sch.step(va_loss); tr_hist.append(tr_loss); va_hist.append(va_loss)
        log(f"  Epoch {ep:3d} | train={tr_loss:.4f} | val={va_loss:.4f}")

        if va_loss < best_va - 1e-5:
            best_va, best_ep, no_imp = va_loss, ep, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_imp += 1
            if no_imp >= ES_PATIENCE:
                log(f"  早停于 epoch {ep}（最佳 epoch={best_ep}, val={best_va:.4f}）"); break

    # 训练曲线（双 Y 轴）
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(tr_hist, 'b-', label='train_loss'); ax1.set_xlabel('epoch')
    ax1.set_ylabel('train_loss', color='b'); ax1.tick_params(axis='y', labelcolor='b')
    ax2 = ax1.twinx(); ax2.plot(va_hist, 'r-', label='val_loss')
    ax2.set_ylabel('val_loss', color='r'); ax2.tick_params(axis='y', labelcolor='r')
    plt.title('Training Curve'); fig.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'training_curve.png'), dpi=120); plt.close()
    log(f"  最佳模型已保存: {ckpt_path}（best_val={best_va:.4f}, epoch={best_ep}）")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    return model


# ───────── 第3步：评估 ─────────
def step3_eval(model, te_dl, feat_names):
    log("\n" + "=" * 60); log("第3步：测试集评估 + 全局特征重要性"); log("=" * 60)
    model.eval(); preds, gts, imps = [], [], []
    with torch.no_grad():
        for xb, yb in te_dl:
            xb = xb.to(DEVICE)
            p, imp = model(xb); preds.append(p.squeeze(-1).cpu().numpy())
            gts.append(yb.numpy()); imps.append(imp.cpu().numpy())
    p = np.concatenate(preds); g = np.concatenate(gts); imp = np.concatenate(imps)
    mse = float(((p - g) ** 2).mean()); mae = float(np.abs(p - g).mean())
    ss_res = ((g - p) ** 2).sum(); ss_tot = ((g - g.mean()) ** 2).sum() + 1e-9
    r2 = 1.0 - ss_res / ss_tot
    log(f"  Test MSE={mse:.4f} | MAE={mae:.4f} | R²={r2:.4f}")

    mean_imp, std_imp = imp.mean(0), imp.std(0)
    order = np.argsort(-mean_imp)
    log("  全局特征重要性（均值±std，降序）:")
    for i in order:
        log(f"    {feat_names[i]:18s} {mean_imp[i]:.4f} ± {std_imp[i]:.4f}")

    plt.figure(figsize=(8, 4.5))
    xs = np.arange(len(feat_names))
    plt.bar(xs, mean_imp[order], yerr=std_imp[order], capsize=5, color='steelblue')
    plt.xticks(xs, [feat_names[i] for i in order], rotation=20)
    plt.ylabel('importance'); plt.title('Global Feature Importance')
    plt.tight_layout(); plt.savefig(os.path.join(MODEL_DIR, 'feature_importance.png'), dpi=120); plt.close()
    return p, g, imp, mean_imp, order


# ───────── 第4步：物理检验 ─────────
def step4_physics(model, feat_names, global_order):
    """global_order: 测试集上算出的全局重要性排序索引（降序）"""
    log("\n" + "=" * 60); log("第4步：物理检验（新种子1000条）"); log("=" * 60)
    gen2 = SyntheticDataGenerator(seed=2026)
    Xn, _ = gen2.generate(n=1000, seq_len=20)
    xt = torch.tensor(Xn).to(DEVICE)
    model.eval()
    with torch.no_grad():
        risk, imp = model(xt)
    risk = risk.squeeze(-1).cpu().numpy(); imp = imp.cpu().numpy()

    hi, lo = risk > 0.7, risk < 0.3
    log(f"  A-高风险画像: hi={hi.sum()} / lo={lo.sum()}")
    if hi.sum() > 0 and lo.sum() > 0:
        hi_m = Xn[hi].reshape(-1, 6).mean(0); lo_m = Xn[lo].reshape(-1, 6).mean(0)
        for i, n_ in enumerate(feat_names):
            log(f"    {n_:18s} hi={hi_m[i]:7.2f} | lo={lo_m[i]:7.2f} | Δ={hi_m[i]-lo_m[i]:+.2f}")

    # B-排序稳定：以"测试集全局 Top-1 / Top-2"为参考，统计本批样本中各样本级排序的命中率
    t1_idx, t2_idx = int(global_order[0]), int(global_order[1])
    t1_name, t2_name = feat_names[t1_idx], feat_names[t2_idx]
    sample_top1 = np.argmax(imp, axis=1)
    sample_top2 = np.argsort(-imp, axis=1)[:, 1]
    top1_hit = (sample_top1 == t1_idx).mean()
    top2_hit = (sample_top2 == t2_idx).mean()
    log(f"  B-排序稳定（参考全局 Top-1={t1_name}, Top-2={t2_name}）:")
    log(f"    样本级 Top-1 命中率 {top1_hit*100:.1f}% | Top-2 命中率 {top2_hit*100:.1f}%")

    sub = Xn[..., 5].mean(axis=1) < 0.3
    if sub.sum() > 5:
        sig_w_sub = imp[sub, 5].mean(); sig_w_all = imp[:, 5].mean()
        log(f"  C-条件场景: signal<0.3 子集 sig权重={sig_w_sub:.4f} vs 全集={sig_w_all:.4f}")

    normal = (risk < 0.3)
    if normal.sum() > 10:
        Xa = Xn[normal][:50].copy(); Xa[..., 0] = 19.0
        with torch.no_grad():
            r2_, _ = model(torch.tensor(Xa).to(DEVICE))
        log(f"  D-对抗扰动: 正常样本风险均值 {risk[normal][:50].mean():.4f} → wind→19后 {r2_.cpu().numpy().mean():.4f}")


# ───────── 第5步：保存报告 ─────────
def step5_save():
    path = os.path.join(MODEL_DIR, 'training_report.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(REPORT.getvalue())
    print("\n" + "★" * 30); print(f"全部完成 ✔  报告: {path}"); print("★" * 30)


def main():
    log(f"设备: {DEVICE}")
    tr_dl, va_dl, te_dl, feat_names, _ = step1_data()
    model = step2_train(tr_dl, va_dl)
    _, _, _, _, order = step3_eval(model, te_dl, feat_names)
    step4_physics(model, feat_names, order)
    step5_save()


if __name__ == "__main__":
    main()
