"""Fair 9-vs-12 feature experiment on manual AirSim flights.

Both models use the same flight-disjoint split, pre-event windows, first
post-takeoff obstacle-collision labels, architecture, and training settings.
The existing production checkpoint is never overwritten.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import argparse
import json
import math
import random
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

try:
    from .air_risk_features import FEATURE_NAMES, LEGACY_FEATURE_NAMES, row_to_features
    from .attention_risk_lstm import FeatureAttentionRiskLSTM
except ImportError:
    from air_risk_features import FEATURE_NAMES, LEGACY_FEATURE_NAMES, row_to_features
    from attention_risk_lstm import FeatureAttentionRiskLSTM


HERE = Path(__file__).resolve().parent
DATA = HERE / "data/UAV_gust_manual_sessions.jsonl"
OUT = HERE / "models/manual_12_experiment"
SEQ_LEN = 20
HORIZON_S = 1.0
SEED = 42
BATCH_SIZE = 128
MAX_EPOCHS = 80
PATIENCE = 12
LEARNING_RATE = 1e-3
DEVICE = torch.device("cpu")


def load_flights():
    flights = defaultdict(list)
    with DATA.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                flights[row["flight_id"]].append(row)
    for rows in flights.values():
        rows.sort(key=lambda r: float(r["timestamp"]))
    return dict(flights)


def make_windows(flights):
    """Create matched 9/12 windows; never label ordinary ground contact positive."""
    data = {}
    audit = {}
    for flight_id, rows in flights.items():
        z0 = float(np.median([r["position"][2] for r in rows[:5]]))
        takeoff = next((i for i, r in enumerate(rows)
            if z0 - float(r["position"][2]) >= 0.5
            and math.sqrt(sum(float(v) ** 2 for v in r["velocity"])) > 0.2), None)
        if takeoff is None:
            audit[flight_id] = {"status": "no_takeoff", "rows": len(rows)}
            continue
        obstacle = next((i for i in range(takeoff + 1, len(rows))
            if rows[i].get("obstacle_collision") is True), None)
        ground = next((i for i in range(takeoff + 1, len(rows))
            if rows[i].get("ground_collision") is True), None)
        events = [(i, typ) for i, typ in ((obstacle, "obstacle"), (ground, "ground")) if i is not None]
        first_event, event_type = min(events) if events else (None, "none")
        timestamps = np.asarray([float(r["timestamp"]) for r in rows], dtype=np.float64)
        raw9 = np.stack([row_to_features(r, LEGACY_FEATURE_NAMES) for r in rows])
        raw12 = np.stack([row_to_features(r, FEATURE_NAMES) for r in rows])
        x9, x12, y = [], [], []
        first_end = takeoff + SEQ_LEN - 1
        last_end = (first_event - 1) if first_event is not None else len(rows) - 1
        for end in range(first_end, last_end + 1):
            future_limit = timestamps[end] + HORIZON_S
            is_positive = bool(event_type == "obstacle" and timestamps[first_event] <= future_limit)
            # For a negative label, require that the entire horizon was observed
            # before an event or the end of the data collection.
            if not is_positive:
                observation_end = timestamps[first_event] if first_event is not None else timestamps[-1]
                if future_limit > observation_end:
                    continue
            x9.append(raw9[end - SEQ_LEN + 1:end + 1])
            x12.append(raw12[end - SEQ_LEN + 1:end + 1])
            y.append(float(is_positive))
        if not y:
            audit[flight_id] = {"status": "no_eligible_windows", "rows": len(rows), "event": event_type}
            continue
        data[flight_id] = {
            "x9": np.asarray(x9, dtype=np.float32),
            "x12": np.asarray(x12, dtype=np.float32),
            "y": np.asarray(y, dtype=np.float32),
        }
        audit[flight_id] = {
            "status": "used", "rows": len(rows), "takeoff_row": takeoff,
            "first_event_row": first_event, "first_event_type": event_type,
            "windows": len(y), "positive_windows": int(sum(y)),
        }
    return data, audit


def split_flights(data, audit):
    rng = np.random.default_rng(SEED)
    groups = defaultdict(list)
    for flight_id in data:
        # Stratify on whether the flight actually contributed positive windows.
        group = "positive" if audit[flight_id]["positive_windows"] else "negative"
        groups[group].append(flight_id)
    result = {"train": [], "val": [], "test": []}
    for ids in groups.values():
        ids = sorted(ids)
        rng.shuffle(ids)
        n_test = max(1, round(len(ids) * 0.20))
        n_val = max(1, round(len(ids) * 0.15))
        result["test"].extend(ids[:n_test])
        result["val"].extend(ids[n_test:n_test + n_val])
        result["train"].extend(ids[n_test + n_val:])
    return result


def arrays_for_split(data, ids, schema):
    return (np.concatenate([data[f][schema] for f in ids]),
            np.concatenate([data[f]["y"] for f in ids]))


def normalize_arrays(train, val, test):
    mean = train.reshape(-1, train.shape[-1]).mean(axis=0)
    std = train.reshape(-1, train.shape[-1]).std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return ([(array - mean).astype(np.float32) / std for array in (train, val, test)], mean, std)


def infer(model, x):
    model.eval()
    outputs = []
    with torch.inference_mode():
        for (xb,) in DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=512):
            p, _ = model(xb.to(DEVICE))
            outputs.append(p[:, 0].cpu().numpy())
    return np.concatenate(outputs)


def choose_threshold(y, p):
    candidates = np.linspace(0.01, 0.99, 99)
    scores = [f1_score(y, p >= threshold, zero_division=0) for threshold in candidates]
    return float(candidates[int(np.argmax(scores))])


def metrics(y, p, threshold):
    predicted = p >= threshold
    return {
        "windows": int(len(y)), "positive_windows": int(y.sum()),
        "threshold": float(threshold),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "tp": int(((predicted == 1) & (y == 1)).sum()),
        "fp": int(((predicted == 1) & (y == 0)).sum()),
        "tn": int(((predicted == 0) & (y == 0)).sum()),
        "fn": int(((predicted == 0) & (y == 1)).sum()),
    }


def train_one(schema, feature_names, data, split):
    train_x, train_y = arrays_for_split(data, split["train"], schema)
    val_x, val_y = arrays_for_split(data, split["val"], schema)
    test_x, test_y = arrays_for_split(data, split["test"], schema)
    (train_x, val_x, test_x), mean, std = normalize_arrays(train_x, val_x, test_x)
    torch.manual_seed(SEED)
    model = FeatureAttentionRiskLSTM(n_features=len(feature_names)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
                        batch_size=BATCH_SIZE, shuffle=True)
    best_ap, best_state, best_epoch, no_improve = -1.0, None, 0, 0
    started = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            p, _ = model(xb.to(DEVICE))
            loss = torch.nn.functional.binary_cross_entropy(p[:, 0], yb.to(DEVICE))
            loss.backward()
            optimizer.step()
        val_p = infer(model, val_x)
        val_ap = float(average_precision_score(val_y, val_p))
        if val_ap > best_ap + 1e-4:
            best_ap = val_ap
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if epoch == 1 or epoch % 5 == 0 or no_improve >= PATIENCE:
            print(f"{schema} epoch={epoch:03d} val_pr_auc={val_ap:.4f} best={best_ap:.4f} elapsed={time.perf_counter()-started:.1f}s", flush=True)
        if no_improve >= PATIENCE:
            break
    model.load_state_dict(best_state)
    val_p = infer(model, val_x)
    threshold = choose_threshold(val_y, val_p)
    test_p = infer(model, test_x)
    return {
        "schema": schema, "features": list(feature_names), "best_epoch": best_epoch,
        "best_val_pr_auc": best_ap, "val": metrics(val_y, val_p, threshold),
        "test": metrics(test_y, test_p, threshold),
        "normalization_mean": mean.tolist(), "normalization_std": std.tolist(),
        "model": model,
    }


def main():
    global SEED, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    SEED = args.seed
    if SEED != 42:
        OUT = OUT / f"seed_{SEED}"
    random.seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    OUT.mkdir(parents=True, exist_ok=True)
    flights = load_flights()
    data, audit = make_windows(flights)
    split = split_flights(data, audit)
    print(f"flights_raw={len(flights)} usable={len(data)} total_windows={sum(len(v['y']) for v in data.values())}", flush=True)
    for name, ids in split.items():
        print(name, "flights", len(ids), "windows", sum(len(data[f]["y"]) for f in ids),
              "positive", sum(int(data[f]["y"].sum()) for f in ids), flush=True)
    results = []
    for schema, names in (("x9", LEGACY_FEATURE_NAMES), ("x12", FEATURE_NAMES)):
        result = train_one(schema, names, data, split)
        torch.save(result.pop("model").state_dict(), OUT / f"attention_lstm_{schema}.pt")
        results.append(result)
        print(schema, "test", json.dumps(result["test"], ensure_ascii=False), flush=True)
    report = {
        "data_file": str(DATA), "raw_flights": len(flights), "usable_flights": len(data),
        "task": "first obstacle collision within (t,t+1s], using only pre-event 20-step windows",
        "sequence_length": SEQ_LEN, "horizon_seconds": HORIZON_S,
        "seed": SEED, "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE, "learning_rate": LEARNING_RATE,
        "split": split, "audit": audit, "results": results,
    }
    (OUT / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("report", OUT / "comparison.json", flush=True)


if __name__ == "__main__":
    main()
