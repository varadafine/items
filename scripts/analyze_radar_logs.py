"""Audit the three expert radar logs without treating device classes as truth.

Run with the project's .venv/Scripts/python.exe. Source files are read only.
Outputs use UTF-8 (CSV with BOM); native logs use GBK and tab delimiters.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_SOURCE = Path(r"C:\Users\Administrator\Desktop\专家提供真实数据\气象台")
NAMES = ["20260613_201032.dat(1).txt", "20260613_224616.dat(1).txt", "20260614_234131.dat(1).txt"]
RAW_COLUMNS = ["时间", "搜索方位", "扫描周期", "目标有效标志", "航迹编号", "幅度", "目标速度", "目标距离", "目标方位", "目标俯仰", "目标高度", "雷达ID", "置信度", "目标类型", "UTC_TIME", "经度", "纬度", "高度"]
NUMERIC = [c for c in RAW_COLUMNS if c not in ("时间", "目标类型", "UTC_TIME")]


def stats(series):
    s = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    return {"n": len(s), "min": s.min(), "p05": s.quantile(.05), "median": s.median(), "p95": s.quantile(.95), "max": s.max()}


def counts(series):
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).items()}


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    return value


def analyze(source, output):
    output.mkdir(parents=True, exist_ok=True)
    files, frames, tracks, numeric_stats, centers = [], [], [], [], []
    for number, name in enumerate(NAMES, 1):
        path = source / name
        raw_text = path.read_text(encoding="gbk")
        df = pd.read_csv(path, sep="\t", encoding="gbk", dtype=str)
        assert list(df.columns) == RAW_COLUMNS, name
        structural_errors = sum(len(line.split("\t")) != 18 for line in raw_text.splitlines()[1:] if line.strip())
        missing = int(df.isna().sum().sum())
        duplicates = int(df.duplicated().sum())
        invalid_numeric = {}
        for c in NUMERIC:
            converted = pd.to_numeric(df[c], errors="coerce")
            invalid_numeric[c] = int((converted.isna() & df[c].notna()).sum())
            df[c] = converted
        df["source_file"] = name
        df["source_row"] = np.arange(2, len(df) + 2)
        df["time_local"] = pd.to_datetime(df["时间"], format="%Y/%m/%d_%H:%M:%S.%f", errors="coerce")
        df["time_utc_raw_parsed"] = pd.to_datetime(df["UTC_TIME"], format="%Y/%m/%d_%H:%M:%S.%f", errors="coerce")
        df["track_key"] = name + "|" + df["雷达ID"].astype(str) + "|" + df["航迹编号"].astype(str)
        df = df.sort_values(["track_key", "time_local", "source_row"], kind="stable")
        group = df.groupby("track_key", sort=False)
        df["dt_s"] = group["time_local"].diff().dt.total_seconds()
        df["scan_step"] = group["扫描周期"].diff()
        df["range_step_m_assumed"] = group["目标距离"].diff()
        df["range_rate_mps_assumed"] = df["range_step_m_assumed"] / df["dt_s"]
        df["previous_reported_speed"] = group["目标速度"].shift()
        df["speed_residual_current"] = df["range_rate_mps_assumed"] + df["目标速度"]
        df["speed_residual_previous"] = df["range_rate_mps_assumed"] + df["previous_reported_speed"]
        # Segment threshold is exploratory, not a vendor definition of track identity.
        df["segment_number"] = df.groupby("track_key")["dt_s"].transform(lambda s: ((s.isna()) | (s > 5) | (s <= 0)).cumsum())
        df["segment_key"] = df["track_key"] + "|" + df["segment_number"].astype(str)
        df["height_geometry_m_assumed"] = df["目标距离"] * np.sin(np.deg2rad(df["目标俯仰"]))
        df["height_geometry_residual"] = df["目标高度"] - df["height_geometry_m_assumed"]
        df["altitude_offset"] = df["高度"] - df["目标高度"]
        df["utc_invalid"] = df.time_utc_raw_parsed.isna()
        df["height_geometry_abs_error_gt_1m"] = df.height_geometry_residual.abs() > 1
        terminal_indices = df.groupby("track_key", sort=False).tail(1).index
        df["is_terminal_record_in_file"] = df.index.isin(terminal_indices)
        for c in NUMERIC + ["dt_s", "altitude_offset", "height_geometry_residual"]:
            numeric_stats.append({"source_file": name, "field": c, **stats(df[c])})
        for key, t in df.groupby("track_key", sort=False):
            intervals = t["dt_s"].dropna()
            tracks.append({
                "source_file": name, "track_key": key, "radar_id": int(t["雷达ID"].iloc[0]), "track_id": int(t["航迹编号"].iloc[0]),
                "rows": len(t), "start": t["time_local"].min(), "end": t["time_local"].max(),
                "span_s": (t["time_local"].max() - t["time_local"].min()).total_seconds(),
                "median_dt_s": intervals.median(), "max_dt_s": intervals.max(), "gaps_gt_5s": int((intervals > 5).sum()),
                "segments_5s": t["segment_key"].nunique(), "flag_0": int((t["目标有效标志"] == 0).sum()),
                "flag_1": int((t["目标有效标志"] == 1).sum()), "classes_raw": ";".join(sorted(t["目标类型"].unique())),
                "class_changes": int((t["目标类型"] != t["目标类型"].shift()).sum() - 1),
                "range_min": t["目标距离"].min(), "range_max": t["目标距离"].max(),
                "relative_height_min": t["目标高度"].min(), "relative_height_max": t["目标高度"].max(),
            })
        track_table = pd.DataFrame(tracks)
        own_tracks = track_table[track_table.source_file == name]
        segments = df.groupby("segment_key").size()
        good_pairs = df[(df.dt_s > 0) & (df.dt_s <= 5)]
        flag_comparison = {}
        for flag, f in good_pairs.groupby("目标有效标志"):
            prev = group[["幅度", "目标速度", "目标俯仰", "目标高度"]].shift().loc[f.index]
            same = (f[["幅度", "目标速度", "目标俯仰", "目标高度"]] == prev).all(axis=1)
            flag_comparison[str(flag)] = {
                "rows": int((df["目标有效标志"] == flag).sum()),
                "terminal_rows": int(((df["目标有效标志"] == flag) & df.is_terminal_record_in_file).sum()),
                "height_geometry_abs_error": stats(df.loc[df["目标有效标志"] == flag, "height_geometry_residual"].abs()),
                "adjacent_pairs_le_5s": len(f), "repeat_amp_speed_elevation_height": int(same.sum()),
                "abs_previous_speed_residual": stats(f.speed_residual_previous.abs()),
                "range_7_5_grid_fraction": float((np.abs(f["目标距离"] / 7.5 - np.round(f["目标距离"] / 7.5)) < .001).mean()),
            }
        # Test two common position conversions. This is a consistency fit, not surveyed geodesy.
        lat_ref = float(df["纬度"].median())
        meters_lat = np.pi * 6378137.0 / 180
        meters_lon = meters_lat * np.cos(np.deg2rad(lat_ref))
        for mode in ("slant_as_horizontal", "horizontal_cos_elevation"):
            radial = df["目标距离"].to_numpy().copy()
            if mode == "horizontal_cos_elevation":
                radial *= np.cos(np.deg2rad(df["目标俯仰"]))
            east = radial * np.sin(np.deg2rad(df["目标方位"]))
            north = radial * np.cos(np.deg2rad(df["目标方位"]))
            lon0 = np.median(df["经度"] - east / meters_lon)
            lat0 = np.median(df["纬度"] - north / meters_lat)
            residual = np.hypot((df["经度"] - lon0) * meters_lon - east, (df["纬度"] - lat0) * meters_lat - north)
            centers.append({"source_file": name, "mode": mode, "lon0_fit": lon0, "lat0_fit": lat0, "residual_m": stats(residual)})
            if mode == "horizontal_cos_elevation":
                df["east_m_assumed"] = east
                df["north_m_assumed"] = north
        original_order = df.sort_values("source_row")
        delta_local_minus_utc = (df.time_local - df.time_utc_raw_parsed).dt.total_seconds()
        files.append({
            "file": name, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": len(df), "columns": 18, "structural_errors": structural_errors, "missing_cells": missing,
            "numeric_conversion_errors": invalid_numeric, "duplicate_rows": duplicates,
            "invalid_local_times": int(df.time_local.isna().sum()), "invalid_utc_times": int(df.time_utc_raw_parsed.isna().sum()),
            "utc_raw_counts_if_invalid": counts(df.loc[df.time_utc_raw_parsed.isna(), "UTC_TIME"]),
            "time_start": df.time_local.min(), "time_end": df.time_local.max(),
            "duration_s": (df.time_local.max() - df.time_local.min()).total_seconds(),
            "local_time_backsteps": int((original_order.time_local.diff().dt.total_seconds() < 0).sum()),
            "local_minus_utc_minus_8h_s": stats(delta_local_minus_utc - 28800),
            "radar_ids": counts(df["雷达ID"]), "track_ids": int(df.track_key.nunique()),
            "raw_track_ids_without_radar_scope": int(df["航迹编号"].nunique()),
            "track_numbers_reused_between_radar_ids": int((df.groupby("航迹编号")["雷达ID"].nunique() > 1).sum()),
            "radar_id_details": [{"radar_id": int(rid), "rows": len(r), "track_ids": r.track_key.nunique(),
                                   "start": r.time_local.min(), "end": r.time_local.max()}
                                  for rid, r in df.groupby("雷达ID")],
            "classes_raw": counts(df["目标类型"]), "confidence": counts(df["置信度"]), "flags_raw": counts(df["目标有效标志"]),
            "scan_min": df["扫描周期"].min(), "scan_max": df["扫描周期"].max(), "scan_count": df["扫描周期"].nunique(),
            "track_rows": stats(own_tracks.rows), "track_span_s": stats(own_tracks.span_s), "track_dt_s": stats(df.dt_s),
            "single_row_tracks": int((own_tracks.rows == 1).sum()), "tracks_ge_20_rows": int((own_tracks.rows >= 20).sum()),
            "tracks_ge_60s_span": int((own_tracks.span_s >= 60).sum()), "tracks_with_class_changes": int((own_tracks.class_changes > 0).sum()),
            "gaps_gt_5s": int((df.dt_s > 5).sum()), "nonpositive_track_dt": int((df.dt_s <= 0).sum()),
            "segments_5s": len(segments), "segments_ge_20_rows": int((segments >= 20).sum()),
            "windows_20_stride_1_5s_segments": int(np.maximum(segments - 19, 0).sum()),
            "segments_ge_20_rows_flag0_only": int((df[df["目标有效标志"] == 0].groupby("segment_key").size() >= 20).sum()),
            "windows_20_stride_1_flag0_only": int(np.maximum(df[df["目标有效标志"] == 0].groupby("segment_key").size() - 19, 0).sum()),
            "flag_comparison": flag_comparison,
            "speed_vs_range_rate_corr_le_5s": good_pairs[["目标速度", "range_rate_mps_assumed"]].corr().iloc[0, 1],
            "range_stats": stats(df["目标距离"]), "speed_raw_stats": stats(df["目标速度"]),
            "relative_height_stats": stats(df["目标高度"]), "altitude_stats": stats(df["高度"]),
            "altitude_offset_stats": stats(df.altitude_offset), "geometry_height_abs_error": stats(df.height_geometry_residual.abs()),
            "longitude_stats": stats(df["经度"]), "latitude_stats": stats(df["纬度"]),
        })
        frames.append(df)
    all_data = pd.concat(frames, ignore_index=True)
    track_table = pd.DataFrame(tracks)
    pd.DataFrame(numeric_stats).to_csv(output / "field_statistics.csv", index=False, encoding="utf-8-sig")
    track_table.sort_values(["source_file", "rows"], ascending=[True, False]).to_csv(output / "track_summary.csv", index=False, encoding="utf-8-sig")
    summary = {"files": files, "position_fits": centers, "totals": {
        "rows": len(all_data), "file_scoped_track_ids": len(track_table),
        "observed_duration_sum_s": sum(f["duration_s"] for f in files),
        "classes_raw": counts(all_data["目标类型"]), "flags_raw": counts(all_data["目标有效标志"]),
        "tracks_ge_20_rows": int((track_table.rows >= 20).sum()), "tracks_with_class_changes": int((track_table.class_changes > 0).sum()),
        "segments_ge_20_rows": sum(f["segments_ge_20_rows"] for f in files),
        "windows_20_stride_1_5s_segments": sum(f["windows_20_stride_1_5s_segments"] for f in files),
        "segments_ge_20_rows_flag0_only": sum(f["segments_ge_20_rows_flag0_only"] for f in files),
        "windows_20_stride_1_flag0_only": sum(f["windows_20_stride_1_flag0_only"] for f in files),
        "negative_relative_height_rows": int((all_data["目标高度"] < 0).sum()),
        "flag1_rows_at_track_end": int(((all_data["目标有效标志"] == 1) & all_data.is_terminal_record_in_file).sum()),
        "track_length_bins": {"1_to_2": int((track_table.rows <= 2).sum()),
                              "3_to_5": int(track_table.rows.between(3, 5).sum()),
                              "6_to_10": int(track_table.rows.between(6, 10).sum()),
                              "11_to_19": int(track_table.rows.between(11, 19).sum()),
                              "20_or_more": int((track_table.rows >= 20).sum())},
    }}
    (output / "summary.json").write_text(json.dumps(safe_json(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    for i, df in enumerate(frames):
        ax = axes[0, i]
        for j, (_, t) in enumerate(df.groupby("segment_key")):
            color = colors[j % 20]
            ax.plot(t.east_m_assumed, t.north_m_assumed, lw=.8, alpha=.55, color=color)
            ax.scatter(t.east_m_assumed, t.north_m_assumed, s=5, alpha=.5, color=color)
        ax.scatter([0], [0], marker="^", color="black", s=55, label="Radar coordinate origin")
        ax.set(title=f"Session {i+1}: {NAMES[i][:15]}", xlabel="East offset (m, assumed)", ylabel="North offset (m, assumed)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=.25)
        ax = axes[1, i]
        for j, (_, t) in enumerate(df.groupby("segment_key")):
            seconds = (t.time_local - df.time_local.min()).dt.total_seconds()
            ax.plot(seconds, t["目标高度"], lw=.8, alpha=.65, color=colors[j % 20])
            ax.scatter(seconds, t["目标高度"], s=5, alpha=.5, color=colors[j % 20])
        ax.set(xlabel="Seconds from session start", ylabel="Radar relative height (m, assumed)")
        ax.grid(alpha=.25)
    fig.suptitle("Radar log audit: device track IDs, not verified physical targets\nLines split at gaps >5 s; raw class labels are not ground truth; position conversion requires vendor confirmation", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, .93])
    fig.savefig(output / "radar_tracks_overview.png", dpi=180)
    plt.close(fig)
    # Preserve the source file's original row order in the record-level export.
    all_data.sort_values(["source_file", "source_row"]).to_csv(output / "records_with_quality.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(safe_json(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=Path("output/radar_log_analysis"))
    args = parser.parse_args()
    analyze(args.source, args.output)
