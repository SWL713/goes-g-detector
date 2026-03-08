from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.signal import find_peaks


OUTDIR = Path(__file__).resolve().parent.parent / "output"
OUTDIR.mkdir(parents=True, exist_ok=True)

BASE = "https://services.swpc.noaa.gov/json/goes"
URL_PRIMARY = f"{BASE}/primary/magnetometers-7-day.json"
URL_SECONDARY = f"{BASE}/secondary/magnetometers-7-day.json"
URL_SOURCES = f"{BASE}/instrument-sources.json"
URL_LONGITUDES = f"{BASE}/satellite-longitudes.json"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "goes-g-detector/0.3"})


@dataclass
class DetectionResult:
    timestamp: pd.Timestamp
    source: str
    score: float
    status: str
    note: str
    peak_time: pd.Timestamp


def fetch_json(url: str):
    response = SESSION.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def load_metadata() -> dict:
    sources = fetch_json(URL_SOURCES)
    longitudes = fetch_json(URL_LONGITUDES)

    if isinstance(sources, list) and len(sources) > 0:
        sources = sources[-1]

    lon_map = {}
    if isinstance(longitudes, list):
        for row in longitudes:
            sat = row.get("satellite")
            lon = row.get("longitude")
            try:
                if sat is not None and lon is not None:
                    lon_map[int(sat)] = float(lon)
            except Exception:
                continue

    mag_sources = sources.get("magnetometers", {}) if isinstance(sources, dict) else {}
    primary_sat = mag_sources.get("primary")
    secondary_sat = mag_sources.get("secondary")

    return {
        "primary_sat": int(primary_sat) if primary_sat is not None else None,
        "secondary_sat": int(secondary_sat) if secondary_sat is not None else None,
        "longitudes": lon_map,
    }


def normalize_mag_json(data, source_name: str) -> pd.DataFrame:
    df = pd.DataFrame(data)
    if df.empty:
        raise ValueError(f"{source_name} dataset is empty")

    time_col = None
    for col in df.columns:
        low = col.lower()
        if low in {"time_tag", "time", "date"} or "time" in low:
            time_col = col
            break

    if time_col is None:
        raise ValueError(f"No time column found in {source_name}: {list(df.columns)}")

    df["time_utc"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=["time_utc"]).copy()

    rename = {}
    for col in df.columns:
        low = col.lower()
        if low in {"hp", "h_p"}:
            rename[col] = "Hp"
        elif low in {"he", "h_e"}:
            rename[col] = "He"
        elif low in {"hn", "h_n"}:
            rename[col] = "Hn"
        elif low in {"ht", "bt", "total", "total_field"}:
            rename[col] = "Ht"
        elif low == "satellite":
            rename[col] = "satellite"

    df = df.rename(columns=rename)

    keep = ["time_utc"] + [c for c in ["Hp", "He", "Hn", "Ht", "satellite"] if c in df.columns]
    df = df[keep].copy()
    df["source_feed"] = source_name
    df = df.sort_values("time_utc").drop_duplicates(subset=["time_utc"])

    return df


def choose_east_west(primary_df: pd.DataFrame, secondary_df: pd.DataFrame, meta: dict) -> dict:
    lon_map = meta["longitudes"]
    primary_sat = meta["primary_sat"]
    secondary_sat = meta["secondary_sat"]

    primary_lon = lon_map.get(primary_sat)
    secondary_lon = lon_map.get(secondary_sat)

    candidates = [
        ("primary", primary_df, primary_sat, primary_lon),
        ("secondary", secondary_df, secondary_sat, secondary_lon),
    ]
    candidates = [c for c in candidates if c[3] is not None]

    if len(candidates) < 2:
        raise ValueError("Could not resolve both GOES longitudes from SWPC metadata")

    east = min(candidates, key=lambda x: x[3])
    west = max(candidates, key=lambda x: x[3])

    return {
        "east_name": east[0],
        "east_df": east[1],
        "east_sat": east[2],
        "east_lon": east[3],
        "west_name": west[0],
        "west_df": west[1],
        "west_sat": west[2],
        "west_lon": west[3],
    }


def prep_trace(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values("time_utc")

    if "Hp" not in out.columns:
        raise ValueError(f"This script expects an Hp column. Columns found: {list(out.columns)}")

    keep_cols = ["time_utc"] + [c for c in ["Hp", "He", "Hn", "Ht"] if c in out.columns]
    out = out[keep_cols].copy()

    for col in ["Hp", "He", "Hn", "Ht"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["Hp"]).copy()

    out = (
        out.set_index("time_utc")
        .resample("1min")
        .median(numeric_only=True)
        .interpolate(limit=5)
        .reset_index()
    )

    if out.empty:
        raise ValueError("Trace is empty after resampling")

    # Light smoothing keeps short-timescale events
    out["hp_smooth_3"] = out["Hp"].rolling(3, center=True, min_periods=1).median()
    out["hp_baseline_180"] = out["hp_smooth_3"].rolling(180, center=True, min_periods=30).median()
    out["hp_resid"] = out["hp_smooth_3"] - out["hp_baseline_180"]
    out["d1"] = out["hp_smooth_3"].diff()
    out["d2"] = out["d1"].diff()

    return out


def trough_depth_between(work: pd.DataFrame, i1: int, i2: int) -> float:
    if i2 <= i1 + 1:
        return 0.0
    y1 = float(work.loc[i1, "hp_smooth_3"])
    y2 = float(work.loc[i2, "hp_smooth_3"])
    trough = float(work.loc[i1:i2, "hp_smooth_3"].min())
    return min(y1, y2) - trough


def cluster_peaks(work: pd.DataFrame, peak_idx: np.ndarray) -> List[List[int]]:
    """
    Merge true double peaks, but allow closely spaced separate events.
    """
    if len(peak_idx) == 0:
        return []

    clusters: List[List[int]] = []
    current = [int(peak_idx[0])]

    for raw_idx in peak_idx[1:]:
        idx = int(raw_idx)
        prev = current[-1]
        dt = idx - prev
        trough_depth = trough_depth_between(work, prev, idx)

        # Merge only if very close and intervening dip is shallow
        should_merge = (dt <= 12 and trough_depth < 4.0)

        if should_merge:
            current.append(idx)
        else:
            clusters.append(current)
            current = [idx]

    clusters.append(current)
    return clusters


def detect_g_candidates(df: pd.DataFrame, source_label: str) -> List[DetectionResult]:
    work = df.copy().reset_index(drop=True)
    y = work["hp_resid"].to_numpy()

    if len(y) < 120:
        return []

    # More sensitive to smaller sharp peaks
    peak_idx, props = find_peaks(y, prominence=1.8, distance=8)

    if len(peak_idx) == 0:
        return []

    clusters = cluster_peaks(work, peak_idx)
    final_peaks = [cluster[-1] for cluster in clusters]

    detections: List[DetectionResult] = []

    for peak in final_peaks:
        start_search = peak + 2
        end_search = min(peak + 45, len(work) - 46)
        if start_search >= end_search:
            continue

        best_idx: Optional[int] = None
        best_score = -np.inf

        for i in range(start_search, end_search):
            future_10 = work.loc[i + 10, "hp_smooth_3"] - work.loc[i, "hp_smooth_3"]
            future_20 = work.loc[i + 20, "hp_smooth_3"] - work.loc[i, "hp_smooth_3"]
            future_30 = work.loc[i + 30, "hp_smooth_3"] - work.loc[i, "hp_smooth_3"]

            neg_frac_10 = (work.loc[i + 1:i + 10, "d1"] < 0).mean()
            neg_frac_20 = (work.loc[i + 1:i + 20, "d1"] < 0).mean()

            early_slope = work.loc[i + 5, "hp_smooth_3"] - work.loc[i, "hp_smooth_3"]
            prev_5 = work.loc[i, "hp_smooth_3"] - work.loc[max(0, i - 5), "hp_smooth_3"]

            # Reward earliest sustained negative turn and sharp charging
            score = (
                (-future_10) * 1.5
                + (-future_20) * 1.2
                + (-future_30) * 0.8
                + neg_frac_10 * 6.0
                + neg_frac_20 * 4.0
                + (-early_slope) * 1.5
                - abs(prev_5) * 0.2
            )

            valid = (
                future_10 <= -1.5 and
                future_20 <= -3.0 and
                neg_frac_10 >= 0.60 and
                neg_frac_20 >= 0.60
            )

            if valid and score > best_score:
                best_score = score
                best_idx = i

        if best_idx is not None:
            # Status: provisional if close in time to peak, confirmed otherwise
            dt_from_peak = best_idx - peak
            status = "provisional" if dt_from_peak <= 12 else "confirmed"

            detections.append(
                DetectionResult(
                    timestamp=work.loc[best_idx, "time_utc"],
                    source=source_label,
                    score=float(best_score),
                    status=status,
                    note="early sustained downslope after final release peak",
                    peak_time=work.loc[peak, "time_utc"],
                )
            )

    detections = sorted(detections, key=lambda d: d.timestamp)

    # Allow substorms within an hour; only suppress very close duplicates
    filtered: List[DetectionResult] = []
    for det in detections:
        if not filtered:
            filtered.append(det)
            continue

        dt_minutes = (det.timestamp - filtered[-1].timestamp).total_seconds() / 60.0
        if dt_minutes < 35:
            if det.score > filtered[-1].score:
                filtered[-1] = det
        else:
            filtered.append(det)

    return filtered


def pick_trace(east_df: pd.DataFrame, west_df: pd.DataFrame):
    east_prepped = prep_trace(east_df)
    east_candidates = detect_g_candidates(east_prepped, "GOES-East")
    if len(east_candidates) >= 1:
        return east_prepped, east_candidates, "GOES-East"

    west_prepped = prep_trace(west_df)
    west_candidates = detect_g_candidates(west_prepped, "GOES-West fallback")
    return west_prepped, west_candidates, "GOES-West fallback"


def save_outputs(trace_df: pd.DataFrame, detections: List[DetectionResult], chosen_source: str, meta: dict) -> None:
    trace_df.to_csv(OUTDIR / "goes_trace_used.csv", index=False)

    detections_df = pd.DataFrame(
        [
            {
                "g_time_utc": det.timestamp.isoformat(),
                "peak_time_utc": det.peak_time.isoformat(),
                "source": det.source,
                "score": det.score,
                "status": det.status,
                "note": det.note,
            }
            for det in detections
        ]
    )
    detections_df.to_csv(OUTDIR / "g_candidates.csv", index=False)

    fig = plt.figure(figsize=(18, 8))
    ax = fig.add_subplot(111)

    ax.plot(trace_df["time_utc"], trace_df["Hp"], label="Hp raw", linewidth=0.8)
    ax.plot(trace_df["time_utc"], trace_df["hp_smooth_3"], label="Hp smooth", linewidth=1.5)

    ymax = float(trace_df["hp_smooth_3"].max())
    ymin = float(trace_df["hp_smooth_3"].min())
    yspan = ymax - ymin if ymax != ymin else 1.0
    ytext = ymax - 0.03 * yspan

    for det in detections:
        ax.axvline(det.timestamp, linewidth=1.2)
        ax.text(
            det.timestamp,
            ytext,
            f"{det.timestamp.strftime('%m-%d %H:%M')}\n{det.status}",
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
        )

    title = (
        f"7-day GOES G detector v2 ({chosen_source}) | "
        f"primary sat={meta['primary_sat']} secondary sat={meta['secondary_sat']}"
    )
    ax.set_title(title)
    ax.set_xlabel("UTC")
    ax.set_ylabel("Hp (nT)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "goes_g_detector_plot.png", dpi=180)
    plt.close(fig)


def main() -> None:
    try:
        meta = load_metadata()

        primary_df = normalize_mag_json(fetch_json(URL_PRIMARY), "primary")
        secondary_df = normalize_mag_json(fetch_json(URL_SECONDARY), "secondary")

        resolved = choose_east_west(primary_df, secondary_df, meta)
        chosen_trace, detections, chosen_source = pick_trace(resolved["east_df"], resolved["west_df"])

        metadata_summary = {
            "primary_satellite": meta["primary_sat"],
            "secondary_satellite": meta["secondary_sat"],
            "primary_longitude": meta["longitudes"].get(meta["primary_sat"]),
            "secondary_longitude": meta["longitudes"].get(meta["secondary_sat"]),
            "east_resolved_feed": resolved["east_name"],
            "east_satellite": resolved["east_sat"],
            "east_longitude": resolved["east_lon"],
            "west_resolved_feed": resolved["west_name"],
            "west_satellite": resolved["west_sat"],
            "west_longitude": resolved["west_lon"],
            "chosen_source_for_detection": chosen_source,
            "n_detections": len(detections),
        }

        with open(OUTDIR / "metadata_summary.json", "w", encoding="utf-8") as f:
            json.dump(metadata_summary, f, indent=2, default=str)

        save_outputs(chosen_trace, detections, chosen_source, meta)

        print("Done.")
        print(json.dumps(metadata_summary, indent=2, default=str))

    except Exception as exc:
        with open(OUTDIR / "run_error.json", "w", encoding="utf-8") as f:
            json.dump({"error": str(exc)}, f, indent=2)
        print(f"ERROR: {exc}")
        raise


if __name__ == "__main__":
    main()
