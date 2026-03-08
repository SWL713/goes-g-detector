from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

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
SESSION.headers.update({"User-Agent": "goes-g-detector/0.1"})


@dataclass
class DetectionResult:
    timestamp: pd.Timestamp
    source: str
    score: float
    note: str


def fetch_json(url: str):
    r = SESSION.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def load_metadata():
    sources = fetch_json(URL_SOURCES)
    longitudes = fetch_json(URL_LONGITUDES)

    if isinstance(sources, list):
        sources = sources[-1]

    lon_map = {}
    for row in longitudes:
        sat = row.get("satellite")
        lon = row.get("longitude")
        if sat is not None:
            lon_map[int(sat)] = lon

    mag_sources = sources.get("magnetometers", {})
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

    # Robust time parsing
    time_col = None
    for c in df.columns:
        if c.lower() in {"time_tag", "time", "date"} or "time" in c.lower():
            time_col = c
            break
    if time_col is None:
        raise ValueError(f"No time column found in {source_name}: {list(df.columns)}")

    df["time_utc"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=["time_utc"]).copy()

    # Try common field names
    rename = {}
    for c in df.columns:
        low = c.lower()
        if low in {"hp", "h_p"}:
            rename[c] = "Hp"
        elif low in {"he", "h_e"}:
            rename[c] = "He"
        elif low in {"hn", "h_n"}:
            rename[c] = "Hn"
        elif low in {"ht", "bt", "total", "total_field"}:
            rename[c] = "Ht"
        elif low == "satellite":
            rename[c] = "satellite"

    df = df.rename(columns=rename)

    keep = ["time_utc"] + [c for c in ["Hp", "He", "Hn", "Ht", "satellite"] if c in df.columns]
    df = df[keep].copy()
    df["source_feed"] = source_name
    df = df.sort_values("time_utc").drop_duplicates(subset=["time_utc"])

    return df


def choose_east_west(primary_df: pd.DataFrame, secondary_df: pd.DataFrame, meta: dict):
    lon_map = meta["longitudes"]
    p_sat = meta["primary_sat"]
    s_sat = meta["secondary_sat"]

    p_lon = lon_map.get(p_sat)
    s_lon = lon_map.get(s_sat)

    # Smaller west longitude magnitude corresponds to East (e.g. ~75 W vs ~137 W)
    # Values are provided as positive west longitudes in SWPC metadata.
    candidates = [
        ("primary", primary_df, p_sat, p_lon),
        ("secondary", secondary_df, s_sat, s_lon),
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
        raise ValueError("This starter script expects an Hp column in the GOES JSON")

    out["Hp"] = pd.to_numeric(out["Hp"], errors="coerce")
    out = out.dropna(subset=["Hp"]).copy()

    # Regularize to 1-minute cadence
    out = out.set_index("time_utc").resample("1min").median().interpolate(limit=5).reset_index()

    # Smooth + detrend
    out["hp_smooth_5"] = out["Hp"].rolling(5, center=True, min_periods=1).median()
    out["hp_baseline_180"] = out["hp_smooth_5"].rolling(180, center=True, min_periods=30).median()
    out["hp_resid"] = out["hp_smooth_5"] - out["hp_baseline_180"]
    out["d1"] = out["hp_smooth_5"].diff()

    return out


def detect_g_candidates(df: pd.DataFrame, source_label: str) -> List[DetectionResult]:
    """
    Very simple ruleset:
    - find local positive peaks in the residual as prior release peaks
    - cluster nearby peaks; keep the final one in each cluster
    - after each final peak, look for the earliest minute where a sustained decline begins
    """
    work = df.copy().reset_index(drop=True)

    y = work["hp_resid"].to_numpy()
    if len(y) < 120:
        return []

    # Candidate release peaks
    peak_idx, props = find_peaks(y, prominence=3.0, distance=20)

    if len(peak_idx) == 0:
        return []

    # Cluster nearby peaks and keep the last one to handle double peaks
    clusters = []
    current = [int(peak_idx[0])]
    for idx in peak_idx[1:]:
        idx = int(idx)
        if idx - current[-1] <= 25:
            current.append(idx)
        else:
            clusters.append(current)
            current = [idx]
    clusters.append(current)

    final_peaks = [c[-1] for c in clusters]

    detections: List[DetectionResult] = []

    for p in final_peaks:
        # Search 5 to 90 min after final peak for start of sustained charging
        start_search = p + 5
        end_search = min(p + 90, len(work) - 61)
        if start_search >= end_search:
            continue

        best_idx: Optional[int] = None
        best_score = -np.inf

        for i in range(start_search, end_search):
            future_45 = work.loc[i + 45, "hp_smooth_5"] - work.loc[i, "hp_smooth_5"]
            future_20 = work.loc[i + 20, "hp_smooth_5"] - work.loc[i, "hp_smooth_5"]
            neg_frac_20 = (work.loc[i + 1:i + 20, "d1"] < 0).mean()
            prev_slope_10 = work.loc[i, "hp_smooth_5"] - work.loc[max(0, i - 10), "hp_smooth_5"]

            # Heuristic score: want downward future motion, lots of negative steps,
            # and not already deep into the fall.
            score = (
                (-future_45) * 1.0
                + (-future_20) * 0.6
                + neg_frac_20 * 8.0
                - abs(prev_slope_10) * 0.25
            )

            if future_45 <= -6.0 and future_20 <= -2.0 and neg_frac_20 >= 0.60:
                if score > best_score:
                    best_score = score
                    best_idx = i

        if best_idx is not None:
            detections.append(
                DetectionResult(
                    timestamp=work.loc[best_idx, "time_utc"],
                    source=source_label,
                    score=float(best_score),
                    note="after final release peak",
                )
            )

    # Suppress candidates too close together; keep strongest in 90 min window
    detections = sorted(detections, key=lambda d: d.timestamp)
    filtered: List[DetectionResult] = []
    for d in detections:
        if not filtered:
            filtered.append(d)
            continue
        dt_min = (d.timestamp - filtered[-1].timestamp).total_seconds() / 60.0
        if dt_min < 90:
            if d.score > filtered[-1].score:
                filtered[-1] = d
        else:
            filtered.append(d)

    return filtered


def pick_trace(east_df: pd.DataFrame, west_df: pd.DataFrame):
    east_candidates = detect_g_candidates(prep_trace(east_df), "GOES-East")
    if len(east_candidates) >= 1:
        return prep_trace(east_df), east_candidates, "GOES-East"

    west_candidates = detect_g_candidates(prep_trace(west_df), "GOES-West fallback")
    return prep_trace(west_df), west_candidates, "GOES-West fallback"


def save_outputs(trace_df: pd.DataFrame, detections: List[DetectionResult], chosen_source: str, meta: dict):
    # Save cleaned trace
    trace_df.to_csv(OUTDIR / "goes_trace_used.csv", index=False)

    # Save detections
    det_df = pd.DataFrame(
        [
            {
                "g_time_utc": d.timestamp.isoformat(),
                "source": d.source,
                "score": d.score,
                "note": d.note,
            }
            for d in detections
        ]
    )
    det_df.to_csv(OUTDIR / "g_candidates.csv", index=False)

    # Plot
    fig = plt.figure(figsize=(16, 7))
    ax = fig.add_subplot(111)

    ax.plot(trace_df["time_utc"], trace_df["Hp"], label="Hp raw", linewidth=0.8)
    ax.plot(trace_df["time_utc"], trace_df["hp_smooth_5"], label="Hp smooth", linewidth=1.5)

    for d in detections:
        ax.axvline(d.timestamp, linewidth=1.2)
        ax.text(
            d.timestamp,
            trace_df["hp_smooth_5"].max(),
            d.timestamp.strftime("%m-%d %H:%M"),
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
        )

    title = (
        f"7-day GOES G detector ({chosen_source}) | "
        f"primary sat={meta['primary_sat']} secondary sat={meta['secondary_sat']}"
    )
    ax.set_title(title)
    ax.set_xlabel("UTC")
    ax.set_ylabel("Hp (nT)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "goes_g_detector_plot.png", dpi=180)
    plt.close(fig)


def main():
    meta = load_metadata()

    primary_df = normalize_mag_json(fetch_json(URL_PRIMARY), "primary")
    secondary_df = normalize_mag_json(fetch_json(URL_SECONDARY), "secondary")

    resolved = choose_east_west(primary_df, secondary_df, meta)
    chosen_trace, detections, chosen_source = pick_trace(resolved["east_df"], resolved["west_df"])

    # Save metadata summary
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


if __name__ == "__main__":
    main()
