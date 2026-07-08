"""
Hydraulics + regression engine for sanitary sewer flow data correction.

Replicates the calculation logic of the Woolpert "Polynomial Method"
correction spreadsheet (v7, M. Kirby, PE):

  * Circular-pipe segment geometry to compute flow area from depth,
    with silt occupying the pipe invert (silt reduces flow area).
  * Q = A * V continuity, with unit conversion to MGD / gpm / cfs.
  * 2nd-degree polynomial regression of V(L) and L(V) fitted to the
    "good" data points, used to reconstruct erroneous readings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- units
CFS_TO = {
    "MGD": 0.6463169,
    "gpm": 448.8312,
    "cfs": 1.0,
}
UNIT_CHOICES = list(CFS_TO.keys())


# ----------------------------------------------------------- geometry
def segment_area_ft2(depth_in, dia_in: float):
    """Area (ft^2) of a circular segment for water depth measured from
    the pipe invert. depth_in and dia_in are in inches; depth is clipped
    to [0, dia]. Vectorized over depth_in."""
    d = np.clip(np.asarray(depth_in, dtype=float), 0.0, dia_in)
    r_ft = dia_in / 24.0  # radius in feet (dia inches / 12 / 2)
    with np.errstate(invalid="ignore"):
        theta = 2.0 * np.arccos(1.0 - 2.0 * d / dia_in)
    return (r_ft**2) / 2.0 * (theta - np.sin(theta))


def flow_area_ft2(level_in, silt_in, dia_in: float):
    """Flow area (ft^2) given a level reading (depth of water above the
    silt surface) and a silt depth at the invert. The water surface sits
    at silt + level above the invert; the silt segment is subtracted —
    the spreadsheet's qflow / qsilt columns."""
    level = np.asarray(level_in, dtype=float)
    silt = np.asarray(silt_in, dtype=float)
    surface = np.clip(silt + level, 0.0, dia_in)
    area = segment_area_ft2(surface, dia_in) - segment_area_ft2(silt, dia_in)
    return np.clip(area, 0.0, None)


def compute_flow(level_in, velocity_fps, silt_in, dia_in: float, units: str):
    """Q = A * V. Level/silt in inches, velocity ft/s, result in `units`."""
    area = flow_area_ft2(level_in, silt_in, dia_in)
    q_cfs = area * np.asarray(velocity_fps, dtype=float)
    return q_cfs * CFS_TO[units]


# --------------------------------------------------------- regression
def fit_polynomials(level: pd.Series, velocity: pd.Series):
    """Fit 2nd-degree polynomials V(L) and L(V) on good data points.
    Returns (v_of_l, l_of_v) as numpy coefficient arrays, or (None, None)
    if there aren't enough distinct points to fit."""
    mask = level.notna() & velocity.notna()
    L = level[mask].to_numpy(dtype=float)
    V = velocity[mask].to_numpy(dtype=float)
    if len(L) < 3 or np.ptp(L) == 0 or np.ptp(V) == 0:
        return None, None
    v_of_l = np.polyfit(L, V, 2)
    l_of_v = np.polyfit(V, L, 2)
    return v_of_l, l_of_v


def velocity_error(level, velocity, v_of_l):
    """Actual velocity minus V(L) prediction (spreadsheet column H)."""
    return np.asarray(velocity, dtype=float) - np.polyval(
        v_of_l, np.asarray(level, dtype=float)
    )


# ------------------------------------------------------------ pipeline
RAW_COLS = ["timestamp", "level_raw", "velocity_raw", "flow_raw", "rain"]


def parse_raw_csv(file_like) -> pd.DataFrame:
    """Read meter-export CSV: col 1 datetime, col 2 level (in),
    col 3 velocity (ft/s), col 4 flow, optional col 5 rain (in)."""
    df = pd.read_csv(file_like)
    if df.shape[1] < 4:
        raise ValueError(
            f"Need at least 4 columns (date/time, level, velocity, flow); "
            f"got {df.shape[1]}."
        )
    df = df.iloc[:, :5].copy()
    names = RAW_COLS[: df.shape[1]]
    df.columns = names
    if "rain" not in df.columns:
        df["rain"] = np.nan
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for c in ["level_raw", "velocity_raw", "flow_raw", "rain"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    return df.reset_index(drop=True)


def apply_adjustments(
    df: pd.DataFrame, rules: list[dict], dia_in: float, units: str
) -> pd.DataFrame:
    """Apply additive level/velocity offsets and silt depth per
    date-range rule (the spreadsheet's blue 'Adjust By' columns),
    then recalculate flow from the adjusted values."""
    out = df.copy()
    out["level_adj_by"] = 0.0
    out["vel_adj_by"] = 0.0
    out["silt"] = 0.0
    for r in rules:
        m = pd.Series(True, index=out.index)
        if r.get("start") is not None:
            m &= out["timestamp"] >= pd.Timestamp(r["start"])
        if r.get("end") is not None:
            m &= out["timestamp"] <= pd.Timestamp(r["end"])
        out.loc[m, "level_adj_by"] += float(r.get("level", 0) or 0)
        out.loc[m, "vel_adj_by"] += float(r.get("velocity", 0) or 0)
        out.loc[m, "silt"] += float(r.get("silt", 0) or 0)
    out["level_adj"] = out["level_raw"] + out["level_adj_by"]
    out["velocity_adj"] = out["velocity_raw"] + out["vel_adj_by"]
    out["flow_adj"] = compute_flow(
        out["level_adj"], out["velocity_adj"], out["silt"], dia_in, units
    )
    return out


def default_good_mask(df: pd.DataFrame, dia_in: float) -> pd.Series:
    """Physically plausible points: 0 < level <= pipe diameter and
    velocity > 0 (the spreadsheet's recommended first sort-and-delete)."""
    return (
        df["level_adj"].gt(0)
        & df["level_adj"].le(dia_in)
        & df["velocity_adj"].gt(0)
    )


def build_corrected(
    df: pd.DataFrame,
    good_mask: pd.Series,
    manual_level: pd.Series,
    manual_velocity: pd.Series,
    dia_in: float,
    units: str,
):
    """Assemble the corrected dataset (the spreadsheet's 'Good Data' →
    'Corrected' columns):

      * good points keep their adjusted level & velocity;
      * for bad points, the user may supply level OR velocity manually —
        the missing one is reconstructed via V(L) or L(V);
      * flow is recalculated from corrected level & velocity.

    Returns (corrected_df, v_of_l, l_of_v).
    """
    good_L = df.loc[good_mask, "level_adj"]
    good_V = df.loc[good_mask, "velocity_adj"]
    v_of_l, l_of_v = fit_polynomials(good_L, good_V)

    lvl = pd.Series(np.nan, index=df.index, dtype=float)
    vel = pd.Series(np.nan, index=df.index, dtype=float)
    lvl[good_mask] = df.loc[good_mask, "level_adj"]
    vel[good_mask] = df.loc[good_mask, "velocity_adj"]

    # manual entries only apply to non-good rows
    bad = ~good_mask
    lvl[bad] = manual_level.reindex(df.index)[bad]
    vel[bad] = manual_velocity.reindex(df.index)[bad]

    if v_of_l is not None:
        need_v = lvl.notna() & vel.isna()
        vel[need_v] = np.polyval(v_of_l, lvl[need_v].to_numpy(dtype=float))
        need_l = vel.notna() & lvl.isna()
        lvl[need_l] = np.polyval(l_of_v, vel[need_l].to_numpy(dtype=float))

    out = df[["timestamp", "rain", "silt"]].copy()
    out["level_corr"] = lvl
    out["velocity_corr"] = vel
    out["flow_corr"] = np.where(
        lvl.notna() & vel.notna(),
        compute_flow(lvl, vel, out["silt"], dia_in, units),
        np.nan,
    )
    return out, v_of_l, l_of_v


def stats_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Summary-sheet style statistics: max/min/count/mean/std per column
    for each named dataset."""
    rows = []
    for name, sub in frames.items():
        for col in sub.columns:
            s = pd.to_numeric(sub[col], errors="coerce").dropna()
            rows.append(
                {
                    "Dataset": name,
                    "Series": col,
                    "Max": s.max() if len(s) else np.nan,
                    "Min": s.min() if len(s) else np.nan,
                    "Count": int(len(s)),
                    "Average": s.mean() if len(s) else np.nan,
                    "Std. Dev.": s.std() if len(s) else np.nan,
                }
            )
    return pd.DataFrame(rows)
