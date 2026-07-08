"""
Sewer Flow Data Correction — Streamlit UI (polynomial method).

Run with:  streamlit run app.py
"""

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import flow_engine as fe

st.set_page_config(
    page_title="Flow Data Correction",
    page_icon="🌊",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem;}
      [data-testid="stMetricValue"] {font-size: 1.4rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

SS = st.session_state
SS.setdefault("raw", None)          # parsed raw DataFrame
SS.setdefault("rules", [])          # adjustment rules
SS.setdefault("good_mask", None)    # boolean Series over raw index
SS.setdefault("manual", None)       # DataFrame: manual level/velocity fills

CHART = dict(template="plotly_white", height=430, margin=dict(t=40, b=40))


# ------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🌊 Flow Data Correction")
    st.caption("Polynomial method — area-velocity meter data")

    site = st.text_input("Site name", value="")
    dia = st.number_input(
        "Measured pipe diameter (inches)",
        min_value=1.0, max_value=240.0, value=15.0, step=0.5,
        help="Use the measured diameter, not the nominal one — it drives "
             "every recalculated flow.",
    )
    units = st.selectbox("Flow units", fe.UNIT_CHOICES, index=0)

    st.divider()
    up = st.file_uploader(
        "Import raw meter data (CSV)",
        type=["csv"],
        help="Columns in order: date/time, level (in), velocity (ft/s), "
             "flow, optional rain (in). Header row expected.",
    )
    if up is not None and st.button("Load data", type="primary", width="stretch"):
        try:
            SS.raw = fe.parse_raw_csv(up)
            SS.good_mask = None
            SS.manual = None
            st.success(f"Loaded {len(SS.raw):,} records.")
        except Exception as e:
            st.error(f"Couldn't read that file: {e}")

    if SS.raw is not None:
        st.caption(
            f"**{len(SS.raw):,} records** · "
            f"{SS.raw['timestamp'].min():%Y-%m-%d %H:%M} → "
            f"{SS.raw['timestamp'].max():%Y-%m-%d %H:%M}"
        )
        if st.button("Clear raw data", width="stretch"):
            SS.raw, SS.good_mask, SS.manual = None, None, None
            st.rerun()


if SS.raw is None:
    st.info(
        "**Get started:** enter the pipe diameter in the sidebar, then "
        "import a raw data CSV (date/time, level, velocity, flow, "
        "optional rain). A sample file ships alongside this app."
    )
    st.stop()

raw = SS.raw
adj = fe.apply_adjustments(raw, SS.rules, dia, units)

if SS.good_mask is None or len(SS.good_mask) != len(adj):
    SS.good_mask = fe.default_good_mask(adj, dia)
if SS.manual is None or len(SS.manual) != len(adj):
    SS.manual = pd.DataFrame(
        {"manual_level": np.nan, "manual_velocity": np.nan}, index=adj.index
    )

tab_adj, tab_corr, tab_res = st.tabs(
    ["**1 · Review & adjust**", "**2 · Correct**", "**3 · Results & export**"]
)


# ------------------------------------------------- helpers for charts
def lv_scatter(frame, lcol, vcol, name, color):
    return go.Scattergl(
        x=frame[lcol], y=frame[vcol], mode="markers", name=name,
        marker=dict(size=4, color=color, opacity=0.55),
        customdata=frame.index,
    )


def time_series(frame, ycols, labels, colors, ytitle):
    fig = go.Figure()
    for c, lab, col in zip(ycols, labels, colors):
        fig.add_trace(
            go.Scattergl(
                x=frame["timestamp"], y=frame[c], mode="lines",
                name=lab, line=dict(width=1, color=col),
            )
        )
    fig.update_layout(**CHART, yaxis_title=ytitle, xaxis_title=None)
    return fig


# ============================================================ TAB 1
with tab_adj:
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.subheader("Calibration adjustments")
        st.caption(
            "Additive offsets for sensor drift, plus silt depth at the "
            "invert (silt reduces the flow area). Leave dates blank to "
            "apply to the whole record."
        )
        with st.form("add_rule", clear_on_submit=True):
            c1, c2 = st.columns(2)
            start = c1.text_input("From (optional)", placeholder="2026-01-01 00:00")
            end = c2.text_input("To (optional)", placeholder="2026-01-15 00:00")
            c3, c4, c5 = st.columns(3)
            lvl_off = c3.number_input("Level ± (in)", value=0.0, step=0.05, format="%.2f")
            vel_off = c4.number_input("Velocity ± (ft/s)", value=0.0, step=0.05, format="%.2f")
            silt = c5.number_input("Silt (in)", min_value=0.0, value=0.0, step=0.1, format="%.2f")
            if st.form_submit_button("Add adjustment", width="stretch"):
                try:
                    rule = {
                        "start": pd.Timestamp(start) if start.strip() else None,
                        "end": pd.Timestamp(end) if end.strip() else None,
                        "level": lvl_off, "velocity": vel_off, "silt": silt,
                    }
                    SS.rules.append(rule)
                    SS.good_mask = None  # adjusted data changed → re-seed
                    st.rerun()
                except Exception as e:
                    st.error(f"Bad date: {e}")

        if SS.rules:
            for i, r in enumerate(SS.rules):
                span = (
                    f"{r['start']:%Y-%m-%d %H:%M}" if r["start"] else "start"
                ) + " → " + (f"{r['end']:%Y-%m-%d %H:%M}" if r["end"] else "end")
                cols = st.columns([5, 1])
                cols[0].markdown(
                    f"`{span}` · level {r['level']:+.2f} in · "
                    f"velocity {r['velocity']:+.2f} ft/s · silt {r['silt']:.2f} in"
                )
                if cols[1].button("✕", key=f"del_rule_{i}"):
                    SS.rules.pop(i)
                    SS.good_mask = None
                    st.rerun()
        else:
            st.caption("No adjustments yet — adjusted data equals raw data.")

    with right:
        st.subheader("Raw vs. adjusted")
        which = st.radio(
            "View", ["Level & velocity scattergraph (LV)",
                     "Level & velocity vs. time (LVT)",
                     "Flow vs. time (QRT)"],
            horizontal=True, label_visibility="collapsed",
        )
        if which.startswith("Level & velocity scatter"):
            fig = go.Figure()
            fig.add_trace(lv_scatter(adj, "level_raw", "velocity_raw", "Raw", "#9aa5b1"))
            fig.add_trace(lv_scatter(adj, "level_adj", "velocity_adj", "Adjusted", "#1f6feb"))
            fig.add_vline(x=dia, line_dash="dot", line_color="#c0392b",
                          annotation_text="pipe dia.")
            fig.update_layout(**CHART, xaxis_title="Level (in)",
                              yaxis_title="Velocity (ft/s)")
            st.plotly_chart(fig, width="stretch")
        elif which.startswith("Level & velocity vs"):
            st.plotly_chart(
                time_series(adj,
                            ["level_raw", "level_adj", "velocity_raw", "velocity_adj"],
                            ["Level raw", "Level adj", "Velocity raw", "Velocity adj"],
                            ["#c5ccd4", "#1f6feb", "#e8c3a0", "#d35400"],
                            "Level (in) / Velocity (ft/s)"),
                width="stretch",
            )
        else:
            st.plotly_chart(
                time_series(adj, ["flow_raw", "flow_adj"],
                            [f"Flow raw ({units})", f"Flow adj ({units})"],
                            ["#9aa5b1", "#0f9d58"], f"Flow ({units})"),
                width="stretch",
            )


# ============================================================ TAB 2
with tab_corr:
    good = SS.good_mask
    n_good, n_bad = int(good.sum()), int((~good).sum())

    st.subheader("Build the good-data set")
    st.caption(
        "The correction polynomials V(L) and L(V) are fitted to the good "
        "points only. Remove erroneous points until the fit represents "
        "normal site behavior — lasso/box-select points on the chart, or "
        "use the tolerance band."
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Good points", f"{n_good:,}")
    m2.metric("Removed / bad", f"{n_bad:,}")

    v_of_l, l_of_v = fe.fit_polynomials(
        adj.loc[good, "level_adj"], adj.loc[good, "velocity_adj"]
    )
    if v_of_l is not None:
        a, b, c = v_of_l
        m3.metric("V(L) fit", f"{a:+.3g}·L² {b:+.3g}·L {c:+.3g}")
    else:
        m3.metric("V(L) fit", "n/a")

    band1, band2, band3, band4 = st.columns([1, 1, 1, 2])
    upper = band1.number_input("Upper tolerance (+ft/s)", min_value=0.0,
                               value=1.0, step=0.1)
    lower = band2.number_input("Lower tolerance (−ft/s)", min_value=0.0,
                               value=1.0, step=0.1)

    n_out = 0
    if v_of_l is not None:
        err = fe.velocity_error(
            adj.loc[good, "level_adj"], adj.loc[good, "velocity_adj"], v_of_l
        )
        outlier_idx = adj.loc[good].index[(err > upper) | (err < -lower)]
        n_out = len(outlier_idx)
    band3.metric("Outside band", f"{n_out:,}",
                 delta="Outliers" if n_out else "Clean", delta_color="inverse")

    with band4:
        st.write("")  # vertical alignment
        bc1, bc2 = st.columns(2)
        if bc1.button(f"Remove {n_out} outliers", disabled=n_out == 0,
                      width="stretch"):
            SS.good_mask.loc[outlier_idx] = False
            st.rerun()
        if bc2.button("Reset good data", width="stretch"):
            SS.good_mask = fe.default_good_mask(adj, dia)
            st.rerun()

    # ---- correction scattergraph with selection-to-remove
    fig = go.Figure()
    fig.add_trace(lv_scatter(adj.loc[~good], "level_adj", "velocity_adj",
                             "Removed", "#d0d5da"))
    fig.add_trace(lv_scatter(adj.loc[good], "level_adj", "velocity_adj",
                             "Good", "#1f6feb"))
    if v_of_l is not None:
        Ls = np.linspace(max(0.01, adj.loc[good, "level_adj"].min()),
                         adj.loc[good, "level_adj"].max(), 200)
        Vs = np.polyval(v_of_l, Ls)
        fig.add_trace(go.Scatter(x=Ls, y=Vs, name="V(L)", mode="lines",
                                 line=dict(color="#c0392b", width=2)))
        fig.add_trace(go.Scatter(x=Ls, y=Vs + upper, name="Upper bound",
                                 mode="lines",
                                 line=dict(color="#c0392b", width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=Ls, y=Vs - lower, name="Lower bound",
                                 mode="lines",
                                 line=dict(color="#c0392b", width=1, dash="dot")))
        vg = adj.loc[good, "velocity_adj"]
        Vr = np.linspace(vg.min(), vg.max(), 200)
        fig.add_trace(go.Scatter(x=np.polyval(l_of_v, Vr), y=Vr, name="L(V)",
                                 mode="lines",
                                 line=dict(color="#8e44ad", width=2, dash="dash")))
    fig.update_layout(**CHART, xaxis_title="Level (in)",
                      yaxis_title="Velocity (ft/s)", dragmode="lasso")
    event = st.plotly_chart(fig, width="stretch", on_select="rerun",
                            key="corr_chart",
                            selection_mode=("lasso", "box", "points"))

    sel = [
        p["customdata"] for p in event.selection.get("points", [])
        if p.get("curve_number") == 1  # only points from the "Good" trace
    ]
    if sel:
        if st.button(f"Remove {len(sel)} selected point(s) from good data",
                     type="primary"):
            SS.good_mask.loc[sel] = False
            st.rerun()
    else:
        st.caption("Tip: lasso or box-select good points on the chart to "
                   "remove them from the fit.")


# ============================================================ TAB 3
with tab_res:
    corrected, v_of_l, l_of_v = fe.build_corrected(
        adj, SS.good_mask,
        SS.manual["manual_level"], SS.manual["manual_velocity"],
        dia, units,
    )
    gaps = corrected["level_corr"].isna() | corrected["velocity_corr"].isna()

    st.subheader("Fill remaining gaps (optional)")
    st.caption(
        "Rows removed from the good data are blank. If you trust one of "
        "the two readings, enter it below — the other is reconstructed "
        "from V(L) or L(V) and flow is recalculated. Rows left blank stay "
        "blank in the export."
    )
    if int(gaps.sum()) == 0:
        st.success("No gaps — every record has corrected values.")
    else:
        gap_view = pd.DataFrame({
            "Timestamp": adj.loc[gaps, "timestamp"],
            "Raw level (in)": adj.loc[gaps, "level_raw"],
            "Raw velocity (ft/s)": adj.loc[gaps, "velocity_raw"],
            "Good level (in)": SS.manual.loc[gaps, "manual_level"],
            "Good velocity (ft/s)": SS.manual.loc[gaps, "manual_velocity"],
        })
        edited = st.data_editor(
            gap_view, hide_index=True, width="stretch", height=280,
            disabled=["Timestamp", "Raw level (in)", "Raw velocity (ft/s)"],
            key="gap_editor",
        )
        new_l = pd.to_numeric(edited["Good level (in)"], errors="coerce")
        new_v = pd.to_numeric(edited["Good velocity (ft/s)"], errors="coerce")
        if (not new_l.equals(SS.manual.loc[gaps, "manual_level"])
                or not new_v.equals(SS.manual.loc[gaps, "manual_velocity"])):
            SS.manual.loc[gaps, "manual_level"] = new_l.to_numpy()
            SS.manual.loc[gaps, "manual_velocity"] = new_v.to_numpy()
            st.rerun()

    st.divider()
    st.subheader("Corrected results")

    merged = adj.join(corrected[["level_corr", "velocity_corr", "flow_corr"]])
    view = st.radio(
        "View", ["Flow vs. time (QRT Corr)", "Level & velocity vs. time (LVT Corr)",
                 "Scattergraph (LV Corr)"],
        horizontal=True, label_visibility="collapsed", key="res_view",
    )
    if view.startswith("Flow"):
        st.plotly_chart(
            time_series(merged, ["flow_adj", "flow_corr"],
                        [f"Adjusted ({units})", f"Corrected ({units})"],
                        ["#c5ccd4", "#0f9d58"], f"Flow ({units})"),
            width="stretch",
        )
    elif view.startswith("Level"):
        st.plotly_chart(
            time_series(merged,
                        ["level_corr", "velocity_corr"],
                        ["Level corrected (in)", "Velocity corrected (ft/s)"],
                        ["#1f6feb", "#d35400"], "Level (in) / Velocity (ft/s)"),
            width="stretch",
        )
    else:
        fig = go.Figure()
        fig.add_trace(lv_scatter(merged, "level_adj", "velocity_adj",
                                 "Adjusted", "#c5ccd4"))
        fig.add_trace(lv_scatter(merged, "level_corr", "velocity_corr",
                                 "Corrected", "#0f9d58"))
        fig.update_layout(**CHART, xaxis_title="Level (in)",
                          yaxis_title="Velocity (ft/s)")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Data summary")
    stats = fe.stats_table({
        "Raw": adj[["level_raw", "velocity_raw", "flow_raw", "rain"]].rename(
            columns={"level_raw": "Level (in)", "velocity_raw": "Velocity (ft/s)",
                     "flow_raw": f"Flow ({units})", "rain": "Rain (in)"}),
        "Adjusted": adj[["level_adj", "velocity_adj", "flow_adj"]].rename(
            columns={"level_adj": "Level (in)", "velocity_adj": "Velocity (ft/s)",
                     "flow_adj": f"Flow ({units})"}),
        "Good": adj.loc[SS.good_mask, ["level_adj", "velocity_adj"]].rename(
            columns={"level_adj": "Level (in)", "velocity_adj": "Velocity (ft/s)"}),
        "Corrected": corrected[["level_corr", "velocity_corr", "flow_corr"]].rename(
            columns={"level_corr": "Level (in)", "velocity_corr": "Velocity (ft/s)",
                     "flow_corr": f"Flow ({units})"}),
    })
    st.dataframe(
        stats.style.format({"Max": "{:.3f}", "Min": "{:.3f}",
                            "Average": "{:.3f}", "Std. Dev.": "{:.3f}"},
                           na_rep="—"),
        width="stretch", hide_index=True, height=330,
    )

    # ---- export
    export = merged[["timestamp", "level_raw", "velocity_raw", "flow_raw",
                     "rain", "level_adj", "velocity_adj", "flow_adj",
                     "level_corr", "velocity_corr", "flow_corr"]].copy()
    buf = io.StringIO()
    export.to_csv(buf, index=False)
    st.download_button(
        "Download corrected data (CSV)",
        data=buf.getvalue(),
        file_name=f"{(site or 'site').replace(' ', '_')}_corrected.csv",
        mime="text/csv",
        type="primary",
    )
