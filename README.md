# Sewer Flow Data Correction — Streamlit App

A Python rebuild of the "Polynomial Method" flow data correction spreadsheet
(sanitary sewer area-velocity meter data). The Excel/VBA workflow — import,
calibration adjustment, silt, polynomial regression correction, outlier
removal, scattergraphs, summary statistics — is replicated in two files:

| File | Role |
| --- | --- |
| `flow_engine.py` | Calculation engine: circular-pipe geometry, Q = A·V, V(L) / L(V) polynomial fits, correction pipeline. No UI dependencies — reusable in scripts, notebooks, or a future FastAPI backend. |
| `app.py` | Streamlit UI on top of the engine. |
| `sample_data.csv` | 4 weeks of synthetic 15-min meter data (15" pipe) with a velocity-sensor fouling episode, level spikes, and dead readings — for trying the workflow. |

## Setup

```bash
pip install streamlit plotly pandas numpy
streamlit run app.py
```

Opens at http://localhost:8501. To share on a LAN:
`streamlit run app.py --server.address 0.0.0.0`

## Workflow (mirrors the spreadsheet)

1. **Sidebar** — enter the *measured* pipe diameter, site name, and flow
   units (MGD / gpm / cfs), then import the raw CSV. Expected columns, in
   order: date/time, level (in), velocity (ft/s), flow, optional rain (in).
2. **Tab 1 · Review & adjust** — add additive calibration offsets for level
   and velocity (optionally limited to a date range, for drift) and silt
   depth. Flow is recalculated from adjusted level/velocity and reduced
   silt area. Compare raw vs. adjusted on the LV / LVT / QRT charts.
3. **Tab 2 · Correct** — the good-data set seeds automatically (drops
   level ≤ 0, level > diameter, velocity ≤ 0). The V(L) and L(V)
   2nd-degree polynomials refit live as you remove points, two ways:
   - **Tolerance band**: set upper/lower velocity tolerances around V(L)
     and click *Remove outliers* (repeat — the fit changes each pass,
     just like the spreadsheet warns).
   - **Lasso/box-select** points directly on the scattergraph and remove
     them — the interactive replacement for sort-and-delete-rows.
4. **Tab 3 · Results & export** — rows removed from the good set are gaps.
   Where you trust one reading, type it into the gap table; the other is
   reconstructed from V(L) or L(V) and flow recalculated. Review the
   corrected charts and summary statistics, then download the full
   raw/adjusted/corrected CSV.

## Notes

- Silt is modeled as a segment at the invert: the water surface sits at
  (silt + level) and the silt segment's area is subtracted from the flow
  area, matching the spreadsheet's qflow/qsilt approach.
- Reset good data at any time from Tab 2; adding or removing an
  adjustment rule also re-seeds it (the adjusted values it was built on
  changed).
- The engine is deliberately UI-free, so promoting this to a multi-user
  FastAPI + database app later means writing a new front end, not new math.
