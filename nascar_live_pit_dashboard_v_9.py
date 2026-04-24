import time
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
import glob

files = glob.glob("nascar-data/cup/*/*.parquet")
latest_file = max(files)

hist = pd.read_parquet(latest_file)

st.markdown("""
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 1.2rem; }
[data-testid="stMetricValue"] { font-size: 1.45rem; }
.banner { padding: 0.85rem 1rem; border-radius: 0.75rem; background: #111827; margin-bottom: 1rem; }
.banner-title { font-size: 1.2rem; font-weight: 700; color: white; }
.banner-subtitle { font-size: 0.95rem; color: #d1d5db; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=5)
def fetch_data():
    r = requests.get(LIVE_FEED_URL, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def safe_int(x):
    try: return int(x) if x not in [None,""] else None
    except: return None


def safe_float(x):
    try: return float(x) if x not in [None,""] else None
    except: return None


def flag_label(v):
    return {1:"Green",2:"Yellow",3:"Red",4:"Checkered",5:"White"}.get(safe_int(v),"—")


def classify_track(track_name: Any) -> str:
    name = str(track_name or "").lower()
    if any(k in name for k in ["daytona", "talladega", "atlanta motor speedway"]):
        return "Superspeedway"
    if any(k in name for k in ["martinsville", "bristol", "richmond", "bowman gray", "north wilkesboro", "clash"]):
        return "Short Track"
    if any(k in name for k in ["road course", "roval", "sonoma", "watkins glen", "cota", "chicago street", "indy road"]):
        return "Road Course"
    return "Speedway"


def track_curve_params(track_name: Any):
    track_type = classify_track(track_name)
    if track_type == "Short Track":
        return {"track_type": track_type, "fresh_laps": 4, "mid_laps": 18, "penalty_1": 0.95, "penalty_2": 0.40, "falloff_rate": 0.030, "falloff_cap": 0.85}
    if track_type == "Superspeedway":
        return {"track_type": track_type, "fresh_laps": 2, "mid_laps": 10, "penalty_1": 0.35, "penalty_2": 0.12, "falloff_rate": 0.006, "falloff_cap": 0.12}
    if track_type == "Road Course":
        return {"track_type": track_type, "fresh_laps": 3, "mid_laps": 12, "penalty_1": 1.35, "penalty_2": 0.65, "falloff_rate": 0.018, "falloff_cap": 0.55}
    return {"track_type": track_type, "fresh_laps": 3, "mid_laps": 15, "penalty_1": 1.20, "penalty_2": 0.50, "falloff_rate": 0.018, "falloff_cap": 0.45}


def normalize(data):
    rows=[]
    ts=datetime.utcnow()

    for v in data.get("vehicles",[]):
        pits=v.get("pit_stops",[])
        last_pit=pits[-1] if pits else {}

        rows.append({
            "ts":ts,
            "lap":safe_int(data.get("lap_number")),
            "flag_state":safe_int(data.get("flag_state")),
            "track_name":data.get("track_name"),
            "run_name":data.get("run_name"),
            "vehicle":str(v.get("vehicle_number")),
            "driver":(v.get("driver") or {}).get("full_name"),
            "pos":safe_int(v.get("running_position")),
            "lap_time":safe_float(v.get("last_lap_time")),
            "pit_count":len(pits),
            "last_pit":safe_int(last_pit.get("lap"))
        })

    return pd.DataFrame(rows)


def enrich(df, hist):
    hist = hist.sort_values(["vehicle","lap","ts"]).drop_duplicates(["vehicle","lap"], keep="last").copy()
    hist["pit_event"] = (
        hist.groupby("vehicle")["pit_count"]
        .diff()
        .fillna(0)
        .clip(lower=0)
    )
    hist["stint_id"] = hist.groupby("vehicle")["pit_event"].cumsum()
    hist["lsp"] = hist.groupby(["vehicle","stint_id"]).cumcount()

    curve = track_curve_params(df.iloc[0]["track_name"] if not df.empty else None)

    pred_rows=[]
    for vehicle,g in hist.groupby("vehicle"):
        g=g.sort_values("lap")
        green=g[g["flag_state"]==GREEN_FLAG_STATE]
        laps=green["lap_time"].dropna().tail(8)
        base=laps.median() if len(laps)>=3 else np.nan
        lsp=g.iloc[-1]["lsp"]

        if pd.isna(lsp):
            penalty = 0.0
            tire_falloff = 0.0
        else:
            penalty = curve["penalty_1"] if lsp <= 1 else curve["penalty_2"] if lsp == 2 else 0.0
            tire_falloff = min(max(lsp - curve["fresh_laps"], 0) * curve["falloff_rate"], curve["falloff_cap"])

        pred_rows.append({
            "vehicle":vehicle,
            "pred":base + penalty + tire_falloff if pd.notna(base) else np.nan
        })

    preds=pd.DataFrame(pred_rows)
    out=df.merge(preds,on="vehicle",how="left")

    latest_lsp=(
        hist.sort_values(["vehicle","lap","ts"])
        .groupby("vehicle",as_index=False)
        .tail(1)[["vehicle","lsp"]]
    )
    out=out.merge(latest_lsp,on="vehicle",how="left")

    out["delta"] = out["lap_time"] - out["pred"]

    def tire_phase(r):
        lsp=r["lsp"]
        pit_count=safe_int(r.get("pit_count")) or 0
        if pd.isna(lsp): return "—"
        if pit_count==0:
            if lsp<=curve["fresh_laps"]: return "Start Run"
            elif lsp<=curve["mid_laps"]: return "Opening Stint"
            else: return "Long Opening Stint"
        if lsp<=curve["fresh_laps"]: return "Fresh"
        elif lsp<=curve["mid_laps"]: return "Mid"
        else: return "Falloff"

    out["tire"] = out.apply(tire_phase, axis=1)
    out["track_type"] = curve["track_type"]
    return out


def format_table(df):
    return pd.DataFrame({
        "Pos":df["pos"],
        "#":df["vehicle"],
        "Driver":df["driver"],
        "Last":df["lap_time"].map(lambda x:f"{x:.3f}" if pd.notna(x) else "—"),
        "Next":df["pred"].map(lambda x:f"{x:.3f}" if pd.notna(x) else "—"),
        "Δ":df["delta"].map(lambda x:f"{x:+.3f}" if pd.notna(x) else "—"),
        "Tire":df["tire"]
    })


def render_banner(df):
    if df.empty: return
    r=df.iloc[0]
    track_type = r.get("track_type") or classify_track(r.get("track_name"))
    st.markdown(f"""
    <div class='banner'>
        <div class='banner-title'>🏁 {r.get('track_name')}</div>
        <div class='banner-subtitle'>Session: {r.get('run_name')} • Track Type: {track_type} • Flag: {flag_label(r.get('flag_state'))} • Lap: {r.get('lap')}</div>
    </div>
    """,unsafe_allow_html=True)


st.title("🏎️ NASCAR Race Intelligence")

if "hist" not in st.session_state:
    st.session_state.hist=[]

try:
    data=fetch_data()
    snap=normalize(data)
    st.session_state.hist.extend(snap.to_dict("records"))
except Exception as e:
    st.error(e)
    st.stop()

hist=pd.DataFrame(st.session_state.hist)

table=enrich(snap,hist)

render_banner(table)

sort_mode = st.radio("View Mode", ["Race Order", "Attack"], horizontal=True)

if sort_mode == "Attack":
    table = table.sort_values("delta", na_position="last")
else:
    table = table.sort_values("pos", na_position="last")

leader=table.iloc[0]
c1,c2,c3,c4=st.columns(4)
c1.metric("Leader",leader["vehicle"])
c2.metric("Lap",leader["lap"])
c3.metric("Flag",flag_label(leader["flag_state"]))
c4.metric("Fastest",f"{table['lap_time'].min():.3f}")

st.dataframe(format_table(table),use_container_width=True,hide_index=True)

st.rerun()
