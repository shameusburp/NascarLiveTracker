import time
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="NASCAR Race Intelligence v12", layout="wide")

LIVE_FEED_URL = "https://cf.nascar.com/live/feeds/live-feed.json"
REQUEST_TIMEOUT = 10
GREEN_FLAG_STATE = 1

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
    hist = hist.sort_values(["vehicle","lap","ts"]).drop_duplicates(["vehicle","lap"])
    hist["last_pit"] = hist.groupby("vehicle")["last_pit"].ffill()
    hist["stint_start_lap"] = hist["last_pit"].fillna(0)
    hist["lsp"] = hist["lap"] - hist["stint_start_lap"]

    pred_rows=[]
    for vehicle,g in hist.groupby("vehicle"):
        g=g.sort_values("lap")
        green=g[g["flag_state"]==GREEN_FLAG_STATE]
        laps=green["lap_time"].dropna().tail(8)
        base=laps.median() if len(laps)>=3 else np.nan
        lsp=g.iloc[-1]["lsp"]
        penalty=1.2 if lsp<=1 else 0.5 if lsp==2 else 0

        pred_rows.append({
            "vehicle":vehicle,
            "pred":base+penalty if pd.notna(base) else np.nan,
            "lsp":lsp
        })

    preds=pd.DataFrame(pred_rows)
    out=df.merge(preds,on="vehicle",how="left")

    out["delta"] = out["lap_time"] - out["pred"]

    def tire_phase(r):
        lsp = r["lsp"]
        pit_count = safe_int(r.get("pit_count")) or 0
        if pd.isna(lsp):
            return "—"
        if pit_count == 0:
            if lsp <= 3:
                return "Start Run"
            elif lsp <= 15:
                return "Opening Stint"
            else:
                return "Long Opening Stint"
        if lsp <= 3:
            return "Fresh"
        elif lsp <= 15:
            return "Mid"
        else:
            return "Falloff"

    out["tire"] = out.apply(tire_phase, axis=1)

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
    st.markdown(f"""
    <div class='banner'>
        <div class='banner-title'>🏁 {r.get('track_name')}</div>
        <div class='banner-subtitle'>Session: {r.get('run_name')} • Flag: {flag_label(r.get('flag_state'))} • Lap: {r.get('lap')}</div>
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

render_banner(snap)

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

display_table = format_table(table)
st.dataframe(display_table,use_container_width=True,hide_index=True)

st.rerun()
