import time
from datetime import datetime
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="NASCAR Race Intelligence v9", layout="wide")

LIVE_FEED_URL = "https://cf.nascar.com/live/feeds/live-feed.json"
REQUEST_TIMEOUT = 10

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; padding-bottom: 1.2rem; }
    [data-testid="stMetricValue"] { font-size: 1.45rem; }
    .banner {
        padding: 0.85rem 1rem;
        border-radius: 0.75rem;
        background: linear-gradient(90deg, #111827 0%, #1f2937 100%);
        border: 1px solid #374151;
        margin-bottom: 1rem;
    }
    .banner-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.2rem;
    }
    .banner-subtitle {
        font-size: 0.95rem;
        color: #d1d5db;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=5)
def fetch_data() -> dict:
    response = requests.get(
        LIVE_FEED_URL,
        timeout=REQUEST_TIMEOUT,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json,text/plain,*/*",
            "Referer": "https://www.nascar.com/",
        },
    )
    response.raise_for_status()
    return response.json()


def safe_int(value: Any):
    try:
        if value in [None, ""]:
            return None
        return int(value)
    except Exception:
        return None


def safe_float(value: Any):
    try:
        if value in [None, ""]:
            return None
        return float(value)
    except Exception:
        return None


def safe_str(value: Any):
    if value in [None, ""]:
        return None
    return str(value)


def normalize(data: dict) -> pd.DataFrame:
    rows = []
    ts = datetime.utcnow()

    vehicles = data.get("vehicles", [])
    if not vehicles and isinstance(data.get("series"), list):
        for block in data["series"]:
            if isinstance(block, dict) and isinstance(block.get("vehicles"), list) and block.get("vehicles"):
                data = {**data, **block}
                vehicles = data.get("vehicles", [])
                break

    track_name = safe_str(data.get("track_name"))
    run_name = safe_str(data.get("run_name"))
    race_id = safe_int(data.get("race_id"))
    series_id = safe_int(data.get("series_id"))
    flag_state = safe_int(data.get("flag_state"))
    lap_number = safe_int(data.get("lap_number"))

    for v in vehicles:
        driver = v.get("driver") or {}
        pits = v.get("pit_stops") or []
        last_pit = pits[-1] if pits and isinstance(pits[-1], dict) else {}

        last_pit_lap = None
        for key in ["lap", "lap_number", "pit_lap", "pit_in_lap"]:
            last_pit_lap = safe_int(last_pit.get(key))
            if last_pit_lap is not None:
                break

        rows.append(
            {
                "ts": ts,
                "snapshot_date": ts.strftime("%Y-%m-%d"),
                "snapshot_time_utc": ts.strftime("%H:%M:%S UTC"),
                "lap": lap_number,
                "flag_state": flag_state,
                "series_id": series_id,
                "race_id": race_id,
                "track_name": track_name,
                "run_name": run_name,
                "vehicle": str(v.get("vehicle_number")),
                "driver": driver.get("full_name") or f"#{v.get('vehicle_number', '?')}",
                "pos": safe_int(v.get("running_position")),
                "lap_time": safe_float(v.get("last_lap_time")),
                "last_lap_speed": safe_float(v.get("last_lap_speed")),
                "delta": safe_float(v.get("delta")),
                "pit_count": len(pits),
                "last_pit": last_pit_lap,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["pos", "vehicle"], na_position="last").reset_index(drop=True)
    return df


def add_hist(df: pd.DataFrame) -> None:
    if "hist" not in st.session_state:
        st.session_state.hist = []
    st.session_state.hist.extend(df.to_dict("records"))
    st.session_state.hist = st.session_state.hist[-50000:]


def get_hist() -> pd.DataFrame:
    return pd.DataFrame(st.session_state.hist) if "hist" in st.session_state and st.session_state.hist else pd.DataFrame()


def enrich(df: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    if hist.empty:
        hist = df.copy()

    hist = hist.sort_values(["vehicle", "lap", "ts"]).drop_duplicates(["vehicle", "lap"], keep="last")
    hist["last_pit"] = hist.groupby("vehicle")["last_pit"].ffill()
    hist["lsp"] = hist["lap"] - hist["last_pit"]
    hist.loc[hist["last_pit"].isna(), "lsp"] = np.nan

    def pred(g: pd.DataFrame) -> pd.Series:
        laps = g["lap_time"].dropna().tail(8)
        base = float(laps.median()) if len(laps) >= 3 else np.nan
        cur = g.iloc[-1]
        lsp = cur.get("lsp")

        penalty = 0.0
        if pd.notna(lsp):
            if lsp <= 1:
                penalty = 1.2
            elif lsp == 2:
                penalty = 0.5

        return pd.Series(
            {
                "vehicle": str(g.name),
                "pred": base + penalty if pd.notna(base) else np.nan,
                "lsp": lsp,
            }
        )

    preds = hist.groupby("vehicle", group_keys=False).apply(pred).reset_index(drop=True)
    out = df.merge(preds, on="vehicle", how="left")
    out["delta_pred"] = out["lap_time"] - out["pred"]

    med = out["lap_time"].median()
    out["strategy"] = ""
    out.loc[(out["lsp"] <= 3) & (out["lap_time"] < med - 0.15), "strategy"] = "Undercut"
    out.loc[(out["lsp"] >= 15), "strategy"] = "Overcut"
    out.loc[out["delta_pred"] < -0.2, "strategy"] = "Gaining"

    def tire_phase(x: Any) -> str:
        if pd.isna(x):
            return "—"
        if x <= 3:
            return "Fresh"
        if x <= 15:
            return "Mid"
        return "Falloff"

    out["tire"] = out["lsp"].apply(tire_phase)
    out["pace"] = out["pred"].fillna(out["lap_time"])
    out["pace_rank"] = out["pace"].rank(method="min")

    max_pos = out["pos"].max() if out["pos"].notna().any() else 1
    max_pace = out["pace"].max() if out["pace"].notna().any() else 1
    out["score"] = (1 - out["pos"] / max_pos) + (1 - out["pace"] / max_pace)
    score_sum = out["score"].sum()
    out["win"] = np.where(score_sum > 0, out["score"] / score_sum * 100, 0)

    return out


def simulate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    pit_loss = 18.0
    spread = 0.35
    results = []
    field_size = len(df)

    for _, r in df.iterrows():
        pos = safe_int(r.get("pos")) or field_size
        pace = safe_float(r.get("pace"))
        lsp = safe_float(r.get("lsp"))

        rejoin = min(pos + int(pit_loss / spread), field_size)

        gain = 0
        ahead = df[df["pos"] < pos].head(5)
        for _, a in ahead.iterrows():
            a_pace = safe_float(a.get("pace"))
            if pace is not None and a_pace is not None and pace < a_pace - 0.15:
                gain += 1

        pit_pos = max(rejoin - gain, 1)

        falloff_penalty = 0.0
        if lsp is not None and not pd.isna(lsp):
            falloff_penalty = max(lsp - 20, 0) * 0.05

        loss = int(falloff_penalty / spread) if falloff_penalty > 0 else 0
        stay_pos = min(pos + loss, field_size)

        results.append(
            {
                "vehicle": r["vehicle"],
                "pit_pos": pit_pos,
                "gain": pos - pit_pos,
                "stay_pos": stay_pos,
                "loss": stay_pos - pos,
            }
        )

    sim = pd.DataFrame(results)
    out = df.merge(sim, on="vehicle", how="left")
    out["call"] = np.where(out["gain"] > out["loss"], "PIT", np.where(out["loss"] > out["gain"], "STAY", "HOLD"))
    return out


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    strat = {"Undercut": "🟢", "Overcut": "🟠", "Gaining": "🔵", "": ""}
    tire = {"Fresh": "🟢", "Mid": "🟠", "Falloff": "🔴", "—": "⚪"}
    call = {"PIT": "🟢 PIT", "STAY": "🟠 STAY", "HOLD": "⚪ HOLD"}

    out = pd.DataFrame(
        {
            "Pos": df["pos"],
            "#": df["vehicle"],
            "Driver": df["driver"],
            "Last": df["lap_time"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—"),
            "Next": df["pred"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—"),
            "Δ": df["delta_pred"].map(lambda x: f"{x:+.2f}" if pd.notna(x) else "—"),
            "Strat": df["strategy"].map(lambda x: (strat.get(x, "") + " " + x).strip()),
            "Tire": df["tire"].map(lambda x: (tire.get(x, "") + " " + x).strip()),
            "Call": df["call"].map(call),
            "Win%": df["win"].map(lambda x: f"{x:.1f}%" if pd.notna(x) else "—"),
        }
    )
    if not out.empty:
        out.iloc[0, out.columns.get_loc("Driver")] = "🏁 " + str(out.iloc[0]["Driver"])
    return out


def lap_chart(hist: pd.DataFrame, selected) -> alt.Chart:
    if hist.empty:
        return alt.Chart(pd.DataFrame())
    df = hist.copy()
    if selected:
        df = df[df["vehicle"].isin(selected)]
    df = df.dropna(subset=["lap", "lap_time"])
    return (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("lap:Q", title="Lap"),
            y=alt.Y("lap_time:Q", title="Lap Time (s)"),
            color=alt.Color("vehicle:N", title="Car"),
            tooltip=["vehicle", "driver", "lap", "lap_time"],
        )
        .properties(height=320)
    )


def render_banner(snap: pd.DataFrame) -> None:
    if snap.empty:
        return
    row = snap.iloc[0]
    track = row.get("track_name") or "Unknown Track"
    session = row.get("run_name") or "Unknown Session"
    date_text = row.get("snapshot_date") or datetime.utcnow().strftime("%Y-%m-%d")
    time_text = row.get("snapshot_time_utc") or datetime.utcnow().strftime("%H:%M:%S UTC")
    lap = row.get("lap")
    race_id = row.get("race_id")

    subtitle_parts = [f"Date: {date_text}", f"Session: {session}", f"Updated: {time_text}"]
    if pd.notna(lap):
        subtitle_parts.append(f"Lap: {int(lap)}")
    if pd.notna(race_id):
        subtitle_parts.append(f"Race ID: {int(race_id)}")

    st.markdown(
        f"""
        <div class="banner">
            <div class="banner-title">🏁 {track}</div>
            <div class="banner-subtitle">{' • '.join(subtitle_parts)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.title("🏎️ NASCAR Race Intelligence")
   
    auto = st.sidebar.checkbox("Auto Refresh", True)
    rate = st.sidebar.slider("Seconds", 3, 10, 5)

    if "last" not in st.session_state:
        st.session_state.last = 0.0

    manual_fetch = st.sidebar.button("Fetch Now")
    if manual_fetch:
        st.session_state.last = 0.0

    should_refresh = auto and (time.time() - st.session_state.last > rate)
    if should_refresh or manual_fetch or "snap" not in st.session_state:
        try:
            data = fetch_data()
            snap = normalize(data)
            add_hist(snap)
            st.session_state.snap = snap
            st.session_state.last = time.time()
            st.session_state.fetch_error = None
        except Exception as exc:
            st.session_state.fetch_error = str(exc)

    if st.session_state.get("fetch_error"):
        st.error(f"Fetch error: {st.session_state.fetch_error}")

    if "snap" not in st.session_state or st.session_state.snap.empty:
        st.info("Waiting for data...")
        return

    hist = get_hist()
    snap = st.session_state.snap

    render_banner(snap)

    table = enrich(snap, hist)
    table = simulate(table)
    table = table.sort_values("pos", na_position="last").reset_index(drop=True)

    leader = table.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Leader", f"#{leader['vehicle']}")
    c2.metric("Lap", int(leader["lap"]) if pd.notna(leader["lap"]) else "—")
    c3.metric("Fastest", f"{table['lap_time'].min():.3f}" if table['lap_time'].notna().any() else "—")
    c4.metric("Avg", f"{table['lap_time'].mean():.3f}" if table['lap_time'].notna().any() else "—")

    st.subheader("🏁 Field")
    st.dataframe(format_table(table), use_container_width=True, hide_index=True)

    default_sel = table["vehicle"].head(5).tolist()
    selected = st.multiselect("Drivers", table["vehicle"].tolist(), default=default_sel)
    st.subheader("📈 Lap Trends")
    st.altair_chart(lap_chart(hist, selected), use_container_width=True)

    st.subheader("🧠 Strategy")
    any_call = False
    for _, r in table.head(8).iterrows():
        if r["call"] == "PIT":
            st.success(f"Pit #{r['vehicle']} → projected gain {int(r['gain'])}")
            any_call = True
        elif r["call"] == "STAY":
            st.warning(f"Stay out #{r['vehicle']} → pit now not favorable")
            any_call = True
    if not any_call:
        st.info("No strong strategy call yet.")

    st.subheader("🏁 Pit vs Stay Out")
    sim_view = table[["driver", "pos", "pit_pos", "gain", "stay_pos", "loss", "call"]].rename(
        columns={
            "driver": "Driver",
            "pos": "Current",
            "pit_pos": "Pit Now →",
            "gain": "Gain",
            "stay_pos": "Stay Out →",
            "loss": "Loss",
            "call": "Call",
        }
    )
    st.dataframe(sim_view, use_container_width=True, hide_index=True)

    if auto:
        time.sleep(0.1)
        st.rerun()


if __name__ == "__main__":
    main()
