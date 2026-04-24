import requests
import pandas as pd
import time
import os
from datetime import datetime

LIVE_FEED_URL = "https://cf.nascar.com/live/feeds/live-feed.json"
POLL_INTERVAL = 5


def normalize(data):
    rows = []

    race_id = data.get("race_id")
    track = str(data.get("track_name", "unknown")).replace(" ", "_").lower()
    series = str(data.get("series_name", "cup")).lower()

    for v in data.get("vehicles", []):
        rows.append({
            "timestamp": datetime.utcnow(),
            "race_id": race_id,
            "track": track,
            "series": series,
            "lap": data.get("lap_number"),
            "flag_state": data.get("flag_state"),
            "vehicle": v.get("vehicle_number"),
            "driver": (v.get("driver") or {}).get("full_name"),
            "position": v.get("running_position"),
            "lap_time": v.get("last_lap_time"),
            "pit_count": len(v.get("pit_stops", []))
        })

    return pd.DataFrame(rows)


def save_snapshot(df):
    if df.empty:
        return

    race_id = df["race_id"].iloc[0]
    track = df["track"].iloc[0]
    series = df["series"].iloc[0]

    folder = f"nascar-data/{series}/{track}"
    os.makedirs(folder, exist_ok=True)

    filepath = f"{folder}/race_{race_id}.parquet"

    if os.path.exists(filepath):
        existing = pd.read_parquet(filepath)
        combined = pd.concat([existing, df]).drop_duplicates(
            subset=["timestamp", "vehicle", "lap"], keep="last"
        )
        combined.to_parquet(filepath, index=False)
    else:
        df.to_parquet(filepath, index=False)


while True:
    try:
        response = requests.get(LIVE_FEED_URL, timeout=10)
        response.raise_for_status()

        data = response.json()
        df = normalize(data)
        save_snapshot(df)

        print(f"Saved lap {df['lap'].iloc[0]}")

    except Exception as e:
        print(e)

    time.sleep(POLL_INTERVAL)
