"""SkyQ REST API — Mosul / Nineveh and San Francisco Bay Area. Run:  uvicorn api:app --port 8000   · docs at /docs"""
from datetime import date, datetime, timedelta
from typing import Literal

from fastapi import FastAPI, HTTPException, Query

import engine

app = FastAPI(title="SkyQ (Mosul + San Francisco)", description="Satellite night lights + weather + Moon → Sky Quality "
              "Score and best observation spots. Use region=mosul (default) or region=sf.")
ObsType = Literal["stargazing", "milky_way", "astrophotography", "meteors"]
Region = Literal["mosul", "sf"]
CLASSES = ["Poor", "Moderate", "Good", "Excellent"]
TIME = Query(None, pattern=r"^\d{1,2}:\d{2}$")


def _when(d: date | None, t: str | None) -> datetime:
    d = d or date.today()
    h, m = map(int, (t or "22:00").split(":"))
    when = datetime.combine(d, datetime.min.time()) + timedelta(hours=h, minutes=m)
    return when + timedelta(days=1) if h < 12 else when  # 00:00-11:59 means after midnight of that night


def _check(lat, lon, region):
    if not engine.in_region(lat, lon, region):
        raise HTTPException(400, f"Location outside the {region} study area {engine.REGIONS[region]['bbox']}")


@app.get("/sky-quality")
def sky_quality(lat: float, lon: float, date: date | None = None, time: str | None = TIME,
                type: ObsType = "stargazing", region: Region = "mosul"):
    _check(lat, lon, region)
    a = engine.analyze(lat, lon, _when(date, time), type, region)
    return dict(region=region, score=a["score"], **{"class": CLASSES[a["cls"]]},
                light_pollution=["High", "Medium", "Low", "Very low"][a["lp_cls"]],
                cloud_cover=a["cloud_cover"], low_cloud_fog=a["cloud_cover_low"], humidity=a["humidity"],
                dust_aod=a["aod"], smoke_pm2_5=a["pm2_5"],
                moon_impact="Low" if a["moon_score"] >= 80 else "Medium" if a["moon_score"] >= 50 else "High",
                elevation_m=a["elevation_m"], road_km=a["road_km"], components=a["components"],
                best_time=None if a["best_time"] is None else [t.strftime("%H:%M") for t in a["best_time"]],
                suitable_for={k: bool(v) for k, v in a["suitable"].items()}, recommended=a["recommended"],
                forecast_fallback=a["notes"])


@app.get("/best-locations")
def best_locations(lat: float, lon: float, radius: float = Query(100, gt=0, le=200), date: date | None = None,
                   time: str | None = TIME, type: ObsType = "stargazing", n: int = Query(3, le=10), region: Region = "mosul"):
    _check(lat, lon, region)
    rec, notes = engine.recommend(lat, lon, radius, _when(date, time), type, n=n, region=region)
    return dict(region=region, forecast_fallback=notes, locations=[
        dict(rank=i + 1, lat=r.lat, lon=r.lon, score=int(r.score), **{"class": CLASSES[int(r.cls)]}, distance_km=round(r.km),
             road_km=engine._num(r.road_km), elevation_m=round(r.elevation_m), cloud_cover=round(r.cloud_cover),
             low_cloud_fog=round(r.cloud_cover_low)) for i, r in enumerate(rec.itertuples())])


@app.get("/pollution-map")
def pollution_map(region: Region = "mosul"):
    t = engine.REGIONS[region]["cells"]()
    return dict(region=region, bbox=engine.REGIONS[region]["bbox"], cell_deg=engine.STEP * 0.02, cells=[
        dict(lat=r.lat, lon=r.lon, score=round(r.lp_score), **{"class": CLASSES[int(r.lp_cls)]}) for r in t.itertuples()])


@app.get("/observation-conditions")
def observation_conditions(lat: float, lon: float, date: date | None = None, type: ObsType = "stargazing", region: Region = "mosul"):
    _check(lat, lon, region)
    b = engine.best_time(lat, lon, date or datetime.now().date(), type, region)
    return dict(region=region, best_window=None if b["window"] is None else [t.strftime("%H:%M") for t in b["window"]],
                hourly=[dict(time=r.time.strftime("%Y-%m-%d %H:%M"), score=int(r.score), **{"class": CLASSES[int(r.cls)]})
                        for r in b["hourly"].itertuples()])
