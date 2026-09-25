"""Iraq Sky AI — REST API (Mosul / Nineveh). Run:  uvicorn api:app --port 8000   · docs at /docs"""
from datetime import date, datetime, timedelta
from typing import Literal

from fastapi import FastAPI, HTTPException, Query

import engine
import skyq

app = FastAPI(title="Iraq Sky AI (Mosul)", description="Satellite night lights + weather + Moon → Sky Quality Score "
              "and best observation spots in Nineveh.")
ObsType = Literal["stargazing", "milky_way", "astrophotography", "meteors"]
CLASSES = ["Poor", "Moderate", "Good", "Excellent"]


def _when(d: date | None, t: str | None) -> datetime:
    d = d or date.today()
    h, m = map(int, (t or "22:00").split(":"))
    when = datetime.combine(d, datetime.min.time()) + timedelta(hours=h, minutes=m)
    return when + timedelta(days=1) if h < 12 else when  # 00:00-11:59 means after midnight of that night


def _check(lat, lon):
    if not skyq.in_nineveh(lat, lon):
        raise HTTPException(400, f"Location outside the study area (Nineveh {engine.NINEVEH})")


@app.get("/sky-quality")
def sky_quality(lat: float, lon: float, date: date | None = None, time: str | None = Query(None, pattern=r"^\d{1,2}:\d{2}$"),
                type: ObsType = "stargazing"):
    _check(lat, lon)
    a = engine.analyze(lat, lon, _when(date, time), type)
    return dict(score=a["score"], **{"class": CLASSES[a["cls"]]}, light_pollution=["High", "Medium", "Low", "Very low"][
                    ["عالي 🔴", "متوسط 🟠", "واطي 🟢", "واطي جداً ⭐"].index(a["light_pollution"])],
                cloud_cover=a["cloud_cover"], humidity=a["humidity"], dust_aod=a["aod"],
                moon_impact={"واطي 🟢": "Low", "متوسط 🟠": "Medium", "عالي 🔴": "High"}[a["moon_impact"]],
                elevation_m=a["elevation_m"], road_km=a["road_km"], components=a["components"],
                best_time=None if a["best_time"] is None else [t.strftime("%H:%M") for t in a["best_time"]],
                suitable_for={k: bool(v) for k, v in a["suitable"].items()}, recommended=a["recommended"],
                forecast_fallback=a["notes"])


@app.get("/best-locations")
def best_locations(lat: float, lon: float, radius: float = Query(100, gt=0, le=200), date: date | None = None,
                   time: str | None = Query(None, pattern=r"^\d{1,2}:\d{2}$"), type: ObsType = "stargazing", n: int = Query(3, le=10)):
    _check(lat, lon)
    rec, notes = engine.recommend(lat, lon, radius, _when(date, time), type, n=n)
    return dict(locations=[dict(rank=i + 1, lat=r.lat, lon=r.lon, score=int(r.score), **{"class": CLASSES[int(r.cls)]},
                                distance_km=round(r.km), road_km=round(r.road_km, 1), elevation_m=round(r.elevation_m),
                                cloud_cover=round(r.cloud_cover)) for i, r in enumerate(rec.itertuples())],
                forecast_fallback=notes)


@app.get("/pollution-map")
def pollution_map():
    t = engine.cells()
    return dict(bbox=engine.NINEVEH, cell_deg=engine.STEP * skyq.RES, cells=[
        dict(lat=r.lat, lon=r.lon, score=round(r.lp_score), **{"class": CLASSES[int(r.lp_cls)]}) for r in t.itertuples()])


@app.get("/observation-conditions")
def observation_conditions(lat: float, lon: float, date: date | None = None, type: ObsType = "stargazing"):
    _check(lat, lon)
    b = engine.best_time(lat, lon, date or datetime.now().date(), type)
    return dict(best_window=None if b["window"] is None else [t.strftime("%H:%M") for t in b["window"]],
                hourly=[dict(time=r.time.strftime("%Y-%m-%d %H:%M"), score=int(r.score), **{"class": CLASSES[int(r.cls)]})
                        for r in b["hourly"].itertuples()])
