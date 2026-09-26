"""Ramadan & Eid forecast for a place: which new moon opens which Hijri month, the crescent-sighting evening
(skyq.plan_hilal: Odeh 2004 + weather/dust), expected start dates, and prayer times for the key days.

A scientific forecast, not an official announcement (in Iraq, the endowment authorities announce the dates)."""
from datetime import date, datetime, timedelta, timezone
from math import ceil, floor
from zoneinfo import ZoneInfo

import numpy as np

import skyq

MONTHS = ["محرم", "صفر", "ربيع الأول", "ربيع الآخر", "جمادى الأولى", "جمادى الآخرة", "رجب", "شعبان", "رمضان", "شوال",
          "ذو القعدة", "ذو الحجة"]
ISLAMIC_EPOCH = 1948439.5  # JD of 1 Muharram 1 AH (civil/tabular epoch)
FAJR_ANGLE = 18.0  # ponytail: common 18° convention; Iraqi endowment calendars may differ by a few minutes
IMSAK_MIN = 10


def _islamic_to_jd(y, m, d):
    return d + ceil(29.5 * (m - 1)) + (y - 1) * 354 + floor((3 + 11 * y) / 30) + ISLAMIC_EPOCH - 1


def gregorian_to_hijri(d: date):
    """Tabular (arithmetic) Islamic calendar. Off by 1-2 days vs. real sighting, so it is only used to label
    which Hijri month a new moon opens (evaluated mid-month, where that error can't change the month)."""
    jd = floor(d.toordinal() + 1721424.5) + 0.5
    y = floor((30 * (jd - ISLAMIC_EPOCH) + 10646) / 10631)
    m = min(12, ceil((jd - (29 + _islamic_to_jd(y, 1, 1))) / 29.5) + 1)
    return int(y), int(m), int(jd - _islamic_to_jd(y, m, 1) + 1)


def month_starts(today: date, days=400):
    """[(conjunction_utc, hijri_year, hijri_month)] for new moons from ~3 weeks ago to `days` ahead."""
    out = []
    for c in skyq.conjunctions(datetime.combine(today - timedelta(days=20), datetime.min.time()), days):
        y, m, _ = gregorian_to_hijri(c.date() + timedelta(days=15))
        out.append((c, y, m))
    return out


def sun_times(lat, lon, d: date, tz="Asia/Baghdad"):
    """Fajr (sun -18°), imsak, sunrise, maghrib/sunset for local date d, at 1-minute resolution (astropy)."""
    z = ZoneInfo(tz)
    local = [datetime.combine(d, datetime.min.time()) + timedelta(minutes=i) for i in range(24 * 60)]
    utc = [t.replace(tzinfo=z).astimezone(timezone.utc).replace(tzinfo=None) for t in local]
    alt = skyq.geometry(utc, lat, lon, None, radec=(0.0, 0.0, 0.0)).sun_alt.to_numpy()
    up = lambda h: next((local[i] for i in range(1, len(alt)) if alt[i - 1] < h <= alt[i]), None)
    down = lambda h: next((local[i] for i in range(1, len(alt)) if alt[i - 1] >= h > alt[i]), None)
    fajr = up(-FAJR_ANGLE)
    sunrise = up(-0.833)
    return dict(fajr=fajr, imsak=fajr - timedelta(minutes=IMSAK_MIN) if fajr else None, sunrise=sunrise,
                maghrib=down(-0.833), eid_prayer=sunrise + timedelta(minutes=20) if sunrise else None)


def _sighting(lat, lon, conj):
    """Crescent forecast for the evenings after this new moon; month starts the day after the first naked-eye /
    binocular sighting by the clear-sky Odeh criterion (A/B). The date follows the crescent's geometry — a sighting
    anywhere clear counts — while `local_zone` says whether dust/cloud at *this* place may hide it that evening.
    If not visible in 3 evenings -> the month is completed (30 days)."""
    hd, notes = skyq.plan_hilal(lat, lon, conj.date(), 3)
    seen = hd[hd.zone.isin(["A", "B"])]
    tele = hd[hd.zone == "C"]
    if len(seen):
        ev = seen.iloc[0]
        basis = "رؤية بالعين أو الناظور"
    else:
        ev = hd.iloc[-1]
        basis = "إكمال الشهر 30 يوم (الهلال ما يبين بالعين قبلها)"
    return dict(start=ev["date"] + timedelta(days=1), evening=ev["date"], zone=ev["zone"], local_zone=ev["final_zone"],
                basis=basis,
                best_time=ev.get("best_time"), moon_az=ev.get("moon_az"), moon_alt=ev.get("moon_alt"),
                lag_min=ev.get("lag_min"), arcl=ev.get("arcl"), age_h=ev.get("age_h"),
                telescope_evening=tele.iloc[0]["date"] if len(tele) else None, evenings=hd, weather_notes=notes)


def events(lat, lon, today: date, tz="Asia/Baghdad"):
    """Next Ramadan start, Eid al-Fitr and Eid al-Adha for this place (each as a dict), soonest first."""
    starts = month_starts(today)
    out = []
    for key, month in (("ramadan", 9), ("fitr", 10), ("adha", 12)):
        for conj, y, m in starts:
            if m != month:
                continue
            s = _sighting(lat, lon, conj)
            day = s["start"] + timedelta(days=9) if key == "adha" else s["start"]
            if day < today:  # already passed -> take next year's
                continue
            title = {"ramadan": "بداية رمضان", "fitr": "عيد الفطر", "adha": "عيد الأضحى"}[key]
            hijri = {"ramadan": f"1 رمضان {y} هـ", "fitr": f"1 شوال {y} هـ", "adha": f"10 ذو الحجة {y} هـ"}[key]
            ev = dict(key=key, title=title, hijri=hijri, date=day, days_left=(day - today).days, conjunction_utc=conj,
                      month_start=s["start"], arafah=s["start"] + timedelta(days=8) if key == "adha" else None,
                      forecast_is_climatology=(s["evening"] - today).days > 16, **{k: v for k, v in s.items() if k != "start"})
            times = sun_times(lat, lon, day, tz)
            if key == "ramadan":
                ev["imsak"], ev["iftar"] = times["imsak"], times["maghrib"]
            else:
                ev["eid_prayer"] = times["eid_prayer"]
            out.append(ev)
            break
    return sorted(out, key=lambda e: e["date"])


if __name__ == "__main__":  # self-check against known dates
    # tabular calendar sanity: 1 Ramadan 1447 ≈ 18-19 Feb 2026; 1 Shawwal 1445 ≈ 9-10 Apr 2024
    y, m, d = gregorian_to_hijri(date(2026, 2, 19))
    assert (y, m) == (1447, 9) and d in (1, 2), (y, m, d)
    y, m, d = gregorian_to_hijri(date(2024, 4, 10))
    assert (y, m) == (1445, 10) and d in (1, 2), (y, m, d)
    # known outcome: Shawwal 1445 crescent seen evening 9 Apr 2024 -> Eid 10 Apr 2024
    starts = [s for s in month_starts(date(2024, 3, 25), 45) if s[2] == 10]
    s = _sighting(*skyq.MOSUL, starts[0][0])
    print("Shawwal 1445 from Mosul:", s["evening"], "->", s["start"], s["basis"])
    assert s["start"] == date(2024, 4, 10)
    t = sun_times(*skyq.MOSUL, date(2026, 9, 26))
    print({k: f"{v:%H:%M}" for k, v in t.items()})
    for e in events(*skyq.MOSUL, date(2026, 9, 26)):
        print(e["title"], e["hijri"], e["date"], "in", e["days_left"], "days | evening", e["evening"], e["zone"], e["basis"],
              "| imsak/iftar" if e["key"] == "ramadan" else "", e.get("imsak"), e.get("iftar"), e.get("eid_prayer"))
    print("OK")
