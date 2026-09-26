"""Iraq crescent campaigns. Visibility predictions never constitute month announcements."""
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
import math

import pandas as pd
import hijri
import skyq

EVENTS = {"ramadan": (9, "هلال رمضان"), "fitr": (10, "هلال شوال · عيد الفطر"),
          "adha": (12, "هلال ذي الحجة · عيد الأضحى")}
CITIES = {"الموصل": (36.34, 43.13), "بغداد": (33.3152, 44.3661),
          "البصرة": (30.5085, 47.7804), "أربيل": (36.1911, 44.0092),
          "السليمانية": (35.557, 45.435), "دهوك": (36.867, 42.95),
          "كركوك": (35.468, 44.392), "النجف": (32.0, 44.33),
          "كربلاء": (32.616, 44.024), "الناصرية": (31.044, 46.257),
          "الرمادي": (33.425, 43.3), "الحلة": (32.473, 44.421),
          "الكوت": (32.513, 45.824), "العمارة": (31.835, 47.145),
          "الديوانية": (31.99, 44.93), "السماوة": (31.318, 45.28),
          "بعقوبة": (33.746, 44.643), "تكريت": (34.608, 43.679)}
ZONES = {"A": "مرشح للرؤية بالعين بسماء صافية", "B": "يحتاج مساعدة بصرية؛ قد يُرى بالعين",
         "C": "مرشح للرصد البصري بالأجهزة فقط", "D": "غير مرشح للرؤية وفق النموذج"}


@lru_cache(maxsize=32)
def campaign(event, year):
    if event not in EVENTS or not 1400 <= year <= 1500:
        raise ValueError("Invalid event or Hijri year")
    month = EVENTS[event][0]
    anchor = date.fromordinal(round(hijri._islamic_to_jd(year, month, 1) - 1721424.5))
    candidates = skyq.conjunctions(datetime.combine(anchor - timedelta(days=5), datetime.min.time()), 10)
    if not candidates:
        raise ValueError("No conjunction found")
    conjunction = min(candidates, key=lambda c: abs((c.date() - anchor).days))
    return conjunction, (conjunction + timedelta(hours=3)).date()


def plan(lat, lon, evening, nights=3, today=None):
    """Keep geometry separate from an experimental weather adjustment; no climatology verdicts."""
    if not (29 <= lat <= 38 and 38 <= lon <= 49):
        raise ValueError("Coordinates must be in the Iraq planning area")
    today = today or datetime.now(timezone(timedelta(hours=3))).date()
    rows = []
    for i in range(nights):
        day = evening + timedelta(days=i)
        h = skyq.hilal_evening(lat, lon, day, utc_offset_h=3)
        h.update(local_zone=None, weather_status="خارج المدى الموثوق للطقس والغبار", cloud_cover=None, aod=None)
        # Both inputs must be available. Never turn seasonal defaults into a visibility forecast.
        if 0 <= (day - today).days <= 4 and h.get("best_time"):
            wx, notes = skyq.weather(lat, lon, day, start_hour=15, hours=7)
            if not notes:
                row = wx.iloc[(wx.local_time - h["best_time"]).abs().argmin()].to_dict()
                if all(math.isfinite(float(row[k])) for k in ("cloud_cover", "aod", "visibility_km")):
                    adjusted = skyq.hilal_weather_adjust(h, row)
                    h.update(local_zone=adjusted["final_zone"], cloud_cover=row["cloud_cover"], aod=row["aod"],
                             weather_status="توقع طقس وغبار · تعديل تجريبي غير معاير ميدانيًا")
            else:
                h["weather_status"] = "بيانات الطقس أو الغبار غير مكتملة؛ نعرض الحساب الفلكي فقط"
        rows.append(h)
    return pd.DataFrame(rows)
