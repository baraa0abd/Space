"""Public page (English, beginner-friendly) — SkyQ San Francisco Bay Area (prototype v2).
Three questions (where / when / what) -> one big answer (go or not, why, when, where instead)."""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

import engine
import sf
import skyq

REGION = "sf"
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
[data-testid="stMainBlockContainer"] { font-family: 'Inter', sans-serif; max-width: 900px; }
.step { font-weight: 800; font-size: 1.15rem; margin: 2px 0 6px; }
.step span { display: inline-block; width: 30px; height: 30px; line-height: 30px; text-align: center; border-radius: 50%;
  background: #466eff; color: #fff; margin-right: 8px; }
.hint { opacity: .75; font-size: .92rem; margin-top: -4px; }
.bridge { height: 6px; border-radius: 3px; margin: 2px 0 10px; background: linear-gradient(90deg, #c0362c, #f28c28, #5b8cff); }
.hero { display: flex; gap: 22px; align-items: center; border-radius: 20px; padding: 20px 22px; margin: 10px 0;
  border: 2px solid var(--c); background: linear-gradient(135deg, color-mix(in srgb, var(--c) 22%, transparent), transparent); }
.ring { flex: 0 0 132px; height: 132px; border-radius: 50%; display: grid; place-items: center;
  background: conic-gradient(var(--c) calc(var(--s) * 1%), rgba(128,128,128,.18) 0); }
.ring .in { width: 104px; height: 104px; border-radius: 50%; background: #0b1020; display: grid; place-items: center;
  text-align: center; line-height: 1.1; }
.ring b { font-size: 2.3rem; } .ring small { display: block; opacity: .7; }
.verdict { font-size: 1.9rem; font-weight: 800; margin: 0 0 4px; }
.chips { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin: 8px 0 4px; }
.chip { border-radius: 14px; padding: 10px 8px; text-align: center; border: 1px solid rgba(128,128,128,.25); border-top: 5px solid var(--c); }
.chip .e { font-size: 1.7rem; } .chip .t { font-size: .85rem; opacity: .8; } .chip .v { font-weight: 800; }
.strip { display: grid; grid-template-columns: repeat(13, 1fr); gap: 4px; margin: 6px 0; }
.hr { border-radius: 8px; padding: 6px 0; text-align: center; font-size: .8rem; background: var(--c); color: #111; }
.hr.sel { outline: 3px solid #fff; } .hr b { display: block; font-size: 1.05rem; }
.spot { border-radius: 16px; padding: 14px 16px; margin: 8px 0; border: 1px solid rgba(128,128,128,.25); border-left: 6px solid var(--c); }
.spot .n { font-size: 1.2rem; font-weight: 800; }
.go { display: inline-block; margin-top: 6px; padding: 6px 14px; border-radius: 10px; background: #466eff; color: #fff !important;
  text-decoration: none !important; font-weight: 700; }
@media (max-width: 640px) { .hero { flex-direction: column; text-align: center; } .chips { grid-template-columns: repeat(2, 1fr); }
  .strip { grid-template-columns: repeat(7, 1fr); } }
</style>""", unsafe_allow_html=True)

PLACES = {  # plans/sf_usa_3day_plan.json places + dark-sky reference sites
    "San Francisco – Downtown": (37.7749, -122.4194), "Twin Peaks": (37.7544, -122.4477),
    "Golden Gate Park": (37.7694, -122.4862), "Oakland": (37.8044, -122.2712), "San Jose": (37.3382, -121.8863),
    "Half Moon Bay": (37.4636, -122.4286), "Chabot Space & Science Center": (37.8187, -122.1807),
    "Mount Tamalpais": (37.906, -122.613), "Point Reyes": (38.04, -122.9), "Mount Diablo": (37.8816, -121.9142),
    "Lick Observatory (Mt Hamilton)": (37.3414, -121.6429), "Henry W. Coe State Park": (37.187, -121.547),
    "Pinnacles National Park": (36.494, -121.146),
}
NAMES = {
    "Sirius": "Sirius", "Vega": "Vega", "Altair": "Altair", "Deneb": "Deneb", "Betelgeuse": "Betelgeuse",
    "Polaris": "Polaris (North Star)", "Pleiades (M45)": "Pleiades (Seven Sisters)", "Orion Nebula (M42)": "Orion Nebula",
    "Andromeda Galaxy (M31)": "Andromeda Galaxy", "Hercules Cluster (M13)": "Hercules Cluster", "Ring Nebula (M57)": "Ring Nebula",
    "Whirlpool Galaxy (M51)": "Whirlpool Galaxy", "Jupiter": "Jupiter 🪐", "Saturn": "Saturn 🪐", "Mars": "Mars 🔴",
}
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DIRS = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"]
LEVEL = [("#e5484d", "😞 Not now", "Poor"), ("#f5a524", "😐 So-so", "Moderate"),
         ("#30a46c", "😊 Good — worth going!", "Good"), ("#5b8cff", "🤩 Excellent! Go stargazing", "Excellent")]
TYPE_UI = {
    "stargazing": ("✨ Stars", "See stars and planets with your eyes — the easiest place to start."),
    "milky_way": ("🌌 Milky Way", "The faint band of our galaxy — needs a very dark place and no Moon."),
    "astrophotography": ("📷 Photos", "Photograph the sky with a phone or camera — needs clear, dark, steady skies."),
    "meteors": ("☄️ Meteors", "Shooting stars — needs patience and a fairly dark sky."),
}
HOUR_LABEL = {19: "7 pm", 20: "8 pm", 21: "9 pm", 22: "10 pm", 23: "11 pm", 0: "midnight", 1: "1 am", 2: "2 am", 3: "3 am", 4: "4 am"}
COLORS = [[229, 72, 77], [245, 165, 36], [48, 164, 108], [91, 140, 255]]
SAFE = "#30a46c"


def compass(az):
    return DIRS[int((az + 22.5) % 360 // 45)]


def bearing(a1, o1, a2, o2):
    p1, p2, dl = np.radians(a1), np.radians(a2), np.radians(o2 - o1)
    return np.degrees(np.arctan2(np.sin(dl) * np.cos(p2), np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl))) % 360


def place_name(lat, lon):
    n, (a, o) = min(PLACES.items(), key=lambda kv: skyq.haversine_km(lat, lon, *kv[1]))
    d = skyq.haversine_km(a, o, lat, lon)
    return f"near {n}" if d < 8 else f"{d:.0f} km {compass(bearing(a, o, lat, lon))} of {n}"


def reasons(a):
    c, out = a["components"], []
    if a["score"] <= 10 and c["weather"] > 10:
        out.append("☀️ It isn't dark yet — wait for the sky to get fully dark")
    out.append("💡 City lights are strong here and hide faint stars" if c["light"] < 35 else
               "💡 Some city glow nearby" if c["light"] < 65 else "💡 Far from city lights 👍")
    if a["cloud_cover_low"] >= 50:
        out.append(f"🌁 Fog / low clouds ({a['cloud_cover_low']}%)" + (" — but you're above most of it 👍" if a["elevation_m"] >= 600 else ""))
    out.append(f"☁️ Cloudy sky ({a['cloud_cover']}%)" if a["cloud_cover"] >= 50 else
               f"⛅ Some clouds ({a['cloud_cover']}%)" if a["cloud_cover"] >= 20 else "☀️ Clear sky 👍")
    if a["pm2_5"] >= 35:
        out.append(f"🔥 Wildfire smoke in the air (PM2.5 {a['pm2_5']})")
    out.append("🌕 Bright Moon washes out faint stars" if c["moon"] < 50 else
               "🌓 The Moon matters a bit" if c["moon"] < 80 else "🌑 Moon isn't a problem 👍")
    return out


@st.cache_resource(show_spinner="⏳ Preparing the Bay Area map…")
def cell_table():
    t = sf.cells()
    t["color"] = [COLORS[int(c)] + [140] for c in t.lp_cls]
    t["label"] = [["Bright sky 🔴", "In-between 🟠", "Dark sky 🟢", "Very dark sky ⭐"][int(c)] for c in t.lp_cls]
    return t


@st.cache_data(ttl=1800, show_spinner="⏳ Checking the sky, clouds, fog, smoke and the Moon…")
def analyze(lat, lon, when, obs):
    return engine.analyze(lat, lon, when, obs, region=REGION)


@st.cache_data(ttl=1800, show_spinner="⏳ Finding the best spots near you…")
def recommend(lat, lon, radius, when, obs):
    return engine.recommend(lat, lon, radius, when, obs, region=REGION)


@st.cache_resource(show_spinner=False)
def obs_model():
    return skyq.load_model()[0]


@st.cache_data(ttl=3600, show_spinner="⏳ Working out where the stars and planets are…")
def targets(lat, lon, night, sqm):
    model, wx, rows = obs_model(), skyq.weather(lat, lon, night), []
    for t, ra_dec in skyq.CATALOG.items():
        mag = skyq.PLANET_MAG[t] if ra_dec is None else ra_dec[2]
        ap = 7 if mag <= 4.5 else 150
        df, rec, _ = skyq.plan_night(model, lat, lon, night, t, sqm, ap, wx=wx)
        best = df.score.idxmax()
        rows.append(dict(target=t, aperture=ap, score=df.score.max(), rec=rec, alt=df.target_alt[best],
                         az=df.target_az[best], hour=df.local_time[best]))
    return pd.DataFrame(rows), wx[1]  # notes: forecast fallback used?


@st.cache_data(ttl=3600, show_spinner=False)
def next_new_moon(today):
    return skyq.conjunctions(datetime.combine(today, datetime.min.time()) - timedelta(days=2), 32)[0]


def reset_map():
    st.session_state.sf_map_v = st.session_state.get("sf_map_v", 0) + 1


CELLS = cell_table()

st.markdown("# 🌉 SkyQ San Francisco")
st.markdown('<div class="bridge"></div>', unsafe_allow_html=True)
st.markdown("#### Answer 3 quick questions and we'll tell you **whether to go stargazing, when, and where** 🌌")

with st.container(border=True):
    st.markdown('<div class="step"><span>1</span>Where are you?</div>', unsafe_allow_html=True)
    place = st.selectbox("Your area", list(PLACES), label_visibility="collapsed", on_change=reset_map)
    lat, lon = PLACES[place]
    map_key = f"sf_map_{st.session_state.get('sf_map_v', 0)}"
    picked = st.session_state.get(map_key, {}).get("selection", {}).get("objects", {}).get("cells", [])
    if picked:
        lat, lon = picked[0]["lat"], picked[0]["lon"]
        place = place_name(lat, lon)
        c1, c2 = st.columns([3, 1])
        c1.info(f"📍 You picked a point on the map: **{place}**")
        c2.button("↩️ Back to list", on_click=reset_map, use_container_width=True)
    with st.expander("🗺️ Or pick a spot on the map (tap any coloured dot)"):
        st.markdown("Each dot shows how **dark** the sky is there: 🔴 bright (city) · 🟠 in-between · 🟢 dark · "
                    "🔵 very dark ⭐ — **darker is better for stars**")
        st.pydeck_chart(pdk.Deck(
            layers=[pdk.Layer("ScatterplotLayer", CELLS, id="cells", get_position="[lon, lat]", get_fill_color="color",
                              get_radius=2000, pickable=True),
                    pdk.Layer("ScatterplotLayer", pd.DataFrame([dict(lat=lat, lon=lon, label="📍 You are here")]),
                              get_position="[lon, lat]", get_fill_color=[255, 255, 255], get_radius=2800, stroked=True,
                              get_line_color=[0, 0, 0], line_width_min_pixels=2, pickable=True)],
            initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=7.6), map_style="dark",
            tooltip={"text": "{label}"}), key=map_key, on_select="rerun", selection_mode="single-object", height=360)

with st.container(border=True):
    st.markdown('<div class="step"><span>2</span>When do you want to go?</div>', unsafe_allow_html=True)
    now = datetime.now()
    today = now.date() - timedelta(days=1) if now.hour < 6 else now.date()
    day = st.segmented_control("Day", ["Tonight", "Tomorrow", "In 2 days", "📅 Other day"], default="Tonight",
                               label_visibility="collapsed") or "Tonight"
    night = {"Tonight": today, "Tomorrow": today + timedelta(days=1), "In 2 days": today + timedelta(days=2)}.get(day)
    if night is None:
        night = st.date_input("Pick a day", value=today + timedelta(days=3), min_value=today, max_value=today + timedelta(days=14))
    hour = st.select_slider("Time", options=list(HOUR_LABEL), value=22, format_func=HOUR_LABEL.get)
    when = datetime.combine(night + timedelta(days=1 if hour < 12 else 0), datetime.min.time()) + timedelta(hours=hour)
    st.markdown(f'<div class="hint">🗓️ {DAYS[night.weekday()]} {night:%b %d} — at {HOUR_LABEL[hour]}</div>', unsafe_allow_html=True)

with st.container(border=True):
    st.markdown('<div class="step"><span>3</span>What do you want to see?</div>', unsafe_allow_html=True)
    obs = st.segmented_control("Type", list(TYPE_UI), default="stargazing", format_func=lambda k: TYPE_UI[k][0],
                               label_visibility="collapsed") or "stargazing"
    st.markdown(f'<div class="hint">{TYPE_UI[obs][1]}</div>', unsafe_allow_html=True)
    radius = st.segmented_control("How far can you travel?", [25, 50, 100], default=100,
                                  format_func={25: "🚶 Close (25 km)", 50: "🚗 Short drive (50 km)", 100: "🚙 Day trip (100 km)"}.get) or 100

a = analyze(lat, lon, when, obs)
rec, rec_notes = recommend(lat, lon, radius, when, obs)
rec = rec[rec.score > a["score"]].reset_index(drop=True) if len(rec) else rec  # only spots better than here
k = a["cls"]
color, verdict, _ = LEVEL[k]
why = "".join(f"<li>{r}</li>" for r in reasons(a))
st.markdown(f"""<div class="hero" style="--c:{color}; --s:{a['score']}">
  <div class="ring"><div class="in"><div><b>{a['score']}</b><small>out of 100</small></div></div></div>
  <div><div class="verdict">{verdict}</div>
  <div>{TYPE_UI[obs][0]} · {place} · {DAYS[night.weekday()]} {HOUR_LABEL[hour]}</div>
  <ul>{why}</ul></div></div>""", unsafe_allow_html=True)

bt = a["best_time"]
best = rec.iloc[0] if len(rec) else None
if best is not None and int(best.score) >= a["score"] + 10:
    st.success(f"👉 **Our tip:** go to **{place_name(best.lat, best.lon)}** ({best.km:.0f} km) — it scores "
               f"**{int(best.score)}/100** there. See the spots below ⬇️")
elif bt:
    st.success(f"👉 **Our tip:** the best time this night is **{bt[0]:%H:%M}–{bt[1]:%H:%M}**")
elif k <= 1:
    st.warning("👉 **Our tip:** try another night or time above — this night has no good window here.")

c = a["components"]
chip = lambda e, t, v, s: (f'<div class="chip" style="--c:{"#30a46c" if s >= 70 else "#f5a524" if s >= 40 else "#e5484d"}">'
                           f'<div class="e">{e}</div><div class="t">{t}</div><div class="v">{v}</div></div>')
fog_s = 100 - a["cloud_cover_low"] * (0.3 if a["elevation_m"] >= 600 else 1)
st.markdown('<div class="chips">'
            + chip("💡", "City lights", "strong" if c["light"] < 35 else "some" if c["light"] < 65 else "low", c["light"])
            + chip("☁️", "Clouds", f"{a['cloud_cover']}%", 100 - a["cloud_cover"])
            + chip("🌁", "Fog", f"{a['cloud_cover_low']}%" + (" (below you)" if a["elevation_m"] >= 600 and a["cloud_cover_low"] else ""), fog_s)
            + chip("🌕", "Moon", "no problem" if c["moon"] >= 80 else "a bit" if c["moon"] >= 50 else "bright", c["moon"])
            + chip("🔥", "Smoke", "clean" if a["pm2_5"] < 12 else "light" if a["pm2_5"] < 35 else "smoky", c["smoke"])
            + "</div>", unsafe_allow_html=True)

st.markdown("##### 🕒 The whole night (green = good)")
cells_html = ""
for r in a["hourly"].itertuples():
    col = ["#e5484d", "#f5a524", "#30a46c", "#5b8cff"][int(r.cls)]
    mark = ["❌", "⚠️", "✅", "⭐"][int(r.cls)]
    sel = " sel" if r.time == when else ""
    cells_html += f'<div class="hr{sel}" style="--c:{col}"><b>{mark}</b>{r.time:%H}:00</div>'
st.markdown(f'<div class="strip">{cells_html}</div>', unsafe_allow_html=True)

st.markdown(f"### 🚗 Where should I go? Best spots within {radius} km")
st.caption("Picked because the sky is dark there, the weather looks good at your chosen time, and they're close to a main road.")
if len(rec):
    for i, r in enumerate(rec.itertuples(), 1):
        col, _, short = LEVEL[int(r.cls)]
        medal = ["🥇", "🥈", "🥉"][i - 1] if i <= 3 else f"{i}."
        road = "" if pd.isna(r.road_km) else f" · 🛣️ {r.road_km:.1f} km from a main road"
        st.markdown(f"""<div class="spot" style="--c:{col}"><div class="n">{medal} {place_name(r.lat, r.lon)} — {int(r.score)}/100 ({short})</div>
          🚗 {r.km:.0f} km away · ☁️ clouds {r.cloud_cover:.0f}% · 🌁 fog {r.cloud_cover_low:.0f}%{road} · ⛰️ {r.elevation_m:.0f} m high<br>
          <a class="go" target="_blank" href="https://www.google.com/maps/dir/?api=1&origin={lat},{lon}&destination={r.lat},{r.lon}">🧭 Take me there</a></div>""",
                    unsafe_allow_html=True)
else:
    st.info("No reachable spot beats your location in this distance. Try 🚙 Day trip (100 km) or another night.")

st.markdown("### 🎒 More")
t1, t2, t3, t4 = st.tabs(["✨ What can I see?", "🌑 Next dark nights", "🙋 Share what you saw", "📖 What does it mean?"])
sqm = a["sqm"] or 18.0
with t1:
    st.caption("Stars and planets you can see from here on this night, and where to look.")
    tg, tg_notes = targets(lat, lon, night, sqm)
    if tg_notes:
        st.warning("⚠️ The live forecast didn't load for this list, so it assumes typical (mostly clear) weather. "
                   "Trust the main answer above for clouds.")
    vis = tg[tg.score >= skyq.FAIR].sort_values("score", ascending=False)
    if len(vis):
        for r in vis.itertuples():
            how = "👁️ with your eyes" if r.aperture == 7 else "🔭 needs a small telescope"
            w = f"{r.rec['start']:%H:%M}–{r.rec['end']:%H:%M}" if r.rec else f"{r.hour:%H:%M}"
            st.markdown(f'<div class="spot" style="--c:{SAFE}"><div class="n">{NAMES[r.target]}</div>{how} · '
                        f'{"clear" if r.score >= skyq.GOOD else "faint"} · ⏰ {w} · 🧭 look <b>{compass(r.az)}</b>, '
                        f'{"low in the sky" if r.alt < 30 else "halfway up" if r.alt < 60 else "high overhead"}</div>', unsafe_allow_html=True)
    else:
        st.warning("Nothing clear from here this night — try one of the spots above.")
with t2:
    nm = next_new_moon(night)
    local = nm.replace(tzinfo=timezone.utc).astimezone(ZoneInfo(sf.TZ))
    st.markdown(f"🌑 **New Moon: {local:%a %b %d}.** The darkest nights (best for the Milky Way and photos) are about "
                f"**{(local - timedelta(days=3)):%b %d} – {(local + timedelta(days=3)):%b %d}**.")
with t3:
    st.markdown("Went out tonight? Tell us what you saw — it helps the model learn the Bay Area 🙏")
    with st.form("contribute_sf"):
        mw = st.radio("Could you see the Milky Way (a faint cloudy band)?", ["Yes 🌌", "No"], horizontal=True)
        sqm_in = st.number_input("Have an SQM sky meter? Enter the reading (optional)", 0.0, 22.5, 0.0, 0.1)
        if st.form_submit_button("📨 Send"):
            sf.contribute(lat, lon, mw.startswith("Yes"), sqm_in or None)
            st.success("Got it — thank you! 🙏")
with t4:
    st.markdown("""
- **Score (out of 100):** how good the conditions are at your place and time. Above 50 = worth going.
- **Light pollution 💡:** city light that escapes into the sky and hides stars. We measure it from **NASA night-time satellite images**.
- **Fog / marine layer 🌁:** the Bay's low clouds. High places (Mt Tam, Mt Diablo, Mt Hamilton) are often **above** it.
- **Smoke 🔥:** wildfire smoke makes the sky hazy and hides faint stars.
- **Milky Way 🌌:** our own galaxy — a faint band you can only see from a really dark place.
- **How do we compute it?** An AI model (XGBoost) that learned from **14,000+ real sky measurements** in the US West, plus live weather, smoke and Moon data.
""")

if a["notes"] or rec_notes:
    st.caption("ℹ️ Some forecasts aren't available for this time, so we used typical Bay Area values.")
st.caption("Sources: NASA night lights (Black Marble / VNP46A2) · Globe at Night · Open-Meteo (weather, smoke, elevation) · OpenStreetMap (roads)")
