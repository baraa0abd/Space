"""Expert / judges page: all controls, model metrics, validation. Home page for the public is home.py."""
import json
from datetime import date, datetime, timedelta

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

import skyq


SITES = {  # name: (lat, lon) — Mosul / Nineveh only
    "Mosul city centre": (36.340, 43.130), "University of Mosul": (36.380, 43.140),
    "Hamdaniya (Qaraqosh)": (36.270, 43.375), "Bashiqa": (36.455, 43.350), "Tel Keppe": (36.490, 43.120),
    "Qayyarah": (35.803, 43.290), "Tel Afar": (36.374, 42.449), "Sinjar": (36.321, 41.877), "Rabia": (36.800, 42.100),
    "Hatra desert": (35.588, 42.718), "Custom…": None,
}


@st.cache_data(show_spinner="Downloading NASA VIIRS Black Marble…")
def base_maps():
    em = skyq.emission_map(skyq.fetch_black_marble())
    return em, skyq.sqm_map(em)


@st.cache_resource(show_spinner="Training observation-quality model…")
def model_bundle():
    return skyq.load_model()


def sky_map(sqm, points: pd.DataFrame, key):
    la0, lo0, la1, lo1 = skyq.BBOX
    layers = [pdk.Layer("BitmapLayer", image=f"'{skyq.colorize(sqm)}'", bounds=[[lo0, la0], [lo0, la1], [lo1, la1], [lo1, la0]], opacity=0.85)]
    if len(points):
        layers.append(pdk.Layer("ScatterplotLayer", points, get_position="[lon, lat]", get_fill_color="color",
                                get_radius=6000, pickable=True, stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=1))
    view = pdk.ViewState(latitude=36.2, longitude=42.6, zoom=7)
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, map_style="dark",
                             tooltip={"text": "{label}\nSQM {sqm} · Bortle {bortle}"}), key=key)
    st.caption("Colour = zenith sky brightness. White/yellow ≈ 17–18 mag/arcsec² (Bortle 8–9, city) · "
               "red/purple ≈ 20–21 (suburban/rural) · transparent ≈ 22 (pristine).")


emission, SQM = base_maps()
model, metrics = model_bundle()

st.title("🔭 SkyQ Mosul")
st.markdown("**Light pollution & astronomical observation-quality prototype** · التلوث الضوئي وجودة الرصد الفلكي — "
            "NASA VIIRS night lights + live weather/aerosol forecast + ML → *where* and *when* to observe, "
            "and *how much* responsible lighting would recover the sky.")

with st.sidebar:
    st.header("Observing site")
    name = st.selectbox("Site", list(SITES), index=0)
    if SITES[name] is None:
        lat = st.number_input("Latitude", value=36.34, min_value=skyq.NINEVEH[0], max_value=skyq.NINEVEH[2], format="%.3f")
        lon = st.number_input("Longitude", value=43.13, min_value=skyq.NINEVEH[1], max_value=skyq.NINEVEH[3], format="%.3f")
    else:
        lat, lon = SITES[name]
    model_sqm = skyq.site_sqm(SQM, lat, lon)
    measured = st.number_input("Measured SQM at site (optional field reading)", value=0.0, min_value=0.0, max_value=22.5,
                               step=0.1, help="From an SQM meter or Globe at Night. Overrides the satellite estimate.")
    site_sqm = measured if measured > 0 else model_sqm
    st.metric("Sky brightness (mag/arcsec²)", f"{site_sqm:.2f}", f"Bortle {skyq.bortle(site_sqm)}", delta_color="off")
    if measured > 0:
        st.caption(f"Satellite model says {model_sqm:.2f} → model error {model_sqm - measured:+.2f} mag. "
                   "Log several of these to recalibrate `GLOW_SCALE`.")

tab_eng, tab_map, tab_plan, tab_fix, tab_model = st.tabs(
    ["🧭 Mosul decision engine", "🗺️ Light-pollution map & site finder", "🌙 Tonight's observation planner", "💡 Responsible-lighting scenario", "🤖 Model & data"])

# ------------------------------------------------------------------ map
@st.cache_resource(show_spinner="Loading Nineveh cell table (VNP46A2 + AI + physics + elevation + roads)…")
def engine_cells():
    import engine
    return engine.cells()


@st.cache_data(show_spinner="Comparing 2016 Black Marble with recent VNP46A2 lights…")
def lights_then_now():
    import engine
    import skyclass
    old = skyclass.crop(skyclass.emission(skyclass.fetch_region()), skyclass.IRAQ_WORK)
    new = engine.recent_emission()
    rows = []
    for n, (a, o) in ((k, v) for k, v in SITES.items() if v):
        r, c = int((skyclass.IRAQ_WORK[2] - a) / skyq.RES), int((o - skyclass.IRAQ_WORK[1]) / skyq.RES)
        rows.append(dict(site=n, light_2016=round(float(old[r, c]), 3), light_2026=round(float(new[r, c]), 3)))
    return pd.DataFrame(rows)


with tab_eng:
    import engine
    ENG_CLS = ["Poor", "Moderate", "Good", "Excellent"]
    st.markdown("**Place + date/time + observation type → Sky Score (0–100) → class → best time → ranked reachable spots.** "
                "Same engine as the public page and the REST API (`engine.py`, `api.py`). Site = sidebar selection.")
    e1, e2, e3, e4 = st.columns(4)
    e_date = e1.date_input("Date", value=date.today(), key="eng_date")
    e_hour = e2.selectbox("Local time", [f"{h:02d}:00" for h in (19, 20, 21, 22, 23, 0, 1, 2, 3, 4)], index=3, key="eng_hour")
    e_type = e3.selectbox("Observation type", list(engine.TYPES), key="eng_type")
    e_rad = e4.select_slider("Max distance (km)", [25, 50, 100, 150], value=100, key="eng_rad")
    h = int(e_hour[:2])
    e_when = datetime.combine(e_date + timedelta(days=1 if h < 12 else 0), datetime.min.time()) + timedelta(hours=h)
    with st.spinner("Running the decision engine…"):
        ea = engine.analyze(lat, lon, e_when, e_type)
        erec, enotes = engine.recommend(lat, lon, e_rad, e_when, e_type)
    for n in set(ea["notes"] + enotes):
        st.warning(f"{n} forecast unavailable for this time → Iraqi climatology used")

    st.subheader(f"{name}: {ea['score']}/100 — {ENG_CLS[ea['cls']]}")
    w = engine.TYPES[e_type][1]
    comp = pd.DataFrame(dict(component=["light pollution", "weather", "moon", "elevation"],
                             score=[ea["components"][k] for k in ("light", "weather", "moon", "elevation")], weight=w))
    comp["weighted"] = (comp.score * comp.weight).round(1)
    c1, c2 = st.columns([3, 2])
    with c1:
        st.dataframe(comp, hide_index=True, width="stretch")
        raw = comp.weighted.sum()
        gate = ("daylight/twilight cap (≤10)" if ea["score"] <= 10 and raw > 10 else
                "overcast cap (≤20, cloud ≥80%)" if ea["cloud_cover"] >= 80 and raw > 20 else "none")
        st.caption(f"Weighted sum {raw:.1f} → gates: **{gate}** → final **{ea['score']}**. "
                   f"Cloud {ea['cloud_cover']}% · humidity {ea['humidity']}% · AOD {ea['aod']} · elevation {ea['elevation_m']} m · "
                   f"{ea['road_km']} km to a major road · best window: "
                   + (f"{ea['best_time'][0]:%H:%M}–{ea['best_time'][1]:%H:%M}" if ea["best_time"] else "none"))
        st.markdown("**Suitable for:** " + " · ".join(f"{'✅' if v else '❌'} {k}" for k, v in ea["suitable"].items()))
    with c2:
        hh = ea["hourly"].assign(hour=lambda d: d.time.dt.strftime("%H:%M"))
        st.altair_chart(alt.Chart(hh).mark_bar().encode(x=alt.X("hour:N", sort=None), y=alt.Y("score:Q", scale=alt.Scale(domain=[0, 100])),
                        color=alt.condition(alt.datum.score >= 51, alt.value("#2ca02c"), alt.value("#d62728"))
                        ).properties(height=220, title="Hourly Sky Score (≥51 = observe)"), width="stretch")

    st.markdown("**Weights by observation type** (light / weather / moon / elevation)")
    st.dataframe(pd.DataFrame({k: dict(zip(["light", "weather", "moon", "elevation"], v[1])) for k, v in engine.TYPES.items()}).T,
                 width="stretch")

    st.subheader(f"Ranked spots ≤ {e_rad} km, ≤ 5 km from a major road")
    if len(erec):
        rc = erec[["lat", "lon", "km", "road_km", "elevation_m", "ai_score", "physics_score", "lp_score", "cloud_cover",
                   "humidity", "aod", "light", "weather", "moon", "elevation", "score", "cls"]].copy()
        rc["cls"] = [ENG_CLS[int(c)] for c in rc.cls]
        st.dataframe(rc.round(1), hide_index=True, width="stretch")
        st.caption(f"Best spot beats {name} by {int(erec.score.iloc[0]) - ea['score']:+d} points.")
    else:
        st.info("No reachable candidate within this distance.")

    st.subheader("Light-pollution layer: AI vs physics vs hybrid")
    cells_t = engine_cells()
    layer = st.radio("Map layer", ["lp_score", "ai_score", "physics_score"], horizontal=True,
                     format_func={"lp_score": "Hybrid (used)", "ai_score": "AI only (XGBoost)", "physics_score": "Physics only"}.get)
    cl = cells_t.assign(c=engine.to_class(cells_t[layer]))
    cl["color"] = [[[220, 40, 40], [245, 150, 20], [40, 180, 80], [70, 110, 255]][int(c)] + [160] for c in cl.c]
    cl["label"] = [f"{ENG_CLS[int(c)]} · {v:.0f}" for c, v in zip(cl.c, cl[layer])]
    st.pydeck_chart(pdk.Deck(layers=[pdk.Layer("ScatterplotLayer", cl, get_position="[lon, lat]", get_fill_color="color",
                                               get_radius=2100, pickable=True)],
                             initial_view_state=pdk.ViewState(latitude=36.2, longitude=42.6, zoom=7), map_style="dark",
                             tooltip={"text": "{label}"}), key="eng_map", height=380)
    near = [engine.nearest(cells_t, a_, o_) for a_, o_ in [v for v in SITES.values() if v]]
    sites_t = pd.DataFrame([dict(site=n, ai=round(r.ai_score), physics=round(r.physics_score), sqm=round(r.sqm, 2),
                                 hybrid=round(r.lp_score), cls=ENG_CLS[int(r.lp_cls)], elevation_m=int(r.elevation_m),
                                 road_km=round(r.road_km, 1)) for n, r in zip([k for k, v in SITES.items() if v], near)])
    st.dataframe(sites_t.merge(lights_then_now(), on="site"), hide_index=True, width="stretch")
    st.caption("light_2016 = Black Marble 2016 composite (training scale); light_2026 = median of recent NASA VNP46A2 daily "
               "images mapped to the same scale. Towns that were dark in 2016 (ISIS-held / displaced, e.g. Hamdaniya, Tel Afar) are fully "
               "lit in 2026, so the engine uses the recent data. "
               "Hybrid = mean(AI, physics) because the AI alone under-rates mid-size city cores (Macro-F1 0.38).")
    st.markdown("**REST API** — `uvicorn api:app --port 8000` → `/sky-quality`, `/best-locations`, `/pollution-map`, "
                "`/observation-conditions` (docs at `/docs`).")

with tab_map:
    c1, c2 = st.columns([2, 1])
    with c2:
        st.subheader("Darker sites nearby")
        radius = st.slider("Max driving distance (km)", 20, 400, 150, 10)
        cands = skyq.darkest_sites(SQM, lat, lon, radius, n=20)
        cands = cands[[skyq.in_nineveh(a, o) for a, o in zip(cands.lat, cands.lon)]].head(5).reset_index(drop=True)
        st.dataframe(cands, hide_index=True, width="stretch")
        if len(cands):
            best = cands.iloc[0]
            gain = best.sqm - site_sqm
            st.success(f"Best site is **{int(best.km_from_you)} km** away at SQM **{best.sqm}** (Bortle {int(best.bortle)}) — "
                       f"**{gain:+.1f} mag/arcsec²**, i.e. the sky is **{10 ** (0.4 * gain):.0f}× darker** than at {name}.")
    with c1:
        pts = pd.concat([
            pd.DataFrame([dict(lat=lat, lon=lon, label=name, sqm=round(site_sqm, 2), bortle=skyq.bortle(site_sqm), color=[0, 200, 255])]),
            cands.assign(label="candidate site", color=[[0, 255, 120]] * len(cands)),
        ])
        sky_map(SQM, pts, "map1")

# ------------------------------------------------------------------ planner
def hourly_planner(night, target, aperture, threshold, radec):
    with st.spinner("Fetching Open-Meteo weather + CAMS aerosol forecast and computing ephemerides…"):
        df, rec, notes = skyq.plan_night(model, lat, lon, night, target, site_sqm, aperture, radec, threshold)
    for n in notes:
        st.warning(n)

    k1, k2, k3, k4 = st.columns(4)
    if rec:
        k1.metric("Recommended window", f"{rec['start']:%H}–{rec['end']:%H}h")
        k2.metric("Window quality", f"{rec['mean_score']:.0%}", f"{rec['mean_score'] - rec['night_mean']:+.0%} vs random hour")
        st.success(f"✅ **OBSERVE** {target} from **{rec['start']:%a %H:%M}** to **{rec['end']:%a %H:%M}** (local time).")
    else:
        k1.metric("Recommended window", "none")
        k2.metric("Best hour", f"{df.score.max():.0%}")
        st.error(f"❌ **DON'T GO** — no hour reaches {threshold:.0%} for {target} this night. "
                 "Try another night, a darker site (map tab) or a brighter target.")
    k3.metric("Moon illumination", f"{df.moon_illum.mean():.0%}")
    k4.metric("Mean seeing (est.)", f"{df.seeing_arcsec.mean():.1f}″")

    df["hour"] = df.local_time.dt.strftime("%a %H:%M")
    df["verdict"] = np.where(df.score >= threshold, "good", np.where(df.score >= threshold / 2, "marginal", "poor"))
    bars = alt.Chart(df).mark_bar().encode(
        x=alt.X("hour:N", sort=None, title="Local time"), y=alt.Y("score:Q", scale=alt.Scale(domain=[0, 1]), title="Observation quality"),
        color=alt.Color("verdict:N", scale=alt.Scale(domain=["good", "marginal", "poor"], range=["#2ca02c", "#f0ad00", "#d62728"])),
        tooltip=["hour", alt.Tooltip("score:Q", format=".2f"), "cloud_cover", "aod", alt.Tooltip("target_alt:Q", format=".0f"),
                 alt.Tooltip("moon_alt:Q", format=".0f"), alt.Tooltip("sky_sqm_now:Q", format=".2f")])
    rule = alt.Chart(pd.DataFrame(dict(t=[threshold]))).mark_rule(strokeDash=[4, 4]).encode(y="t:Q")
    st.altair_chart(bars + rule, width="stretch")

    show = df[["hour", "score", "physics_score", "cloud_cover", "humidity", "aod", "visibility_km", "seeing_arcsec",
               "sun_alt", "moon_alt", "moon_illum", "target_alt", "airmass", "sky_sqm_now"]]
    st.dataframe(show.round(2), hide_index=True, width="stretch")
    st.caption("`score` = ML model · `physics_score` = transparent reference formula · `sky_sqm_now` = light pollution + moon + twilight.")

    with st.expander("📝 Log a real observation (field measurement → improves the model)"):
        with st.form("log"):
            h = st.selectbox("Hour you observed", df.hour)
            rating = st.select_slider("How good was it really?", options=[0.0, 0.25, 0.5, 0.75, 1.0], value=0.75,
                                      format_func={0.0: "0 – nothing", 0.25: "1 – poor", 0.5: "2 – fair",
                                                   0.75: "3 – good", 1.0: "4 – excellent"}.get)
            if st.form_submit_button("Save observation"):
                skyq.log_observation({**df[df.hour == h].iloc[0].to_dict(), "quality": rating})
                st.success(f"Saved. {len(skyq.load_logs())} field logs — retrain in the Model tab.")



def hilal_panel(night):
    nxt = skyq.conjunctions(datetime.combine(night, datetime.min.time()) - timedelta(days=2), 32)[0]
    st.info(f"Next new moon (conjunction): **{nxt:%d %b %Y %H:%M} UTC** — crescent is usually first possible on the "
            f"evening of that day or the next. Showing the 3 evenings from {nxt:%d %b}.")
    with st.spinner("Computing Sun/Moon ephemerides at minute resolution + forecast at best time…"):
        hd, notes = skyq.plan_hilal(lat, lon, nxt.date(), 3)
    for n in notes:
        st.warning(n)
    hit = hd[hd.final_zone.isin(["A", "B", "C"])]
    if len(hit):
        h = hit.iloc[0]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("First visible evening", f"{h.date:%a %d %b}")
        k2.metric("Best time", f"{h.best_time:%H:%M}", f"sunset {h.sunset:%H:%M}", delta_color="off")
        k3.metric("Odeh zone (clear → with weather)", f"{h.zone} → {h.final_zone}", f"V = {h.odeh_v:+.2f}", delta_color="off")
        k4.metric("Moon age at sunset", f"{h.age_h:.0f} h", f"lag {h.lag_min:.0f} min", delta_color="off")
        st.success(f"🌙 **{h.final_verdict}.** Look **{h.moon_alt:.1f}° above the horizon at azimuth {h.moon_az:.0f}°** "
                   f"(west, {abs(h.daz):.1f}° {'left' if h.daz > 0 else 'right'} of where the Sun set) at **{h.best_time:%H:%M}**.")
    else:
        st.error("❌ Crescent **not visible** on any of these evenings from this site (geometry and/or weather).")
    show = hd.assign(date=hd.date.map(lambda d: f"{d:%a %d %b}"))
    for c in ["sunset", "moonset", "best_time"]:
        show[c] = show[c].map(lambda t: f"{t:%H:%M}" if pd.notna(t) else "–")
    cols = [c for c in ["date", "sunset", "moonset", "age_h", "lag_min", "best_time", "moon_alt", "arcv", "arcl", "daz",
                        "width_arcmin", "odeh_v", "zone", "yallop_q", "yallop_zone", "cloud_cover", "aod",
                        "extra_extinction_mag", "final_zone", "verdict"] if c in show]
    st.dataframe(show[cols].round(2), hide_index=True, width="stretch")
    st.caption("**zone** = clear-sky Odeh (2004) criterion, fitted on 737 crescent observations: "
               "A naked eye · B optical aid, maybe naked eye · C optical aid only · D not visible. "
               "**final_zone** downgrades it for forecast cloud and dust/aerosol extinction along the low line of sight. "
               "ARCV = Moon–Sun altitude difference, ARCL = elongation, W = crescent width (′), at best time "
               "(sunset + 4/9 × lag). Yallop q shown as a cross-check.")


with tab_plan:
    c1, c2, c3, c4 = st.columns(4)
    night = c1.date_input("Night of", value=date.today(), min_value=date.today() - timedelta(days=60),
                          max_value=date.today() + timedelta(days=30))
    target = c2.selectbox("Target", [skyq.HILAL] + list(skyq.CATALOG) + ["Custom RA/Dec…"], index=2)
    aperture = c3.selectbox("Instrument", [7, 50, 80, 150, 200, 300, 400], index=3,
                            format_func=lambda a: "Naked eye" if a == 7 else f"{a} mm telescope")
    threshold = c4.slider("'Good' threshold", 0.3, 0.9, 0.6, 0.05)
    radec = None
    if target.startswith("Custom"):
        d1, d2, d3 = st.columns(3)
        radec = (d1.number_input("RA (deg)", 0.0, 360.0, 83.8), d2.number_input("Dec (deg)", -90.0, 90.0, -5.4),
                 d3.number_input("Magnitude", -2.0, 16.0, 6.0))


    if target == skyq.HILAL:
        hilal_panel(night)
    else:
        hourly_planner(night, target, aperture, threshold, radec)

# ------------------------------------------------------------------ intervention
with tab_fix:
    st.markdown("Apply [DarkSky lighting principles](https://darksky.org/resources/guides-and-how-tos/lighting-principles/) "
                "to every light within a radius of the site and see how much sky is recovered.")
    c1, c2 = st.columns([1, 2])
    with c1:
        r_km = st.slider("Apply within (km of site)", 5, 150, 40, 5)
        shielded = st.slider("Fixtures fully shielded", 0, 100, 70, 5, format="%d%%") / 100
        dimming = st.slider("Dimming after curfew", 0, 90, 30, 5, format="%d%%") / 100
        warm = st.checkbox("Switch to warm ≤3000 K LEDs", True)
        em2 = skyq.intervention(emission, lat, lon, r_km, shielded, dimming, warm)
        SQM2 = skyq.sqm_map(em2)
        before, after = skyq.site_sqm(SQM, lat, lon), skyq.site_sqm(SQM2, lat, lon)
        st.metric("Site sky brightness", f"{after:.2f}", f"{after - before:+.2f} mag/arcsec²")
        st.metric("Bortle class", f"{skyq.bortle(before)} → {skyq.bortle(after)}")
        nelm_b, nelm_a = (float(skyq.limiting_mag(s, 7)) for s in (before, after))
        st.metric("Naked-eye limiting mag", f"{nelm_a:.1f}", f"{nelm_a - nelm_b:+.1f}")
        stars = skyq.visible_stars
        st.metric("Visible stars (est.)", f"{stars(nelm_a):,.0f}", f"{stars(nelm_a) - stars(nelm_b):+,.0f}")
        area = (SQM2 - SQM > 0.1).sum() * (skyq.RES * 111) ** 2 * np.cos(np.radians(33))
        st.caption(f"Sky improved by >0.1 mag over ≈ {area:,.0f} km².")
    with c2:
        sky_map(SQM2, pd.DataFrame([dict(lat=lat, lon=lon, label=name, sqm=round(after, 2), bortle=skyq.bortle(after),
                                         color=[0, 200, 255])]), "map2")

# ------------------------------------------------------------------ model
@st.cache_data(show_spinner="Validating light-pollution map against Globe at Night…")
def lp_validation():
    cal = skyq.calibrate_glow_scale(emission)
    res, pts = skyq.validate_light_pollution(emission)
    return cal, res, pts


@st.cache_data(show_spinner="Checking hilal criterion on historical crescents…")
def hilal_validation():
    return skyq.validate_hilal()


def confusion_chart(cm, labels, title):
    d = pd.DataFrame([dict(actual=labels[i], predicted=labels[j], n=v) for i, r in enumerate(cm) for j, v in enumerate(r)])
    base = alt.Chart(d, title=title).encode(x=alt.X("predicted:N", sort=labels), y=alt.Y("actual:N", sort=labels))
    dark = alt.condition(alt.datum.n > max(map(max, cm)) / 2, alt.value("white"), alt.value("black"))
    return (base.mark_rect().encode(color=alt.Color("n:Q", scale=alt.Scale(scheme="blues"), legend=None))
            + base.mark_text(fontSize=14).encode(text="n:Q", color=dark))


with tab_model:
    cal, lp, lp_pts = lp_validation()
    hv = hilal_validation()
    report = dict(observation_quality_model=metrics, light_pollution_calibration=cal, light_pollution_validation=lp,
                  hilal_cases=hv.to_dict("records"))
    top = st.columns([3, 1, 1])
    top[0].markdown(f"Trained **{metrics['trained']}** · {metrics['n_train']:,} train / {metrics['n_test']:,} held-out samples · "
                    f"real field logs: **{metrics['real_logs']}**")
    if top[1].button("🔁 Retrain with field logs"):
        model_bundle.clear()
        skyq.train()
        st.rerun()
    top[2].download_button("⬇️ Report (JSON)", json.dumps(report, indent=2, default=str), "skyq_model_report.json",
                           "application/json")

    s0, s1, s2, s3, s4 = st.tabs(["Sky-quality classifier (AI)", "Observation-quality model",
                                  "Light-pollution map vs field data", "Hilal criterion", "Data sources & limits"])
    with s0:
        import skyclass
        _, cr = skyclass.load()
        st.markdown(f"**Task:** NASA VIIRS night-light features → ground sky class (Poor/Moderate/Good/Excellent). "
                    f"**Ground truth:** {cr['n']:,} Globe at Night observations 2012–2021 across the Mediterranean & Middle East "
                    f"({', '.join(f'{k}: {v:,}' for k, v in cr['sources'].items())}). **Validation:** spatial GroupKFold on "
                    f"{cr['n_blocks']} 2°×2° blocks (unseen regions), plus an Iraq hold-out ({cr['iraq_points']} points; too few field records exist in Nineveh alone for a Mosul-only test). "
                    f"**Selected:** {cr['best']} (best Macro-F1).")
        rows = {**cr["models"], **cr["baselines"]}
        tab = pd.DataFrame({k: {m: v for m, v in d.items() if m != "iraq_holdout"} for k, d in rows.items()}).T
        tab["iraq_holdout_macro_f1"] = [d.get("iraq_holdout", {}).get("macro_f1") for d in rows.values()]
        st.dataframe(tab[["macro_f1", "macro_f1_fold_std", "accuracy", "balanced_accuracy", "weighted_f1",
                          "within_one_class", "quadratic_kappa", "iraq_holdout_macro_f1"]], width="stretch")
        c1, c2 = st.columns(2)
        c1.altair_chart(confusion_chart(cr["confusion"], skyclass.CLASSES, f"Confusion – {cr['best']} (spatial CV)"), width="stretch")
        c2.markdown("**Per-class (spatial CV)**")
        c2.dataframe(pd.DataFrame(cr["per_class"]).T, width="stretch")
        c2.markdown("**Class counts**: " + ", ".join(f"{k} {v:,}" for k, v in cr["class_counts"].items()))
        fi = pd.DataFrame(dict(feature=list(cr["feature_importance"]), importance=list(cr["feature_importance"].values())))
        st.altair_chart(alt.Chart(fi).mark_bar().encode(x="importance:Q", y=alt.Y("feature:N", sort="-x")), width="stretch")
        st.warning("Honest result: Macro-F1 0.38 on unseen regions, far below the 0.80 target, and no better than the physics-only rule (0.38); ~79% of predictions "
                   "are within one class. The limits are the inputs, not the model choice (LR/RF/XGBoost all land "
                   "0.36–0.38): the free Black Marble layer is a 2016 visual composite that saturates in cities, and citizen "
                   "naked-eye labels are noisy. Next step: VNP46A2 radiance (Earthdata) + more SQM-meter readings in Iraq, "
                   "which the in-app 'contribute' form collects.")
    with s1:
        b, r, t3, cv = metrics["binary"], metrics["regression"], metrics["three_class"], metrics["cv"]
        st.markdown("**Decision: GOOD night (score ≥ 0.6) vs not** — held-out test set")
        cols = st.columns(6)
        for col, (k, v) in zip(cols, [("Accuracy", b["accuracy"]), ("Precision", b["precision"]), ("Recall", b["recall"]),
                                      ("F1", b["f1"]), ("ROC-AUC", b.get("roc_auc", float("nan"))),
                                      ("CV F1 (5-fold)", f"{cv['f1_mean']:.2f}±{cv['f1_std']:.2f}")]):
            col.metric(k, v if isinstance(v, str) else f"{v:.3f}")
        st.markdown("**Score regression & 3-class (poor / fair / good)**")
        cols = st.columns(6)
        for col, (k, v) in zip(cols, [("MAE", r["mae"]), ("RMSE", r["rmse"]), ("R²", r["r2"]),
                                      ("3-class accuracy", t3["accuracy"]), ("Macro-F1", t3["macro_f1"]),
                                      ("Weighted-F1", t3["weighted_f1"])]):
            col.metric(k, f"{v:.3f}")
        c1, c2 = st.columns(2)
        c1.altair_chart(confusion_chart(b["confusion"], ["not good", "good"], "Confusion – GOOD vs not"), width="stretch")
        c2.altair_chart(confusion_chart(t3["confusion"], skyq.CLASSES, "Confusion – 3 classes"), width="stretch")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Per-class report**")
            st.dataframe(pd.DataFrame(t3["per_class"]).T.assign(share=pd.Series(metrics["class_balance"])), width="stretch")
            st.markdown("**vs. today's rules of thumb** (same test set)")
            st.dataframe(pd.DataFrame(metrics["baselines"]).T[["accuracy", "precision", "recall", "f1"]], width="stretch")
        with c2:
            st.markdown("**Feature importance** (what drives observing quality)")
            imp = pd.DataFrame(dict(feature=skyq.FEATURES, importance=model.feature_importances_)).sort_values("importance")
            st.altair_chart(alt.Chart(imp).mark_bar().encode(x="importance:Q", y=alt.Y("feature:N", sort=None)), width="stretch")
        if "real_logs_in_sample" in metrics:
            st.markdown("**Agreement with real observer logs** (in-sample)")
            st.json(metrics["real_logs_in_sample"])
        st.warning("These scores measure how well the Random Forest reproduces the **physics-based label** on unseen "
                   "simulated nights (label noise σ = 0.03 caps them). They are not yet accuracy against real observer "
                   "judgments — that comes from the field-log loop in the planner tab.")
    with s2:
        st.markdown(f"Independent ground truth: **{lp['n']} Globe at Night citizen observations** (2009–2025) inside the map "
                    "region (Iraq, Kuwait, W. Iran, SE Turkey…), clear or ≤¼ cloud. Only 2 records fall inside Nineveh, so Mosul is validated "
                    "with the surrounding region. Their naked-eye limiting magnitude is "
                    "compared with the model's at the same pixel.")
        cols = st.columns(5)
        cols[0].metric("MAE (limiting mag)", lp["mae"])
        cols[1].metric("Bias", f"{lp['bias']:+.2f}")
        cols[2].metric("Within ±1 mag", f"{lp['within_1mag']:.0%}")
        cols[3].metric("Spearman ρ", lp["spearman"])
        cols[4].metric("Leave-one-out MAE", cal["loo_mae"], f"{cal['loo_mae'] - cal['uncalibrated_mae']:+.2f} vs uncalibrated",
                       delta_color="inverse")
        cols = st.columns(5)
        cols[0].metric("Dark-sky (NELM≥4) accuracy", lp["dark_sky_accuracy"])
        cols[1].metric("Precision", lp["dark_sky_precision"])
        cols[2].metric("Recall", lp["dark_sky_recall"])
        cols[3].metric("F1", lp["dark_sky_f1"])
        if "sqm_meter" in lp:
            cols[4].metric(f"SQM-meter MAE (n={lp['sqm_meter']['n']})", lp["sqm_meter"]["mae"],
                           f"bias {lp['sqm_meter']['bias']:+.2f}", delta_color="off")
        c1, c2 = st.columns([3, 2])
        pts = lp_pts.assign(jitter=lp_pts.LimitingMag + np.random.default_rng(0).uniform(-0.15, 0.15, len(lp_pts)))
        sc = alt.Chart(pts).mark_circle(size=70, opacity=0.8).encode(
            x=alt.X("jitter:Q", title="Observed limiting mag (Globe at Night)", scale=alt.Scale(domain=[-0.5, 7.5])),
            y=alt.Y("model_nelm:Q", title="Model limiting mag", scale=alt.Scale(domain=[-0.5, 7.5])),
            color="Country:N", tooltip=["UTDate", "Country", "LimitingMag", alt.Tooltip("model_nelm:Q", format=".1f")])
        diag = alt.Chart(pd.DataFrame(dict(x=[0, 7], y=[0, 7]))).mark_line(strokeDash=[4, 4], color="gray").encode(x="x", y="y")
        c1.altair_chart(sc + diag, width="stretch")
        c2.markdown(f"""**Calibration.** The single free parameter (`GLOW_SCALE`) was fitted on these points:
{cal['uncalibrated_scale']:.0f} (2-anchor guess) → **{cal['fitted_scale']:.0f}**; out-of-sample MAE {cal['uncalibrated_mae']} → **{cal['loo_mae']}** mag.
The SQM-meter readings (never used for fitting) independently show near-zero bias.

**Why it isn't tighter:** Globe at Night magnitudes are integer chart-matches by untrained eyes on nights with
unknown haze/moon, and several points share one pixel. Spearman ρ = {lp['spearman']} → the map ranks sites correctly more
often than not, but ±1.5 mag per single reading is the honest error bar.""")
        st.dataframe(lp_pts.round(2), hide_index=True, width="stretch")
    with s3:
        st.markdown("Crescent visibility uses the **Odeh (2004)** criterion (fitted by Odeh on 737 crescent observations; used "
                    "by ICOP) at Yallop's best time, with **Yallop (1997) q** as a cross-check. Below: clear-cut historical "
                    "evenings, computed for Mosul.")
        st.metric("Historical cases correct", f"{int(hv.correct.sum())} / {len(hv)}")
        st.dataframe(hv, hide_index=True, width="stretch")
        st.caption("Sanity checks, not a statistical validation — Odeh's published fit is the criterion's validation. "
                   "The weather downgrade (cloud, dust extinction) is our addition and should be calibrated against Iraqi "
                   "sighting reports.")
    with s4:
        st.markdown("""
| Input | Source | Unit | Resolution / accuracy |
|---|---|---|---|
| Night lights | NASA VIIRS **Black Marble** (GIBS WMS, 2016 composite) | relative radiance | ~500 m, resampled to 2 km; city cores saturate |
| Sky brightness | Walker-law (d^-2.5) convolution of night lights, 200 km radius, fitted to Globe at Night | mag/arcsec² (SQM), Bortle | ±1.5 mag per point (LOO) |
| Field truth | **Globe at Night** 2006–2025 (45 usable points), SQM readings, observer logs | limiting mag, mag/arcsec², 0–4 | citizen science |
| Cloud, humidity, wind 10 m & 250 hPa, visibility | Open-Meteo forecast (≤16 days) | %, %, km/h, km | hourly, ~10 km |
| Aerosol optical depth (dust) | Open-Meteo / Copernicus CAMS (≤5 days) | AOD @550 nm | hourly, ~40 km |
| Sun, Moon, target, conjunctions | Astropy | deg, airmass | < 1′ |
| Hilal | Odeh (2004), Yallop (1997) | V, q, zones A–D | criterion-level |

**Limits of validity**
- Observation-quality labels are **physics-generated**; field logs (×50 weight) progressively replace them.
- Seeing is a jet-stream-wind proxy, not a DIMM. Moonlight/twilight are compact empirical fits.
- Black Marble is a 2016 composite — newer lighting is missing. Upgrade: VNP46A2 monthly radiance (Earthdata).
- Beyond the forecast horizon the planner uses climatology and says so.
""")
