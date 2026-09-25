"""Public page (Iraqi Arabic, beginner-friendly) — Iraq Sky AI, Mosul.
Three questions (where / when / what) -> one big answer (go or not, why, when, where instead)."""
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

import engine
import skyq

# ------------------------------------------------------------------ look & feel
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;800&display=swap');
[data-testid="stMainBlockContainer"], .stMarkdown, .stSelectbox, .stTextInput, .stAlert, .stRadio, .stDateInput, [data-testid="stMetric"],
[data-testid="stExpander"], [data-testid="stSegmentedControl"], .stTabs, .stForm { direction: rtl; text-align: right;
  font-family: 'Cairo', sans-serif; }
h1, h2, h3, h4 { font-family: 'Cairo', sans-serif !important; }
.block-container { max-width: 900px; }
[data-testid="stSlider"] { direction: ltr; }  /* sliders break in RTL */
.flag { height: 6px; border-radius: 3px; margin: 2px 0 10px;
  background: linear-gradient(90deg, #CE1126 33%, #ffffff 33% 66%, #000000 66%); }
.step { font-weight: 800; font-size: 1.15rem; margin: 2px 0 6px; }
.step span { display: inline-block; width: 30px; height: 30px; line-height: 30px; text-align: center; border-radius: 50%;
  background: #466eff; color: #fff; margin-left: 8px; }
.hint { opacity: .75; font-size: .92rem; margin-top: -4px; }
.hero { display: flex; gap: 22px; align-items: center; border-radius: 20px; padding: 20px 22px; margin: 10px 0;
  border: 2px solid var(--c); background: linear-gradient(135deg, color-mix(in srgb, var(--c) 22%, transparent), transparent); }
.ring { flex: 0 0 132px; height: 132px; border-radius: 50%; display: grid; place-items: center;
  background: conic-gradient(var(--c) calc(var(--s) * 1%), rgba(128,128,128,.18) 0); }
.ring .in { width: 104px; height: 104px; border-radius: 50%; background: var(--bg, #0b1020); display: grid; place-items: center;
  text-align: center; line-height: 1.1; }
.ring b { font-size: 2.3rem; } .ring small { display: block; opacity: .7; }
.verdict { font-size: 1.9rem; font-weight: 800; margin: 0 0 4px; }
.why li { margin: 2px 0; }
.chips { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 8px 0 4px; }
.chip { border-radius: 14px; padding: 10px 8px; text-align: center; border: 1px solid rgba(128,128,128,.25);
  border-top: 5px solid var(--c); }
.chip .e { font-size: 1.7rem; } .chip .t { font-size: .85rem; opacity: .8; } .chip .v { font-weight: 800; }
.strip { display: grid; grid-template-columns: repeat(13, 1fr); gap: 4px; margin: 6px 0; direction: rtl; }
.hr { border-radius: 8px; padding: 6px 0; text-align: center; font-size: .8rem; background: var(--c); color: #111; }
.hr.sel { outline: 3px solid #fff; } .hr b { display: block; font-size: 1.05rem; }
.spot { border-radius: 16px; padding: 14px 16px; margin: 8px 0; border: 1px solid rgba(128,128,128,.25);
  border-right: 6px solid var(--c); }
.spot .n { font-size: 1.2rem; font-weight: 800; }
.go { display: inline-block; margin-top: 6px; padding: 6px 14px; border-radius: 10px; background: #466eff; color: #fff !important;
  text-decoration: none !important; font-weight: 700; }
@media (max-width: 640px) { .hero { flex-direction: column; text-align: center; } .chips { grid-template-columns: repeat(2, 1fr); }
  .strip { grid-template-columns: repeat(7, 1fr); } }
</style>""", unsafe_allow_html=True)

PLACES = {  # الموصل ونينوى فقط
    "الموصل – مركز المدينة": (36.340, 43.130), "جامعة الموصل": (36.380, 43.140),
    "الحمدانية (قره قوش)": (36.270, 43.375), "بعشيقة": (36.455, 43.350), "تلكيف": (36.490, 43.120),
    "القيارة": (35.803, 43.290), "تلعفر": (36.374, 42.449), "سنجار": (36.321, 41.877), "ربيعة": (36.800, 42.100),
    "الحضر": (35.588, 42.718),
}
AR_NAMES = {
    "Sirius": "الشِّعرى اليمانية", "Vega": "النسر الواقع", "Altair": "النسر الطائر", "Deneb": "ذنب الدجاجة",
    "Betelgeuse": "منكب الجوزاء", "Polaris": "النجم القطبي (الجدي)", "Pleiades (M45)": "الثريّا",
    "Orion Nebula (M42)": "سديم الجبّار", "Andromeda Galaxy (M31)": "مجرة المرأة المسلسلة",
    "Hercules Cluster (M13)": "عنقود هرقل", "Ring Nebula (M57)": "سديم الحلقة", "Whirlpool Galaxy (M51)": "مجرة الدوّامة",
    "Jupiter": "المشتري 🪐", "Saturn": "زحل 🪐", "Mars": "المريخ 🔴",
}
DAYS = ["الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت", "الأحد"]
DIRS = ["شمال", "شمال شرق", "شرق", "جنوب شرق", "جنوب", "جنوب غرب", "غرب", "شمال غرب"]
HILAL_AR = {"A": "يبين بالعين 👁️", "B": "يبين بالناظور، وممكن بالعين 🔭", "C": "بس بالتلسكوب 🔭", "D": "ما يبين ❌"}
LEVEL = [  # score class -> (color, verdict, short)
    ("#e5484d", "😞 لا، مو هسه", "ما تصلح"),
    ("#f5a524", "😐 متوسطة", "متوسطة"),
    ("#30a46c", "😊 زينة، تستاهل!", "زينة"),
    ("#5b8cff", "🤩 ممتازة! روح ارصد", "ممتازة"),
]
TYPE_UI = {  # engine key -> (button label, one-line explanation)
    "stargazing": ("✨ نجوم", "تشوف النجوم والكواكب بعينك — أسهل شي وتصلح للبداية."),
    "milky_way": ("🌌 درب التبانة", "الشريط الأبيض الضبابي بالسما — يحتاج مكان مظلم كلش وبلا قمر."),
    "astrophotography": ("📷 تصوير", "تصوّر السما بالموبايل أو الكاميرا — يحتاج سما صافية ومظلمة."),
    "meteors": ("☄️ شهب", "نجوم تركض بسرعة بالسما — يحتاج صبر وسما مظلمة شوية."),
}
HOUR_LABEL = {19: "7 المسا", 20: "8 بالليل", 21: "9 بالليل", 22: "10 بالليل", 23: "11 بالليل", 0: "12 نص الليل",
              1: "1 بعد نص الليل", 2: "2 بعد نص الليل", 3: "3 الفجر", 4: "4 الفجر"}
SAFE = "#30a46c"


def compass(az):
    return DIRS[int((az + 22.5) % 360 // 45)]


def bearing(a1, o1, a2, o2):
    p1, p2, dl = np.radians(a1), np.radians(a2), np.radians(o2 - o1)
    return np.degrees(np.arctan2(np.sin(dl) * np.cos(p2), np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl))) % 360


def place_name(lat, lon):
    """Human name for any point: 'قرب بعشيقة' or '18 كم شمال شرق بعشيقة'."""
    n, (a, o) = min(PLACES.items(), key=lambda kv: skyq.haversine_km(lat, lon, *kv[1]))
    d = skyq.haversine_km(a, o, lat, lon)
    return f"قرب {n}" if d < 8 else f"{d:.0f} كم {compass(bearing(a, o, lat, lon))} {n}"


def reasons(a):
    """Plain-language 'why' bullets from the engine result."""
    c, out = a["components"], []
    if a["hourly"] is not None and a["score"] <= 10 and c["weather"] > 10:
        out.append("☀️ الشمس بعدها ما غابت، أو بعده الضوّ — لازم تنتظر الظلام")
    out.append("💡 أضواء المدينة قوية هنا وتغطي النجوم الخافتة" if c["light"] < 35 else
               "💡 أكو شوية أضواء مدن قريبة" if c["light"] < 65 else "💡 المكان بعيد عن أضواء المدن 👍")
    out.append(f"☁️ السما مغيّمة ({a['cloud_cover']}%)" if a["cloud_cover"] >= 50 else
               f"⛅ أكو شوية غيوم ({a['cloud_cover']}%)" if a["cloud_cover"] >= 20 else "☀️ السما صافية بلا غيوم 👍")
    if a["aod"] >= 0.5:
        out.append("🌫️ أكو غبار (عجاج) بالجو يخلي السما باهتة")
    out.append("🌕 القمر مضوي ويطمس النجوم الخافتة" if c["moon"] < 50 else
               "🌓 القمر شوية مأثر" if c["moon"] < 80 else "🌑 القمر مو مأثر 👍")
    return out


# ------------------------------------------------------------------ cached calls
@st.cache_resource(show_spinner="⏳ دا نجهّز خريطة الموصل…")
def cell_table():
    t = engine.cells()
    t["color"] = [[[229, 72, 77], [245, 165, 36], [48, 164, 108], [91, 140, 255]][int(c)] + [140] for c in t.lp_cls]
    t["label"] = [f"{['سما مضوية 🔴', 'سما نص ونص 🟠', 'سما مظلمة 🟢', 'سما مظلمة كلش ⭐'][int(c)]}" for c in t.lp_cls]
    return t


@st.cache_data(ttl=1800, show_spinner="⏳ دا نفحص السما، الغيوم، الغبار والقمر…")
def analyze(lat, lon, when, obs):
    return engine.analyze(lat, lon, when, obs)


@st.cache_data(ttl=1800, show_spinner="⏳ دا ندوّر على أحسن الأماكن حواليك…")
def recommend(lat, lon, radius, when, obs):
    return engine.recommend(lat, lon, radius, when, obs)


@st.cache_resource(show_spinner=False)
def obs_model():
    return skyq.load_model()[0]


@st.cache_data(show_spinner=False)
def sqm_grid():
    return skyq.sqm_map(skyq.emission_map(skyq.fetch_black_marble()))


@st.cache_data(ttl=3600, show_spinner="⏳ دا نحسب وين النجوم والكواكب…")
def targets(lat, lon, night, sqm):
    model, wx, rows = obs_model(), skyq.weather(lat, lon, night), []
    for t, ra_dec in skyq.CATALOG.items():
        mag = skyq.PLANET_MAG[t] if ra_dec is None else ra_dec[2]
        ap = 7 if mag <= 4.5 else 150
        df, rec, _ = skyq.plan_night(model, lat, lon, night, t, sqm, ap, wx=wx)
        best = df.score.idxmax()
        rows.append(dict(target=t, aperture=ap, score=df.score.max(), rec=rec, alt=df.target_alt[best],
                         az=df.target_az[best], hour=df.local_time[best]))
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600, show_spinner="⏳ دا نحسب الهلال…")
def hilal(lat, lon, today):
    nxt = skyq.conjunctions(datetime.combine(today, datetime.min.time()) - timedelta(days=2), 32)[0]
    df, _ = skyq.plan_hilal(lat, lon, nxt.date(), 3)
    return nxt, df


def reset_map():
    st.session_state.map_v = st.session_state.get("map_v", 0) + 1


CELLS = cell_table()

# ------------------------------------------------------------------ header
st.markdown("# 🔭 سما الموصل")
st.markdown('<div class="flag"></div>', unsafe_allow_html=True)
st.markdown("#### جاوب على 3 أسئلة بسيطة، واحنه نگلك **تطلع ترصد لو لا، ومتى، ووين** 🌌")

# ------------------------------------------------------------------ Q1 where
with st.container(border=True):
    st.markdown('<div class="step"><span>1</span>وينك هسه؟</div>', unsafe_allow_html=True)
    place = st.selectbox("منطقتك", list(PLACES), label_visibility="collapsed", on_change=reset_map)
    lat, lon = PLACES[place]
    map_key = f"sky_map_{st.session_state.get('map_v', 0)}"
    picked = st.session_state.get(map_key, {}).get("selection", {}).get("objects", {}).get("cells", [])
    if picked:
        lat, lon = picked[0]["lat"], picked[0]["lon"]
        place = place_name(lat, lon)
        c1, c2 = st.columns([3, 1])
        c1.info(f"📍 اخترت نقطة من الخريطة: **{place}**")
        c2.button("↩️ رجوع للقائمة", on_click=reset_map, use_container_width=True)
    with st.expander("🗺️ أو اختار من الخريطة (اضغط على أي نقطة ملوّنة)"):
        st.markdown("كل نقطة ملوّنة = شكد السما **مظلمة** هناك: "
                    "🔴 مضوية (مدينة) · 🟠 نص ونص · 🟢 مظلمة · 🔵 مظلمة كلش ⭐ — **كل ما أظلم أحسن للنجوم**")
        st.pydeck_chart(pdk.Deck(
            layers=[pdk.Layer("ScatterplotLayer", CELLS, id="cells", get_position="[lon, lat]", get_fill_color="color",
                              get_radius=2100, pickable=True),
                    pdk.Layer("ScatterplotLayer", pd.DataFrame([dict(lat=lat, lon=lon, label="📍 انت هنا")]),
                              get_position="[lon, lat]", get_fill_color=[255, 255, 255], get_radius=3000, stroked=True,
                              get_line_color=[0, 0, 0], line_width_min_pixels=2, pickable=True)],
            initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=7.3), map_style="dark",
            tooltip={"text": "{label}"}), key=map_key, on_select="rerun", selection_mode="single-object", height=360)

# ------------------------------------------------------------------ Q2 when
with st.container(border=True):
    st.markdown('<div class="step"><span>2</span>شوكت تريد ترصد؟</div>', unsafe_allow_html=True)
    now = datetime.now()
    today = now.date() - timedelta(days=1) if now.hour < 6 else now.date()
    day = st.segmented_control("اليوم", ["الليلة", "باچر", "بعد باچر", "📅 يوم ثاني"], default="الليلة",
                               label_visibility="collapsed") or "الليلة"
    night = {"الليلة": today, "باچر": today + timedelta(days=1), "بعد باچر": today + timedelta(days=2)}.get(day)
    if night is None:
        night = st.date_input("اختار اليوم", value=today + timedelta(days=3), min_value=today, max_value=today + timedelta(days=14))
    hour = st.select_slider("الساعة", options=list(HOUR_LABEL), value=22, format_func=HOUR_LABEL.get)
    when = datetime.combine(night + timedelta(days=1 if hour < 12 else 0), datetime.min.time()) + timedelta(hours=hour)
    st.markdown(f'<div class="hint">🗓️ {DAYS[night.weekday()]} {night:%d/%m} — الساعة {HOUR_LABEL[hour]}</div>',
                unsafe_allow_html=True)

# ------------------------------------------------------------------ Q3 what
with st.container(border=True):
    st.markdown('<div class="step"><span>3</span>شتريد تشوف؟</div>', unsafe_allow_html=True)
    obs = st.segmented_control("نوع الرصد", list(TYPE_UI), default="stargazing", format_func=lambda k: TYPE_UI[k][0],
                               label_visibility="collapsed") or "stargazing"
    st.markdown(f'<div class="hint">{TYPE_UI[obs][1]}</div>', unsafe_allow_html=True)
    radius = st.segmented_control("شكد تگدر تبعد؟", [25, 50, 100], default=100,
                                  format_func={25: "🚶 قريب (25 كم)", 50: "🚗 مشوار (50 كم)", 100: "🚙 طلعة (100 كم)"}.get) or 100

# ------------------------------------------------------------------ THE ANSWER
a = analyze(lat, lon, when, obs)
rec, rec_notes = recommend(lat, lon, radius, when, obs)
k = a["cls"]
color, verdict, _ = LEVEL[k]
why = "".join(f"<li>{r}</li>" for r in reasons(a))
st.markdown(f"""<div class="hero" style="--c:{color}; --s:{a['score']}">
  <div class="ring"><div class="in"><div><b>{a['score']}</b><small>من 100</small></div></div></div>
  <div><div class="verdict">{verdict}</div>
  <div>{TYPE_UI[obs][0]} · {place} · {DAYS[night.weekday()]} {HOUR_LABEL[hour]}</div>
  <ul class="why">{why}</ul></div></div>""", unsafe_allow_html=True)

# next action in one sentence
bt = a["best_time"]
best = rec.iloc[0] if len(rec) else None
if best is not None and int(best.score) >= a["score"] + 10:
    st.success(f"👉 **نصيحتنا:** روح لـ **{place_name(best.lat, best.lon)}** ({best.km:.0f} كم) — درجتها هناك "
               f"**{int(best.score)}/100**. شوف الأماكن جوّه ⬇️")
elif bt:
    st.success(f"👉 **نصيحتنا:** أحسن وقت بهاي الليلة من **{bt[0]:%H:%M}** لـ **{bt[1]:%H:%M}**")
elif k <= 1:
    st.warning("👉 **نصيحتنا:** جرّب ليلة ثانية أو ساعة ثانية فوق، لأن هاي الليلة ما بيها وقت زين.")

# factor chips
c = a["components"]
chip = lambda e, t, v, s: (f'<div class="chip" style="--c:{"#30a46c" if s >= 70 else "#f5a524" if s >= 40 else "#e5484d"}">'
                           f'<div class="e">{e}</div><div class="t">{t}</div><div class="v">{v}</div></div>')
light_word = "قوية" if c["light"] < 35 else "متوسطة" if c["light"] < 65 else "قليلة"
moon_word = "مو مأثر" if c["moon"] >= 80 else "شوية" if c["moon"] >= 50 else "مضوي"
dust_s = max(0, 100 - (a["aod"] - 0.1) * 150)
dust_word = "صافي" if a["aod"] < 0.3 else "خفيف" if a["aod"] < 0.5 else "عجاج"
st.markdown('<div class="chips">' + chip("💡", "أضواء المدينة", light_word, c["light"])
            + chip("☁️", "الغيوم", f"{a['cloud_cover']}%", 100 - a["cloud_cover"])
            + chip("🌕", "القمر", moon_word, c["moon"]) + chip("🌫️", "الغبار", dust_word, dust_s) + "</div>",
            unsafe_allow_html=True)

# hour strip
st.markdown("##### 🕒 كل ساعات الليلة (اخضر = زين)")
cells_html = ""
for r in a["hourly"].itertuples():
    col = ["#e5484d", "#f5a524", "#30a46c", "#5b8cff"][int(r.cls)]
    mark = ["❌", "⚠️", "✅", "⭐"][int(r.cls)]
    sel = " sel" if r.time == when else ""
    cells_html += f'<div class="hr{sel}" style="--c:{col}"><b>{mark}</b>{r.time:%H}:00</div>'
st.markdown(f'<div class="strip">{cells_html}</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ where instead
st.markdown(f"### 🚗 وين أروح؟ أحسن أماكن ضمن {radius} كم")
st.caption("اخترناها لأن سماها مظلمة، والجو هناك زين بالوقت اللي اخترته، وقريبة من شارع رئيسي تگدر توصله بالسيارة.")
if len(rec):
    for i, r in enumerate(rec.itertuples(), 1):
        col, _, short = LEVEL[int(r.cls)]
        medal = ["🥇", "🥈", "🥉"][i - 1] if i <= 3 else f"{i}."
        st.markdown(f"""<div class="spot" style="--c:{col}"><div class="n">{medal} {place_name(r.lat, r.lon)} —
          {int(r.score)}/100 ({short})</div>
          🚗 {r.km:.0f} كم عنك · ☁️ غيوم {r.cloud_cover:.0f}% · 🛣️ {r.road_km:.1f} كم من الشارع · ⛰️ ارتفاع {r.elevation_m:.0f} م<br>
          <a class="go" target="_blank" href="https://www.google.com/maps/dir/?api=1&origin={lat},{lon}&destination={r.lat},{r.lon}">
          🧭 خذني هناك</a></div>""", unsafe_allow_html=True)
else:
    st.info("ماكو أماكن ضمن هاي المسافة قريبة من شارع. جرّب «🚙 طلعة (100 كم)».")

# ------------------------------------------------------------------ more (tabs)
st.markdown("### 🎒 أشياء زيادة")
t1, t2, t3, t4, t5 = st.tabs(["✨ شنو أشوف؟", "🌒 الهلال", "💡 خلّي سما مدينتك أحلى", "🙋 شاركنا", "📖 شنو يعني؟"])
SQM = sqm_grid()
sqm = skyq.site_sqm(SQM, lat, lon)
with t1:
    st.caption("النجوم والكواكب اللي تبين من مكانك بهاي الليلة، ووين تباوع.")
    tg = targets(lat, lon, night, sqm)
    vis = tg[tg.score >= skyq.FAIR].sort_values("score", ascending=False)
    if len(vis):
        for r in vis.itertuples():
            how = "👁️ بعينك" if r.aperture == 7 else "🔭 تحتاج تلسكوب صغير"
            clear = "واضح" if r.score >= skyq.GOOD else "خافت"
            w = f"{r.rec['start']:%H:%M}–{r.rec['end']:%H:%M}" if r.rec else f"{r.hour:%H:%M}"
            st.markdown(f'<div class="spot" style="--c:{SAFE}"><div class="n">{AR_NAMES[r.target]}</div>'
                        f'{how} · يبين {clear} · ⏰ {w} · 🧭 باوع لـ<b>{compass(r.az)}</b> وارفع راسك '
                        f'{"شوية" if r.alt < 30 else "لنص السما" if r.alt < 60 else "لفوق"}</div>', unsafe_allow_html=True)
    else:
        st.warning("من مكانك ما راح يبين شي واضح بهاي الليلة — روح لواحد من الأماكن فوق.")
with t2:
    nxt, hd = hilal(lat, lon, night)
    st.markdown(f"🌑 القمر الجديد يتولد **{DAYS[nxt.weekday()]} {nxt + timedelta(hours=3):%d/%m}** "
                f"الساعة {nxt + timedelta(hours=3):%H:%M}. بعدها بليلة أو ليلتين يبين الهلال بعد الغروب بالغرب:")
    for r in hd.itertuples():
        d = f"{DAYS[r.date.weekday()]} {r.date:%d/%m}"
        if r.final_zone in ("A", "B", "C"):
            st.markdown(f'<div class="spot" style="--c:{SAFE}"><div class="n">🌙 مساء {d}: {HILAL_AR[r.final_zone]}</div>'
                        f'⏰ الساعة {r.best_time:%H:%M} (بعد الغروب بشوية) · 🧭 باوع لـ<b>{compass(r.moon_az)}</b> '
                        f'واطي كلش قرب الأفق</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="spot" style="--c:#e5484d"><div class="n">🌑 مساء {d}: {HILAL_AR["D"]}</div></div>',
                        unsafe_allow_html=True)
with t3:
    st.markdown("لو المدينة تقلل الإضاءة اللي ما إلها داعي (ضوّ يطلع للسما بدل الشارع)، شگد تصير السما أحلى؟")
    cut = st.segmented_control("شكد نقلل؟", [0.2, 0.4, 0.6], default=0.4, format_func=lambda v: f"نقلل {v:.0%}") or 0.4
    em = skyq.emission_map(skyq.fetch_black_marble())
    after = skyq.site_sqm(skyq.sqm_map(skyq.intervention(em, lat, lon, 30, 0, cut, False)), lat, lon)
    n0, n1 = skyq.visible_stars(skyq.limiting_mag(sqm, 7)), skyq.visible_stars(skyq.limiting_mag(after, 7))
    m1, m2 = st.columns(2)
    m1.metric("⭐ نجوم تشوفها هسه", f"~{n0:,.0f}")
    m2.metric("⭐ نجوم تشوفها بعد التقليل", f"~{n1:,.0f}", f"+{n1 - n0:,.0f} نجمة")
    st.caption("شلون؟ غطاء للمصباح يوجّه الضوّ للأرض 🔦 · ضوّ أقل بالليل المتأخر ⏲️ · لون أصفر دافئ بدل الأبيض الأزرق 🟡. "
               "(تقدير تقريبي)")
with t4:
    st.markdown("شفت السما الليلة؟ گلنا شنو شفت، وهاي تساعد النموذج يصير أذكى لأهل الموصل 🙏")
    with st.form("contribute"):
        mw = st.radio("تگدر تشوف درب التبانة (الشريط الأبيض الضبابي)؟", ["إي 🌌", "لا"], horizontal=True)
        sqm_in = st.number_input("عندك جهاز قياس SQM؟ اكتب الرقم (مو ضروري)", 0.0, 22.5, 0.0, 0.1)
        if st.form_submit_button("📨 أرسل"):
            import skyclass
            skyclass.contribute(lat, lon, mw.startswith("إي"), sqm_in or None)
            st.success("وصلت! شكراً إلك 🙏")
with t5:
    st.markdown("""
- **الدرجة (من 100):** شكد الظروف زينة للرصد بمكانك ووقتك. فوق 50 = تستاهل تطلع.
- **التلوث الضوئي 💡:** ضوّ المدينة اللي يطلع للسما ويخلي النجوم ما تبين. نقيسه من **صور الأقمار الصناعية لناسا** بالليل.
- **درب التبانة 🌌:** مجرّتنا! تبين كشريط أبيض ضبابي بس بمكان مظلم كلش.
- **الغبار / العجاج 🌫️:** يخلي السما باهتة ويطمس النجوم الخافتة.
- **القمر 🌕:** لما يكون بدر يضوّي السما هواية، فأحسن رصد يكون بليالي المحاق (بلا قمر).
- **شلون نحسبها؟** نموذج ذكاء اصطناعي (XGBoost) تعلّم من **15 ألف قياس** حقيقي للسما، ويّا توقعات الجو والغبار والقمر.
""")

if a["notes"] or rec_notes:
    st.caption("ℹ️ بعض توقعات الجو ما متوفرة لهذا الوقت، فاستخدمنا المعدل الطبيعي لجو الموصل.")
st.caption("المصادر: ناسا (صور الأضواء الليلية VNP46A2) · Globe at Night · Open-Meteo (جو وغبار وارتفاع) · OpenStreetMap (شوارع)")
