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
.lpbar { height: 12px; border-radius: 6px; background: rgba(128,128,128,.25); margin: 8px 0; overflow: hidden; }
.lpbar div { height: 100%; border-radius: 6px; }
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


def factor_groups(a, lp_pct):
    """All factors that affect observing, grouped; live value + status where we measure it, tips where we can't.
    Status: 🟢 good · 🟠 so-so · 🔴 bad · ℹ️ advice (not measured)."""
    g = lambda v, ok, mid: "🟢" if v < ok else "🟠" if v < mid else "🔴"
    sun = a["sun_alt"]
    twilight = ("🟢", "ليل مظلم كامل") if sun < -18 else ("🟠", "شفق فلكي خفيف") if sun < -12 else \
        ("🟠", "شفق بحري — السما بعدها مضيئة") if sun < -6 else ("🔴", "شفق مدني — بعد الغروب مباشرة") if sun < 0 else ("🔴", "نهار")
    moon_s = a["components"]["moon"]
    moon_txt = f"مضاء {a['moon_illum']:.0%} · " + ("تحت الأفق 👍" if a["moon_alt"] <= 0 else f"ارتفاعه {a['moon_alt']:.0f}°")
    info = "ℹ️"
    return [
        ("🌦️ الجو والتلوث المحيط", [
            ("☁️ الغيوم والضباب", f"{a['cloud_cover']}%", g(a["cloud_cover"], 20, 50),
             "تحجب ضوء النجوم أو تضعفه، وحتى الغيوم الرقيقة تقلّل جودة الرصد."),
            ("🌫️ الغبار والعواصف الترابية والدخان", f"AOD {a['aod']} · الرؤية {a['visibility_km']} كم", g(a["aod"], 0.3, 0.5),
             "يشتّت ويمتص الضوء، فتقل شفافية الجو ويصير الهلال الخافت أصعب بالتمييز."),
            ("💡 التلوث الضوئي", f"{lp_pct:.0f}%", g(lp_pct, 40, 70),
             "إنارة الشوارع والمباني والإعلانات ترفع سطوع السما، فتختفي النجوم الأضعف بالخلفية."),
            ("💨 اضطراب الهواء (Seeing)", f"{a['seeing_arcsec']}″ · ريح فوق {a['wind250']} كم/س", g(a["seeing_arcsec"], 1.5, 2.5),
             "اختلاف حرارة وحركة طبقات الهواء يخلّي النجم يرتعش ويتوسّع ويفقد التفاصيل — يأثر أكثر على التلسكوب والتصوير."),
            ("💧 الرطوبة وبخار الماء", f"{a['humidity']}%", g(a["humidity"], 70, 85),
             "بخار الماء يمتص أطوال موجية معيّنة (خصوصاً تحت الحمراء)، والندى على العدسات يضعف الصورة."),
            ("🔥 حرارة الأسطح القريبة", "نصيحة", info,
             "الهواء الحار الصاعد من الإسفلت والأسطح يشوّه الصورة: ابتعد عنها، وخلي التلسكوب يبرد 30–60 دقيقة قبل الرصد."),
        ]),
        ("🌅 الإضاءة الطبيعية ومكان الجرم بالسما", [
            ("🌇 الشفق وضوء الشمس", twilight[1], twilight[0],
             "بعد الغروب تبقى السما مضيئة شوية. هذا أهم عامل لرصد الهلال الجديد لأن الهلال ضعيف وخلفيته مضيئة."),
            ("🌕 ضوء القمر", moon_txt, "🟢" if moon_s >= 80 else "🟠" if moon_s >= 50 else "🔴",
             "القمر يرفع سطوع السما ويصعّب رصد النجوم والأجرام الخافتة. أحسن رصد بليالي المحاق."),
            ("📐 قرب الجرم من الأفق", "باوع فوق 30° إذا تگدر", info,
             "ضوء الجرم القريب من الأفق يمر بغلاف جوي أطول فيضعف أكثر. شوف ارتفاع كل نجمة بتبويب «شنو أشوف؟»."),
            ("🏢 عوائق الأفق", "اختار مكان مفتوح", info,
             "المباني والجبال والأشجار تحجب الأجرام الواطية حتى لو الجو صافي، خصوصاً الهلال لأنه قريب من الأفق."),
        ]),
        ("📷 تشويش الكاميرا والأجهزة (النويز)", [
            ("🎲 ضوضاء الفوتونات (Shot noise)", "طبيعي", info,
             "عدد فوتونات الضوء الخافت يتذبذب، فيصعب تمييزه. الحل: تعريض أطول أو جمع صور كثيرة (Stacking)."),
            ("🌡️ التيار المظلم (Dark current)", "يزيد بالحرارة", info,
             "الحساس يولّد إشارة حتى بدون ضوء. برّد الكاميرا وصوّر صور Dark frames وانطرحها من الصور."),
            ("📶 ضوضاء القراءة (Read noise)", "بكل صورة", info,
             "إلكترونيات الحساس تضيف تشويش لكل صورة. صور أقل بتعريض أطول تقلّل أثرها."),
            ("🔴 البكسلات الساخنة واختلاف حساسية البكسلات", "ممكن تنفهم نجوم", info,
             "نقاط أو أنماط زائفة. تتصلّح بصور المعايرة Dark و Flat."),
            ("☄️ الأشعة الكونية", "نقاط وخطوط زائفة", info,
             "تضرب الحساس وتترك نقاط أو خطوط مضيئة. جمع صور متعددة يشيلها."),
            ("🛰️ مرور الأقمار الصناعية", "خطوط بالصور", info,
             "تترك خطوط مضيئة خصوصاً وقت الشفق. صوّر بعد الظلام الكامل واستخدم Stacking."),
            ("🔧 أخطاء التركيب", "افحص قبل الرصد", info,
             "اهتزاز الحامل، ضعف التتبّع، سوء التركيز (Focus)، واتّساخ العدسات كلها تخرّب الصورة."),
        ]),
    ]


def noise_sources(a, lp_pct, moon_tip):
    """Everything that degrades the sky here and now as a 0-100 'noise' level (higher = worse), plus how to reduce it."""
    c = a["components"]
    return [
        ("💡", "التلوث الضوئي", f"{lp_pct:.0f}%", lp_pct,
         "روح لمكان أظلم (شوف الأماكن جوّه)، باوع لجهة بعيدة عن المدينة، وبالتصوير استخدم فلتر تلوث ضوئي."),
        ("🌕", "ضوء القمر", f"مضاء {a['moon_illum']:.0%}" + (" · غايب" if a["moon_alt"] <= 0 else f" · ارتفاعه {a['moon_alt']:.0f}°"),
         100 - c["moon"], moon_tip),
        ("🌫️", "الغبار والعجاج", f"AOD {a['aod']}", float(np.clip((a["aod"] - 0.1) / 0.6, 0, 1) * 100),
         "انتظر يوم أو يومين بعد العجاج، وروح لمكان أعلى — الغبار يتجمّع قرب الأرض."),
        ("☁️", "الغيوم", f"{a['cloud_cover']}%", float(a["cloud_cover"]),
         "اختار ساعة خضرة من شريط الساعات فوق، أو جرّب ليلة ثانية."),
        ("💨", "اضطراب الهواء", f"{a['seeing_arcsec']}″", float(np.clip((a["seeing_arcsec"] - 1) / 2, 0, 1) * 100),
         "رصد النجوم العالية (فوق 45°)، ابتعد عن الإسفلت والأسطح الحارة، وخلي التلسكوب يبرد ساعة قبل الرصد."),
        ("💧", "الرطوبة والندى", f"{a['humidity']}%", float(np.clip((a["humidity"] - 60) / 35, 0, 1) * 100),
         "استخدم غطاء ندى أو سخانة للعدسة، وغطّي الأجهزة لما ما تستعملها."),
    ]


def noise_word(v):
    return ("قليل", "#30a46c") if v < 30 else ("متوسط", "#f5a524") if v < 60 else ("عالي", "#e5484d")


@st.cache_data(ttl=3600, show_spinner=False)
def moon_track(lat, lon, night):
    times = engine.local_night(night)
    return pd.concat([pd.DataFrame(dict(time=times)), engine.sky_geometry(lat, lon, times).reset_index(drop=True)], axis=1)


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
    return pd.DataFrame(rows), wx[1]  # notes: forecast fallback used?


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
rec = rec[rec.score > a["score"]].reset_index(drop=True) if len(rec) else rec  # only spots better than here
k = a["cls"]
color, verdict, _ = LEVEL[k]
why = "".join(f"<li>{r}</li>" for r in reasons(a))
_w = engine.REGIONS["mosul"]["weights"][obs]
_c = a["components"]
_raw = [_c["light"] * _w[0], _c["weather"] * _w[1], _c["moon"] * _w[2], _c["elevation"] * _w[3]]
_parts = [int(v) for v in _raw]
if a["sun_alt"] <= -12 and a["cloud_cover"] < 80:  # no cap: round so the parts add up exactly to the score
    for i in sorted(range(4), key=lambda i: _raw[i] - _parts[i], reverse=True)[:max(0, a["score"] - sum(_parts))]:
        _parts[i] += 1
score_parts = (f"💡 ظلام السما {_parts[0]} + ☁️ الجو {_parts[1]} + 🌕 القمر {_parts[2]} + ⛰️ الارتفاع {_parts[3]}")
if a["sun_alt"] > -12:
    score_parts += " ← وتنقص لأن الشمس بعدها ما غابت"
elif a["cloud_cover"] >= 80:
    score_parts += " ← وتنقص لأن الغيوم فوق 80%"
score_parts += " · التفاصيل بـ «🧮 منين اجت الدرجة» تحت"
st.markdown(f"""<div class="hero" style="--c:{color}; --s:{a['score']}">
  <div class="ring"><div class="in"><div><b>{a['score']}</b><small>من 100</small></div></div></div>
  <div><div class="verdict">{verdict}</div>
  <div>{TYPE_UI[obs][0]} · {place} · {DAYS[night.weekday()]} {HOUR_LABEL[hour]}</div>
  <ul class="why">{why}</ul>
  <div class="hint">🧮 <b>{a['score']}</b> = {score_parts}</div></div></div>""", unsafe_allow_html=True)

# where the score comes from (live breakdown)
with st.expander(f"🧮 منين اجت الدرجة {a['score']}؟ شلون انحسبت؟"):
    c_ = a["components"]
    w_ = engine.REGIONS["mosul"]["weights"][obs]
    cell_ = engine.nearest(engine.cells(), lat, lon)
    rows_ = [
        ("💡 ظلام السما (التلوث الضوئي)", c_["light"], w_[0],
         f"معدل رأيين: **الذكاء الاصطناعي** ({cell_.ai_score:.0f}) و**النموذج الفيزيائي** ({cell_.physics_score:.0f}، "
         f"سطوع السما {cell_.sqm:.1f}) من صور ناسا الليلية الحديثة"),
        ("☁️ الجو", c_["weather"], w_[1],
         f"غيوم {a['cloud_cover']}% · رطوبة {a['humidity']}% · غبار AOD {a['aod']} · رؤية {a['visibility_km']} كم "
         "(توقعات Open-Meteo / Copernicus لهاي الساعة)"),
        ("🌕 القمر", c_["moon"], w_[2],
         f"مضاء {a['moon_illum']:.0%} · " + ("غايب تحت الأفق" if a["moon_alt"] <= 0 else f"ارتفاعه {a['moon_alt']:.0f}°") +
         " (محسوب بمكتبة Astropy)"),
        ("⛰️ الارتفاع", c_["elevation"], w_[3], f"{a['elevation_m']} م فوق البحر (كل ما أعلى، غبار وضباب أقل فوقك)"),
    ]
    st.markdown("**الدرجة = مجموع (درجة كل عامل × وزنه)**. الأوزان تتغيّر حسب نوع الرصد اللي اخترته:")
    st.markdown("\n".join(f"- **{n}: {s}/100 × {wt:.0%} = {s * wt:.1f}** ← {why_}" for n, s, wt, why_ in rows_))
    raw_ = sum(s * wt for _, s, wt, _ in rows_)
    note_ = ""
    if a["sun_alt"] > -12:
        note_ = " · بس لأن **الشمس بعدها ما غابت كفاية**، الدرجة تنقص تلقائياً لـ 10 كحد أقصى"
    elif a["cloud_cover"] >= 80:
        note_ = " · بس لأن **الغيوم فوق 80%**، الدرجة تنقص تلقائياً لـ 20 كحد أقصى"
    if note_:
        st.markdown(f"**المجموع ≈ {raw_:.0f}**{note_} ← **الدرجة النهائية {a['score']}/100**")
    else:
        st.markdown(f"**المجموع = {a['score']}/100** (الأرقام فوق مقرّبة، فممكن الجمع اليدوي يفرق درجة وحدة)")
    st.caption("التصنيف: 0–25 ما تصلح · 26–50 متوسطة · 51–75 زينة · 76–100 ممتازة. "
               "الأوزان حالياً **مختارة من الفريق** (مو متعلّمة بعد)، وراح نتعلّمها من قياسات حقيقية بالمرحلة الجاية.")

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

# light-pollution percentage at this place (sky brightness from recent NASA lights; 22 = pristine, 16 = brightest city)
site_sqm = a["sqm"] if a.get("sqm") is not None else skyq.site_sqm(sqm_grid(), lat, lon)
lp_pct = float(np.clip((22.0 - site_sqm) / 6.0, 0, 1) * 100)
stars_here = float(skyq.visible_stars(skyq.limiting_mag(site_sqm, 7)))
stars_dark = float(skyq.visible_stars(skyq.limiting_mag(22.0, 7)))
hidden_pct = max(0.0, 100 * (1 - stars_here / stars_dark))
brighter = 10 ** (0.4 * (22.0 - site_sqm))
lp_level, lp_col = next((w, cl) for lim, w, cl in [(20, "خفيف جداً", "#30a46c"), (40, "خفيف", "#30a46c"),
                                                    (60, "متوسط", "#f5a524"), (80, "عالي", "#e5484d"),
                                                    (101, "عالي جداً", "#e5484d")] if lp_pct < lim)
st.markdown(f"""<div class="spot" style="--c:{lp_col}">
  <div class="n">💡 نسبة التلوث الضوئي بمكانك: <span style="color:{lp_col}">{lp_pct:.0f}%</span> — {lp_level}</div>
  <div class="lpbar"><div style="width:{lp_pct:.0f}%; background:{lp_col}"></div></div>
  ⭐ يخفي تقريباً <b>{hidden_pct:.0f}%</b> من النجوم: تشوف <b>~{stars_here:,.0f}</b> نجمة بدل <b>~{stars_dark:,.0f}</b> بسما مظلمة ·
  🌃 السما هنا أضوى من الطبيعي بـ <b>{brighter:,.0f} مرة</b></div>""", unsafe_allow_html=True)
st.caption("0% = سما البر الصافية بلا أي ضوء · 100% = أضوى مركز مدينة. محسوبة من صور ناسا الليلية الحديثة لمكانك. "
           "🔬 كل العوامل اللي تأثر على رصدك (اضطراب الهواء، الرطوبة، الشفق، نويز الكاميرا…) بتبويب «كل العوامل» جوّه.")

# ------------------------------------------------------------------ noise here, always visible
mt = moon_track(lat, lon, night)
if a["moon_alt"] <= 0:
    moon_tip = "القمر غايب بهذا الوقت 👍"
else:
    after = mt[(mt.time > when) & (mt.moon_alt <= 0)]
    nxt_nm = skyq.conjunctions(datetime.combine(night, datetime.min.time()) - timedelta(days=2), 32)[0]
    moon_tip = (f"انتظر لحد ما يغيب القمر الساعة {after.time.iloc[0]:%H:%M}، أو باوع بعكس اتجاهه." if len(after) else
                f"القمر ما يغيب هاي الليلة. أحسن ليالي للرصد حول المحاق ({nxt_nm + timedelta(hours=3):%d/%m})، "
                "وبالتصوير باوع بعكس اتجاه القمر.")
src = sorted(noise_sources(a, lp_pct, moon_tip), key=lambda x: -x[3])
total_noise = 100 - a["score"]
st.markdown(f"### 🔊 النويز الموجود بمكانك هسه — {total_noise}%")
st.caption("النويز = كل شي يخرّب السما (ضوّ، قمر، غبار، غيوم، اضطراب هوا، رطوبة). كل ما يقل أحسن. "
           "تحت كل مصدر عالي: شلون تقلله.")
rows_html = ""
for e, name, val, lvl, tip in src:
    word, col = noise_word(lvl)
    tip_html = f'<div class="hint">🛠️ {tip}</div>' if lvl >= 30 else ""
    rows_html += (f'<div class="spot" style="--c:{col}; padding:10px 14px"><b>{e} {name}</b> — {val} · '
                  f'<span style="color:{col}; font-weight:800">نويز {word} ({lvl:.0f}%)</span>'
                  f'<div class="lpbar"><div style="width:{max(lvl, 2):.0f}%; background:{col}"></div></div>{tip_html}</div>')
st.markdown(rows_html, unsafe_allow_html=True)
with st.expander("📷 شلون تقلل نويز الكاميرا والتلسكوب؟"):
    st.markdown("""
- **🎲 ضوضاء الفوتونات:** صوّر **صور كثيرة واجمعها (Stacking)** بدل صورة وحدة، أو زيد وقت التعريض.
- **🌡️ التيار المظلم:** **برّد الكاميرا**، وصوّر **Dark frames** (نفس الإعدادات بس الغطاء مسدود) وانطرحها.
- **📶 ضوضاء القراءة:** صور **أقل بتعريض أطول** أحسن من صور كثيرة قصيرة كلش.
- **🔴 البكسلات الساخنة:** صور المعايرة **Dark و Flat** تصلحها، و**Dithering** (تحريك بسيط بين الصور) يساعد.
- **☄️ الأشعة الكونية و🛰️ خطوط الأقمار الصناعية:** **Stacking بطريقة Sigma-clipping** يشيلها، وصوّر بعد الظلام الكامل.
- **🔧 الأجهزة:** ثبّت الحامل، ضبّط **التتبّع والتركيز (Focus)**، ونظّف العدسات قبل الرصد.
""")

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
st.markdown(f"### 🚗 وين أروح حتى يقل النويز؟ أحسن أماكن ضمن {radius} كم")
st.caption("اخترناها لأن النويز هناك أقل: سماها أظلم، والجو زين بالوقت اللي اخترته، وقريبة من شارع رئيسي تگدر توصله بالسيارة. "
           "الأسهم تقارن النويز هناك ويّا مكانك: 🟢 أقل · 🔴 أكثر.")


def cmp(e, here, there, fmt, tol):
    col = "#30a46c" if there < here - tol else "#e5484d" if there > here + tol else "#9aa0a6"
    return f'{e} {fmt(here)} ← <b style="color:{col}">{fmt(there)}</b>'

if len(rec):
    for i, r in enumerate(rec.itertuples(), 1):
        col, _, short = LEVEL[int(r.cls)]
        medal = ["🥇", "🥈", "🥉"][i - 1] if i <= 3 else f"{i}."
        st.markdown(f"""<div class="spot" style="--c:{col}"><div class="n">{medal} {place_name(r.lat, r.lon)} —
          {int(r.score)}/100 ({short})</div>
          🚗 {r.km:.0f} كم عنك · 🛣️ {r.road_km:.1f} كم من الشارع · ⛰️ ارتفاع {r.elevation_m:.0f} م<br>
          🔉 النويز هناك: <b>{100 - int(r.score)}%</b> بدل <b>{total_noise}%</b> ·
          {cmp("💡", lp_pct, float(np.clip((22 - r.sqm) / 6, 0, 1) * 100), lambda v: f"{v:.0f}%", 3)} ·
          {cmp("🌫️", a["aod"], r.aod, lambda v: f"{v:.2f}", 0.03)} ·
          {cmp("☁️", a["cloud_cover"], r.cloud_cover, lambda v: f"{v:.0f}%", 5)} ·
          {cmp("💨", a["seeing_arcsec"], float(skyq.seeing(r.wind10, r.wind250)), lambda v: f"{v:.1f}″", 0.1)}<br>
          <a class="go" target="_blank" href="https://www.google.com/maps/dir/?api=1&origin={lat},{lon}&destination={r.lat},{r.lon}">
          🧭 خذني هناك</a></div>""", unsafe_allow_html=True)
else:
    st.info("ماكو مكان أحسن من مكانك ضمن هاي المسافة. جرّب «🚙 طلعة (100 كم)» أو ليلة ثانية.")

# ------------------------------------------------------------------ more (tabs)
st.markdown("### 🎒 أشياء زيادة")
t0, t1, t2, t3, t4, t5 = st.tabs(["🔬 كل العوامل", "✨ شنو أشوف؟", "🌒 الهلال", "💡 خلّي سما مدينتك أحلى", "🙋 شاركنا",
                                  "📖 شنو يعني؟"])
with t0:
    st.markdown("كل عامل يأثر على الرصد بوحدة من ثلاث طرق: **يقلّل الضوء الواصل**، أو **يزيد سطوع خلفية السما**، "
                "أو **يشوّش الصورة والقياس**. القيم تحت **لمكانك ووقتك**: 🟢 زين · 🟠 متوسط · 🔴 سيء · ℹ️ نصيحة.")
    for title, rows in factor_groups(a, lp_pct):
        st.markdown(f"#### {title}")
        for name, val, stt, why_txt in rows:
            col = {"🟢": "#30a46c", "🟠": "#f5a524", "🔴": "#e5484d"}.get(stt, "#5b8cff")
            st.markdown(f'<div class="spot" style="--c:{col}"><div class="n">{stt} {name} — {val}</div>{why_txt}</div>',
                        unsafe_allow_html=True)
    st.markdown("#### 🌒 النجوم مو مثل الهلال")
    st.info("للنجوم الخافتة أهم شي **ظلام السما**. للهلال أهم شي **تباينه ويّا الشفق** و**ارتفاعه فوق الأفق**. "
            "لهذا نحسبهم بطريقتين مختلفتين (عوامل الهلال بتبويب «الهلال»).")
    st.markdown("#### 🛰️ لو التلسكوب بالفضاء؟")
    st.markdown("تختفي **الغيوم والغبار واضطراب الهواء**، بس تبقى **ضوضاء الحساس والأشعة الكونية والضوء الشارد** "
                "والخلفية من الغبار بين الكواكب. يعني حتى الرصد الفضائي مو خالي من التشويش.")
    st.caption("المصادر: المرصد الأوروبي الجنوبي (ESO) · توثيق حساسات هابل (STScI) · المرصد البحري الأمريكي (USNO) · ناسا. "
               "اضطراب الهواء (Seeing) تقدير من سرعة الريح العالية، مو قياس مباشر.")
SQM = sqm_grid()
sqm = skyq.site_sqm(SQM, lat, lon)
with t1:
    st.caption("النجوم والكواكب اللي تبين من مكانك بهاي الليلة، ووين تباوع.")
    tg, tg_notes = targets(lat, lon, night, sqm)
    if tg_notes:
        st.warning("⚠️ توقع الجو ما اشتغل لهاي القائمة، فهي تفترض جو طبيعي (غالباً صافي). اعتمد على الجواب الرئيسي فوق للغيوم.")
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
        facts = ""
        if pd.notna(getattr(r, "arcl", None)):
            facts = (f'<br>📐 الاستطالة (بعده عن الشمس): <b>{r.arcl:.1f}°</b> · ⬆️ ارتفاعه: <b>{r.moon_alt:.1f}°</b> · '
                     f'⏳ يبقى بعد الغروب: <b>{r.lag_min:.0f} دقيقة</b> · 🌙 سماكة الهلال: <b>{r.width_arcmin:.2f}′</b>')
        if r.final_zone in ("A", "B", "C"):
            st.markdown(f'<div class="spot" style="--c:{SAFE}"><div class="n">🌙 مساء {d}: {HILAL_AR[r.final_zone]}</div>'
                        f'⏰ الساعة {r.best_time:%H:%M} (بعد الغروب بشوية) · 🧭 باوع لـ<b>{compass(r.moon_az)}</b> '
                        f'واطي كلش قرب الأفق{facts}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="spot" style="--c:#e5484d"><div class="n">🌑 مساء {d}: {HILAL_AR["D"]}</div>{facts}</div>',
                        unsafe_allow_html=True)
    st.caption("شنو يقرر رؤية الهلال؟ **الاستطالة**، **ارتفاعه فوق الأفق**، **مدة بقائه بعد الغروب**، **سماكته وسطوعه مقابل الشفق**، "
               "و**صفاء الأفق وخبرة الراصد ونوع الجهاز**. **عمر الهلال وحده ما يكفي**، وحساب موقعه بدقة ما يعني ضمان رؤيته.")
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
- **نسبة التلوث الضوئي (%):** 0% يعني سما البر الصافية، و100% يعني أضوى مركز مدينة. كل ما تقل النسبة تشوف نجوم أكثر.
- **درب التبانة 🌌:** مجرّتنا! تبين كشريط أبيض ضبابي بس بمكان مظلم كلش.
- **الغبار / العجاج 🌫️:** يخلي السما باهتة ويطمس النجوم الخافتة.
- **القمر 🌕:** لما يكون بدر يضوّي السما هواية، فأحسن رصد يكون بليالي المحاق (بلا قمر).
- **شلون نحسبها؟** نموذج ذكاء اصطناعي (XGBoost) تعلّم من **15 ألف قياس** حقيقي للسما، ويّا توقعات الجو والغبار والقمر.
""")

if a["notes"] or rec_notes:
    st.caption("ℹ️ بعض توقعات الجو ما متوفرة لهذا الوقت، فاستخدمنا المعدل الطبيعي لجو الموصل.")
st.caption("المصادر: ناسا (صور الأضواء الليلية VNP46A2) · Globe at Night · Open-Meteo (جو وغبار وارتفاع) · OpenStreetMap (شوارع)")
