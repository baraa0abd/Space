"""Public page (Iraqi Arabic, beginner-friendly) — Iraq Sky AI, Mosul.
Three questions (where / when / what) -> one big answer (go or not, why, when, where instead).
UI follows the "سما الموصل" design: ask screen (question bar + map) -> result screen (5 sections)."""
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

import engine
import skyq

# ------------------------------------------------------------------ design system
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&family=Lalezar&display=swap');
:root { --bg:#0a0e1c; --panel:#121a33; --panel2:#0e1530; --line:rgba(255,255,255,.07); --text:#eef0f6; --muted:#98a1b7;
  --gold:#f3b941; --green:#5ad1ad; --red:#ec6a55; --blue:#6b8cff; --cream:#f6f1e7; --ink:#15130f; }
.stApp { background-color: var(--bg);
  background-image: radial-gradient(1px 1px at 40px 60px, rgba(255,255,255,.45), transparent 60%),
    radial-gradient(1px 1px at 160px 220px, rgba(255,255,255,.35), transparent 60%),
    radial-gradient(1.5px 1.5px at 300px 120px, rgba(255,255,255,.4), transparent 60%),
    radial-gradient(1px 1px at 380px 330px, rgba(255,255,255,.3), transparent 60%);
  background-size: 420px 420px; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] { max-width: 1180px; padding-top: 1.2rem; direction: rtl; text-align: right;
  font-family: 'IBM Plex Sans Arabic', sans-serif; color: var(--text); }
[data-testid="stMainBlockContainer"] * { font-family: 'IBM Plex Sans Arabic', sans-serif; }
[data-testid="stIconMaterial"], .material-symbols-rounded { font-family: 'Material Symbols Rounded' !important; }
[data-testid="stSlider"] { direction: ltr; }
.disp, .disp * { font-family: 'Lalezar', 'IBM Plex Sans Arabic', sans-serif !important; font-weight: 400; letter-spacing: 0; }
.muted { color: var(--muted); }
.ltr { direction: ltr; unicode-bidi: isolate; display: inline-block; }
/* brand */
.brand { display:flex; align-items:center; gap:12px; justify-content:flex-start; }
.brand .logo { width:46px; height:46px; border-radius:12px; background:var(--gold); display:grid; place-items:center; font-size:1.35rem; }
.brand .name { font-family:'Lalezar',sans-serif !important; font-size:1.9rem; color:var(--text); }
.summary { color:var(--muted); font-size:1rem; padding-top:12px; text-align:left; }
.rule { height:1px; background:var(--line); margin:10px -2rem 14px; }
/* ask screen */
.hero-title { text-align:center; font-size:clamp(2.6rem, 6vw, 4.6rem); line-height:1.15; margin:26px 0 6px; color:#f3eee4; }
.hero-sub { text-align:center; color:var(--muted); font-size:1.15rem; margin-bottom:26px; }
.q-label { color:#6f6a60; font-size:.9rem; font-weight:600; margin-bottom:-6px; }
.q-hint { text-align:center; color:var(--muted); font-size:.95rem; margin:10px 0 18px; }
.st-key-qbar { background:var(--cream); border-radius:28px; padding:14px 18px 12px; box-shadow:0 18px 50px rgba(0,0,0,.35); }
.st-key-qbar [data-testid="stColumn"]:not(:last-child) { border-inline-end: 1px solid #e3dccd; padding-inline-end: 10px; }
.st-key-qbar [data-baseweb="select"] > div { background:transparent !important; border:none !important; box-shadow:none !important; }
.st-key-qbar [data-baseweb="select"] div, .st-key-qbar [data-baseweb="select"] span { color:var(--ink) !important; font-weight:700;
  font-size:1.12rem; }
.st-key-qbar [data-baseweb="select"] svg { fill:#6f6a60; }
.st-key-qbar [data-testid="stPopover"] button { background:transparent; border:none; color:var(--ink); font-weight:700;
  font-size:1.08rem; box-shadow:none; justify-content:flex-start; padding-inline:6px; }
.st-key-qbar [data-testid="stPopover"] button p { color:var(--ink); }
.st-key-go button { background:var(--ink) !important; color:var(--gold) !important; border:none !important; border-radius:20px !important;
  min-height:74px; width:100%; font-weight:800; font-size:1.15rem; }
.st-key-go button p { color:var(--gold); font-size:1.15rem; font-weight:800; }
.st-key-mapwrap { background:var(--panel2); border:1px solid var(--line); border-radius:24px; padding:12px; margin-top:6px; }
.st-key-mapwrap [data-testid="stDeckGlJsonChart"] { border-radius:18px; overflow:hidden; }
.legend { display:flex; gap:18px; justify-content:flex-start; color:var(--muted); font-size:.9rem; padding:8px 6px 2px; }
.legend i { display:inline-block; width:10px; height:10px; border-radius:50%; margin-inline-end:6px; vertical-align:middle; }
/* result nav pills */
.st-key-sec [data-testid="stButtonGroup"] { gap:6px; flex-wrap:wrap; }
.st-key-sec button { background:transparent !important; border:none !important; color:var(--muted) !important; border-radius:999px !important;
  padding:8px 20px !important; font-weight:600; }
.st-key-sec button p { font-size:1.02rem; }
.st-key-sec button[kind="segmented_controlActive"], .st-key-sec button[data-testid="stBaseButton-segmented_controlActive"] {
  background:var(--gold) !important; color:var(--ink) !important; }
.st-key-sec button[kind="segmented_controlActive"] p, .st-key-sec button[data-testid="stBaseButton-segmented_controlActive"] p { color:var(--ink); font-weight:800; }
.st-key-edit button { border-radius:999px !important; border:1px solid var(--line) !important; background:transparent !important;
  color:var(--gold) !important; padding:6px 18px !important; }
.st-key-edit button p { color:var(--gold); font-weight:700; }
/* cards */
.card { background:var(--panel); border:1px solid var(--line); border-radius:22px; padding:22px 24px; margin-bottom:14px; }
.card.soft { background:#16223f; }
.card h4 { margin:0 0 4px; font-size:1.05rem; color:var(--text); font-weight:700; }
.sec-title { font-size:clamp(2rem, 4vw, 3rem); margin:18px 0 4px; color:#f3eee4; }
.sec-sub { color:var(--muted); font-size:1.05rem; margin-bottom:22px; }
.linkbtn button { width:100%; border-radius:999px !important; min-height:56px; font-weight:800 !important; border:none !important; }
.st-key-to_spots button, .st-key-to_spots2 button { background:var(--green) !important; color:var(--ink) !important; }
.st-key-to_spots button p, .st-key-to_spots2 button p { color:var(--ink); font-weight:800; }
.st-key-to_where button { background:var(--gold) !important; color:var(--ink) !important; }
.st-key-to_where button p { color:var(--ink); font-weight:800; }
.st-key-to_noise button, .st-key-to_calc button { background:transparent !important; border:none !important; color:var(--gold) !important;
  padding:0 !important; min-height:0 !important; }
.st-key-to_noise button p, .st-key-to_calc button p { color:var(--gold); font-weight:700; }
/* result hero */
.hero { display:flex; gap:34px; align-items:center; justify-content:flex-start; margin:30px 0 26px; }
.ring { flex:0 0 250px; height:250px; border-radius:50%; display:grid; place-items:center;
  background: conic-gradient(from 0deg, var(--c) calc(var(--s) * 1%), #1f2748 0); }
.ring .in { width:212px; height:212px; border-radius:50%; background:var(--bg); display:grid; place-items:center; text-align:center; }
.ring b { font-size:5.2rem; line-height:1; color:#f3eee4; font-weight:700; } .ring small { display:block; color:var(--muted); font-size:1.05rem; }
.badge { display:inline-block; padding:6px 18px; border-radius:999px; font-weight:700; font-size:1rem;
  background:color-mix(in srgb, var(--c) 22%, transparent); color:var(--c); margin-bottom:10px; }
.headline { font-size:clamp(2.2rem, 4.4vw, 3.6rem); line-height:1.25; color:#f3eee4; }
.reasons { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
.rc { background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:18px 20px; font-size:1.05rem; color:var(--text); }
.dot { display:inline-block; width:12px; height:12px; border-radius:50%; margin-inline-end:10px; vertical-align:middle; }
.tip .k { color:#9fc3e6; font-size:.95rem; } .tip .place { font-size:1.35rem; font-weight:700; margin:6px 0; color:var(--text); }
.big { font-size:3.2rem; font-weight:800; line-height:1.05; } .unit { color:var(--muted); font-size:1rem; }
.row { display:flex; justify-content:space-between; align-items:center; gap:10px; }
/* hours strip */
.strip { display:grid; grid-template-columns:repeat(13, 1fr); gap:6px; margin:6px 0 18px; }
.hr { border-radius:10px; padding:8px 0; text-align:center; font-size:.8rem; background:color-mix(in srgb, var(--c) 85%, #000); color:#111; font-weight:700; }
.hr.sel { outline:3px solid #fff; } .hr b { display:block; font-size:1rem; }
/* factor cards (how calculated) */
.fgrid { display:grid; grid-template-columns:repeat(4, 1fr); gap:16px; }
.fcard { background:var(--panel); border:1px solid var(--line); border-radius:20px; padding:22px; }
.fcard .t { font-weight:700; font-size:1.1rem; color:var(--text); }
.fcard .v { font-size:3rem; font-weight:800; color:#f3eee4; margin:6px 0 2px; } .fcard .v small { font-size:1rem; color:var(--muted); font-weight:500; }
.bar { height:8px; border-radius:6px; background:#232b4d; overflow:hidden; margin:10px 0 12px; }
.bar div { height:100%; border-radius:6px; }
.fcard .d { color:var(--muted); font-size:.93rem; min-height:44px; border-bottom:1px solid var(--line); padding-bottom:12px; }
.fcard .f { display:flex; justify-content:space-between; padding-top:12px; color:var(--muted); } .fcard .f b { color:var(--gold); }
.total { background:#17203d; border-radius:22px; padding:22px 28px; display:flex; justify-content:space-between; align-items:center; gap:20px; margin-top:18px; }
.total .n { font-size:3rem; font-weight:800; color:var(--gold); }
.scale { display:flex; gap:6px; flex:1; max-width:560px; direction:rtl; }
.scale div { flex:1; text-align:center; font-size:.85rem; color:var(--muted); }
.scale div i { display:block; height:10px; border-radius:6px; margin-bottom:6px; }
.scale div.on i { outline:2px solid #fff; outline-offset:2px; } .scale div.on { color:var(--text); font-weight:700; }
/* noise */
.nzbig { border-radius:22px; padding:26px 28px; background:color-mix(in srgb, var(--c) 22%, #121a33); text-align:left; }
.nzbig .n { font-size:4.4rem; font-weight:800; color:var(--c); line-height:1; } .nzbig .w { color:color-mix(in srgb, var(--c) 70%, #fff); }
details.nz { background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:16px 20px; margin-bottom:10px; }
details.nz[open] { border-color:rgba(107,140,255,.35); }
details.nz summary { list-style:none; cursor:pointer; display:grid; grid-template-columns:190px 1fr 64px; align-items:center; gap:18px; }
details.nz summary::-webkit-details-marker { display:none; }
details.nz summary b { color:var(--text); font-size:1.08rem; } details.nz summary span { font-weight:800; text-align:left; }
details.nz .tipx { color:var(--text); margin-top:12px; font-size:1rem; }
/* where to go */
.pcard { background:var(--panel); border:1px solid var(--line); border-radius:22px; padding:22px; min-height:440px; }
.pcard.first { border:2px solid var(--gold); }
.pbadge { display:inline-block; padding:4px 14px; border-radius:10px; background:#232b52; color:var(--text); font-size:.9rem; font-weight:700; }
.pcard.first .pbadge { background:var(--gold); color:var(--ink); }
.pname { font-size:1.4rem; font-weight:700; color:var(--text); margin:10px 0; }
.pscore { font-size:3rem; font-weight:800; color:var(--green); line-height:1; }
.cmpbox { background:#0c1227; border-radius:16px; padding:14px 18px; margin:16px 0; }
.cmpbox .row { padding:5px 0; color:var(--muted); } .cmpbox .row b { font-weight:800; }
.meta { color:var(--muted); font-size:.95rem; display:flex; gap:16px; flex-wrap:wrap; margin-bottom:18px; }
.gobtn { display:block; text-align:center; background:var(--gold); color:var(--ink) !important; text-decoration:none !important;
  border-radius:999px; padding:14px; font-weight:800; font-size:1.08rem; }
/* all factors */
.legend2 { display:flex; gap:18px; justify-content:flex-end; color:var(--muted); }
.gtitle { color:var(--gold); font-weight:700; font-size:1.15rem; margin:4px 0 12px; }
details.fx { background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:14px 18px; margin-bottom:10px; }
details.fx summary { list-style:none; cursor:pointer; display:flex; justify-content:space-between; align-items:center; gap:12px; }
details.fx summary::-webkit-details-marker { display:none; }
details.fx summary b { color:var(--text); } details.fx summary span { color:var(--muted); font-size:.95rem; }
details.fx p { color:var(--muted); margin:10px 0 0; font-size:.95rem; }
.spot { background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:14px 18px; margin:8px 0;
  border-inline-start:5px solid var(--c); }
.spot .n { font-size:1.12rem; font-weight:700; color:var(--text); }
.foot { color:var(--muted); font-size:.85rem; margin-top:26px; }
@media (max-width: 900px) {
  .hero { flex-direction:column; align-items:center; text-align:center; } .ring { flex-basis:200px; height:200px; }
  .ring .in { width:168px; height:168px; } .ring b { font-size:4rem; }
  .reasons, .fgrid { grid-template-columns:1fr 1fr; } .strip { grid-template-columns:repeat(7, 1fr); }
  details.nz summary { grid-template-columns:1fr 60px; } details.nz summary .bar { grid-column:1 / -1; }
  .total { flex-direction:column; } }
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
RED, GOLD, GREEN, BLUE = "#ec6a55", "#f3b941", "#5ad1ad", "#6b8cff"
LEVEL = [  # score class -> (color, headline, short)
    (RED, "لا، مو هسه —<br>السما ما تساعد", "ما تصلح"),
    (GOLD, "تگدر ترصد،<br>بس مو أحسن وقت", "متوسطة"),
    (GREEN, "زينة،<br>تستاهل تطلع!", "زينة"),
    (BLUE, "ممتازة!<br>روح ارصد", "ممتازة"),
]
TYPE_UI = {  # engine key -> (label, one-line explanation)
    "stargazing": ("النجوم", "النجوم: تشوف النجوم والكواكب بعينك — أسهل شي وتصلح للبداية."),
    "milky_way": ("درب التبانة", "درب التبانة: الشريط الأبيض الضبابي بالسما — يحتاج مكان مظلم كلش وبلا قمر."),
    "astrophotography": ("التصوير الفلكي", "التصوير الفلكي: تصوّر السما بالموبايل أو الكاميرا — يحتاج سما صافية ومظلمة."),
    "meteors": ("الشهب", "الشهب: نجوم تركض بسرعة بالسما — يحتاج صبر وسما مظلمة شوية."),
}
HOUR_LABEL = {19: "7 المسا", 20: "8 بالليل", 21: "9 بالليل", 22: "10 بالليل", 23: "11 بالليل", 0: "12 نص الليل",
              1: "1 بعد نص الليل", 2: "2 بعد نص الليل", 3: "3 الفجر", 4: "4 الفجر"}
SECTIONS = ["النتيجة", "شلون انحسبت؟", "النويز", "وين أروح؟", "كل العوامل", "المزيد"]
SAFE = GREEN


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


def reason_items(a):
    """Plain-language 'why' items from the engine result, each with a status colour."""
    c, out = a["components"], []
    if a["hourly"] is not None and a["score"] <= 10 and c["weather"] > 10:
        out.append(("الشمس بعدها ما غابت، أو بعده الضوّ — لازم تنتظر الظلام", RED))
    out.append(("أضواء المدينة قوية وتغطي النجوم الخافتة", RED) if c["light"] < 35 else
               ("أكو شوية أضواء مدن قريبة", GOLD) if c["light"] < 65 else ("المكان بعيد عن أضواء المدن", GREEN))
    out.append((f"السما مغيّمة ({a['cloud_cover']}%)", RED) if a["cloud_cover"] >= 50 else
               (f"أكو شوية غيوم ({a['cloud_cover']}%)", GOLD) if a["cloud_cover"] >= 20 else ("السما صافية بلا غيوم", GREEN))
    if a["aod"] >= 0.5:
        out.append(("أكو غبار (عجاج) يخلي السما باهتة", GOLD))
    out.append(("القمر مضوي ويطمس النجوم الخافتة", RED) if c["moon"] < 50 else
               ("القمر شوية مأثر", GOLD) if c["moon"] < 80 else ("القمر مو مأثر", GREEN))
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
        ("الجو والتلوث", [
            ("الغيوم والضباب", f"{a['cloud_cover']}%", g(a["cloud_cover"], 20, 50),
             "تحجب ضوء النجوم أو تضعفه، وحتى الغيوم الرقيقة تقلّل جودة الرصد."),
            ("الغبار والدخان", f"AOD {a['aod']} · الرؤية {a['visibility_km']} كم", g(a["aod"], 0.3, 0.5),
             "يشتّت ويمتص الضوء، فتقل شفافية الجو ويصير الهلال الخافت أصعب بالتمييز."),
            ("التلوث الضوئي", f"{lp_pct:.0f}%", g(lp_pct, 40, 70),
             "إنارة الشوارع والمباني والإعلانات ترفع سطوع السما، فتختفي النجوم الأضعف بالخلفية."),
            ("اضطراب الهواء", f"{a['seeing_arcsec']}″ · ريح فوق {a['wind250']} كم/س", g(a["seeing_arcsec"], 1.5, 2.5),
             "اختلاف حرارة وحركة طبقات الهواء يخلّي النجم يرتعش ويتوسّع ويفقد التفاصيل — يأثر أكثر على التلسكوب والتصوير."),
            ("الرطوبة", f"{a['humidity']}%", g(a["humidity"], 70, 85),
             "بخار الماء يمتص أطوال موجية معيّنة (خصوصاً تحت الحمراء)، والندى على العدسات يضعف الصورة."),
            ("حرارة الأسطح", "نصيحة", info,
             "الهواء الحار الصاعد من الإسفلت والأسطح يشوّه الصورة: ابتعد عنها، وخلي التلسكوب يبرد 30–60 دقيقة قبل الرصد."),
        ]),
        ("الضوء ومكان الجرم", [
            ("الشفق", twilight[1], twilight[0],
             "بعد الغروب تبقى السما مضيئة شوية. هذا أهم عامل لرصد الهلال الجديد لأن الهلال ضعيف وخلفيته مضيئة."),
            ("ضوء القمر", moon_txt, "🟢" if moon_s >= 80 else "🟠" if moon_s >= 50 else "🔴",
             "القمر يرفع سطوع السما ويصعّب رصد النجوم والأجرام الخافتة. أحسن رصد بليالي المحاق."),
            ("قرب الأفق", "فوق 30°", info,
             "ضوء الجرم القريب من الأفق يمر بغلاف جوي أطول فيضعف أكثر. شوف ارتفاع كل نجمة بقسم «المزيد ← شنو أشوف؟»."),
            ("عوائق الأفق", "مكان مفتوح", info,
             "المباني والجبال والأشجار تحجب الأجرام الواطية حتى لو الجو صافي، خصوصاً الهلال لأنه قريب من الأفق."),
        ]),
        ("نويز الكاميرا", [
            ("ضوضاء الفوتونات", "طبيعي", info,
             "عدد فوتونات الضوء الخافت يتذبذب، فيصعب تمييزه. الحل: تعريض أطول أو جمع صور كثيرة (Stacking)."),
            ("التيار المظلم", "بالحرارة", info,
             "الحساس يولّد إشارة حتى بدون ضوء. برّد الكاميرا وصوّر صور Dark frames وانطرحها من الصور."),
            ("ضوضاء القراءة", "بكل صورة", info,
             "إلكترونيات الحساس تضيف تشويش لكل صورة. صور أقل بتعريض أطول تقلّل أثرها."),
            ("البكسلات الساخنة", "نجوم زائفة", info,
             "نقاط أو أنماط زائفة. تتصلّح بصور المعايرة Dark و Flat."),
            ("الأشعة الكونية", "نقاط وخطوط", info,
             "تضرب الحساس وتترك نقاط أو خطوط مضيئة. جمع صور متعددة يشيلها."),
            ("الأقمار الصناعية", "خطوط", info,
             "تترك خطوط مضيئة خصوصاً وقت الشفق. صوّر بعد الظلام الكامل واستخدم Stacking."),
            ("أخطاء التركيب", "افحص", info,
             "اهتزاز الحامل، ضعف التتبّع، سوء التركيز (Focus)، واتّساخ العدسات كلها تخرّب الصورة."),
        ]),
    ]


def noise_sources(a, lp_pct, moon_tip):
    """Everything that degrades the sky here and now as a 0-100 'noise' level (higher = worse), plus how to reduce it."""
    c = a["components"]
    return [
        ("💡", "التلوث الضوئي", f"{lp_pct:.0f}%", lp_pct,
         "روح لمكان أظلم (شوف «وين أروح؟»)، باوع لجهة بعيدة عن المدينة، وبالتصوير استخدم فلتر تلوث ضوئي."),
        ("🌕", "ضوء القمر", f"مضاء {a['moon_illum']:.0%}" + (" · غايب" if a["moon_alt"] <= 0 else f" · ارتفاعه {a['moon_alt']:.0f}°"),
         100 - c["moon"], moon_tip),
        ("🌫️", "الغبار والعجاج", f"AOD {a['aod']}", float(np.clip((a["aod"] - 0.1) / 0.6, 0, 1) * 100),
         "انتظر يوم أو يومين بعد العجاج، وروح لمكان أعلى — الغبار يتجمّع قرب الأرض."),
        ("☁️", "الغيوم", f"{a['cloud_cover']}%", float(a["cloud_cover"]),
         "اختار ساعة خضرة من شريط الساعات بقسم «النتيجة»، أو جرّب ليلة ثانية."),
        ("💨", "اضطراب الهواء", f"{a['seeing_arcsec']}″", float(np.clip((a["seeing_arcsec"] - 1) / 2, 0, 1) * 100),
         "رصد النجوم العالية (فوق 45°)، ابتعد عن الإسفلت والأسطح الحارة، وخلي التلسكوب يبرد ساعة قبل الرصد."),
        ("💧", "الرطوبة والندى", f"{a['humidity']}%", float(np.clip((a["humidity"] - 60) / 35, 0, 1) * 100),
         "استخدم غطاء ندى أو سخانة للعدسة، وغطّي الأجهزة لما ما تستعملها."),
    ]


def noise_word(v):
    return ("قليل", GREEN) if v < 30 else ("متوسط", GOLD) if v < 60 else ("عالي", RED)


@st.cache_data(ttl=3600, show_spinner=False)
def moon_track(lat, lon, night):
    times = engine.local_night(night)
    return pd.concat([pd.DataFrame(dict(time=times)), engine.sky_geometry(lat, lon, times).reset_index(drop=True)], axis=1)


# ------------------------------------------------------------------ cached calls
@st.cache_resource(show_spinner="⏳ دا نجهّز خريطة الموصل…")
def cell_table():
    t = engine.cells()
    t["color"] = [[[243, 185, 65], [214, 136, 82], [143, 123, 214], [77, 107, 240]][int(c)] + [200] for c in t.lp_cls]
    t["label"] = [["سما مضوية", "سما نص ونص", "سما مظلمة", "سما مظلمة كلش"][int(c)] for c in t.lp_cls]
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


# ------------------------------------------------------------------ state
ss = st.session_state
now = datetime.now()
today = now.date() - timedelta(days=1) if now.hour < 6 else now.date()
DAY_OPTS = ["الليلة", "باچر", "بعد باچر", "📅 يوم ثاني"]
ss.setdefault("view", "ask")
ss.setdefault("q", dict(place=list(PLACES)[0], day="الليلة", date=today + timedelta(days=3), hour=22, obs="stargazing",
                        radius=100, pick=None))


def seed(key, val):
    if key not in ss:
        ss[key] = val


def reset_map():
    ss.map_v = ss.get("map_v", 0) + 1
    ss.q["pick"] = None


def night_of(day, other):
    return {"الليلة": today, "باچر": today + timedelta(days=1), "بعد باچر": today + timedelta(days=2)}.get(day or "الليلة", other)


def go():
    ss.q.update(place=ss.w_place, day=ss.w_day or "الليلة", date=ss.get("w_date", ss.q["date"]), hour=ss.w_hour,
                obs=ss.w_obs, radius=ss.w_rad)
    ss.view, ss.sec, ss._sec_last = "result", "النتيجة", "النتيجة"


def edit():
    ss.view = "ask"


def to(section):
    def _cb():
        ss.sec = ss._sec_last = section
    return _cb


def keep_sec():
    """Pills must always have one section selected (clicking the active pill would otherwise clear it)."""
    if ss.sec is None:
        ss.sec = ss.get("_sec_last", "النتيجة")
    ss._sec_last = ss.sec


def brand():
    st.markdown('<div class="brand"><div class="logo">🔭</div><div class="name">سما الموصل</div></div>', unsafe_allow_html=True)


CELLS = cell_table()

# ================================================================== ASK SCREEN
if ss.view == "ask":
    brand()
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title disp">تطلع ترصد الليلة؟</div>'
                '<div class="hero-sub">جاوب على 3 أسئلة بسيطة، واحنه نگلك تطلع لو لا، ومتى، ووين.</div>', unsafe_allow_html=True)
    for key_, val_ in [("w_place", ss.q["place"]), ("w_day", ss.q["day"]), ("w_date", ss.q["date"]),
                       ("w_hour", ss.q["hour"]), ("w_obs", ss.q["obs"]), ("w_rad", ss.q["radius"])]:
        seed(key_, val_)

    map_key = f"sky_map_{ss.get('map_v', 0)}"
    picked = ss.get(map_key, {}).get("selection", {}).get("objects", {}).get("cells", [])
    if picked:
        ss.q["pick"] = (picked[0]["lat"], picked[0]["lon"])

    with st.container(key="qbar"):
        c1, c2, c3, c4, c5 = st.columns([1.4, 1.4, 1.0, 0.9, 0.95], vertical_alignment="center")
        with c1:
            st.markdown('<div class="q-label">1 · وينك هسه؟</div>', unsafe_allow_html=True)
            st.selectbox("وينك هسه؟", list(PLACES), key="w_place", label_visibility="collapsed", on_change=reset_map)
        with c2:
            st.markdown('<div class="q-label">2 · شوكت تريد ترصد؟</div>', unsafe_allow_html=True)
            n_ = night_of(ss.w_day, ss.w_date)
            with st.popover(f"{DAYS[n_.weekday()]} {n_:%d/%m} · {HOUR_LABEL[ss.w_hour]}", use_container_width=True):
                st.segmented_control("اليوم", DAY_OPTS, key="w_day")
                if ss.w_day == "📅 يوم ثاني":
                    st.date_input("اختار اليوم", key="w_date", min_value=today, max_value=today + timedelta(days=14))
                st.select_slider("الساعة", options=list(HOUR_LABEL), key="w_hour", format_func=HOUR_LABEL.get)
        with c3:
            st.markdown('<div class="q-label">3 · شتريد تشوف؟</div>', unsafe_allow_html=True)
            st.selectbox("شتريد تشوف؟", list(TYPE_UI), key="w_obs", format_func=lambda k: TYPE_UI[k][0],
                         label_visibility="collapsed")
        with c4:
            st.markdown('<div class="q-label">شكد تبعد؟</div>', unsafe_allow_html=True)
            st.selectbox("شكد تبعد؟", [25, 50, 100], key="w_rad", format_func=lambda v: f"{v} كم", label_visibility="collapsed")
        with c5:
            st.button("شوف النتيجة  ‹", key="go", on_click=go, use_container_width=True)

    hint = TYPE_UI[ss.w_obs][1]
    if ss.q["pick"]:
        hint = f"📍 اخترت نقطة من الخريطة: <b>{place_name(*ss.q['pick'])}</b> — {hint}"
    st.markdown(f'<div class="q-hint">{hint}</div>', unsafe_allow_html=True)
    if ss.q["pick"]:
        st.button("↩️ رجوع لقائمة المناطق", on_click=reset_map, key="unpick")

    plat, plon = ss.q["pick"] or PLACES[ss.w_place]
    with st.container(key="mapwrap"):
        st.markdown('<div class="muted" style="padding:2px 6px 8px">أو اختار مكانك من الخريطة: اضغط على أي نقطة</div>',
                    unsafe_allow_html=True)
        st.pydeck_chart(pdk.Deck(
            layers=[pdk.Layer("ScatterplotLayer", CELLS, id="cells", get_position="[lon, lat]", get_fill_color="color",
                              get_radius=1900, pickable=True),
                    pdk.Layer("ScatterplotLayer", pd.DataFrame([dict(lat=plat, lon=plon, label="📍 انت هنا")]),
                              get_position="[lon, lat]", get_fill_color=[255, 255, 255], get_radius=3200, stroked=True,
                              get_line_color=[20, 20, 20], line_width_min_pixels=2, pickable=True)],
            initial_view_state=pdk.ViewState(latitude=plat, longitude=plon, zoom=7.3), map_style="dark",
            tooltip={"text": "{label}"}), key=map_key, on_select="rerun", selection_mode="single-object", height=420)
        st.markdown('<div class="legend"><span><i style="background:#f3b941"></i>سما مضوية</span>'
                    '<span><i style="background:#d68852"></i>نص ونص</span><span><i style="background:#8f7bd6"></i>مظلمة</span>'
                    '<span><i style="background:#4d6bf0"></i>سما مظلمة كلش</span></div>', unsafe_allow_html=True)
    st.stop()

# ================================================================== RESULT SCREEN
q = ss.q
lat, lon = q["pick"] or PLACES[q["place"]]
place = place_name(lat, lon) if q["pick"] else q["place"]
night = night_of(q["day"], q["date"])
hour, obs, radius = q["hour"], q["obs"], q["radius"]
when = datetime.combine(night + timedelta(days=1 if hour < 12 else 0), datetime.min.time()) + timedelta(hours=hour)

# --- engine (unchanged logic)
a = analyze(lat, lon, when, obs)
rec, rec_notes = recommend(lat, lon, radius, when, obs)
rec = rec[rec.score > a["score"]].reset_index(drop=True) if len(rec) else rec  # only spots better than here
k = a["cls"]
color, headline, short = LEVEL[k]
site_sqm = a["sqm"] if a.get("sqm") is not None else skyq.site_sqm(sqm_grid(), lat, lon)
lp_pct = float(np.clip((22.0 - site_sqm) / 6.0, 0, 1) * 100)
stars_here = float(skyq.visible_stars(skyq.limiting_mag(site_sqm, 7)))
stars_dark = float(skyq.visible_stars(skyq.limiting_mag(22.0, 7)))
hidden_pct = max(0.0, 100 * (1 - stars_here / stars_dark))
brighter = 10 ** (0.4 * (22.0 - site_sqm))
lp_level, lp_col = next((w, cl) for lim, w, cl in [(20, "خفيف جداً", GREEN), (40, "خفيف", GREEN), (60, "متوسط", GOLD),
                                                    (80, "عالي", RED), (101, "عالي جداً", RED)] if lp_pct < lim)
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
nz_word, nz_col = noise_word(total_noise)
bt = a["best_time"]
best = rec.iloc[0] if len(rec) else None

# --- header: brand · summary · change
h1, h2, h3 = st.columns([3, 5, 1], vertical_alignment="center")
with h1:
    brand()
with h2:
    st.markdown(f'<div class="summary">{TYPE_UI[obs][0]} · {place} · {DAYS[night.weekday()]} {HOUR_LABEL[hour]}</div>',
                unsafe_allow_html=True)
with h3:
    st.button("غيّر", key="edit", on_click=edit)
st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
seed("sec", "النتيجة")
st.segmented_control("القسم", SECTIONS, key="sec", label_visibility="collapsed", on_change=keep_sec)
sec = ss.sec or "النتيجة"

# ------------------------------------------------------------------ النتيجة
if sec == "النتيجة":
    left, right = st.columns([1.75, 1], gap="large")
    with left:
        st.markdown(f"""<div class="hero" style="--c:{color}; --s:{a['score']}">
          <div class="ring"><div class="in"><div><b>{a['score']}</b><small>من 100</small></div></div></div>
          <div><div class="badge" style="--c:{color}">{short}</div><div class="headline disp">{headline}</div></div></div>""",
                    unsafe_allow_html=True)
        st.markdown('<div class="reasons">' + "".join(
            f'<div class="rc"><span class="dot" style="background:{c_}"></span>{t_}</div>' for t_, c_ in reason_items(a))
            + "</div>", unsafe_allow_html=True)
        st.markdown('<h4 style="margin:26px 0 8px">🕒 كل ساعات الليلة <span class="muted" style="font-weight:400">'
                    '(الأخضر = زين)</span></h4>', unsafe_allow_html=True)
        cells_html = ""
        for r in a["hourly"].itertuples():
            col = [RED, GOLD, GREEN, BLUE][int(r.cls)]
            mark = ["❌", "⚠️", "✅", "⭐"][int(r.cls)]
            sel = " sel" if r.time == when else ""
            cells_html += f'<div class="hr{sel}" style="--c:{col}"><b>{mark}</b>{r.time:%H}:00</div>'
        st.markdown(f'<div class="strip">{cells_html}</div>', unsafe_allow_html=True)
    with right:
        if best is not None and int(best.score) >= a["score"] + 10:
            st.markdown(f"""<div class="card soft tip"><div class="k">نصيحتنا</div>
              <div class="place">روح لـ {place_name(best.lat, best.lon)}</div>
              <div><span class="big" style="color:{GREEN}">{int(best.score)}</span> <span class="unit"><span class="ltr">/100</span> هناك</span></div></div>""",
                        unsafe_allow_html=True)
            st.button("شوف الأماكن", key="to_spots", on_click=to("وين أروح؟"), use_container_width=True)
        elif bt:
            st.markdown(f"""<div class="card soft tip"><div class="k">نصيحتنا</div>
              <div class="place">أحسن وقت بهاي الليلة</div>
              <div class="big" style="color:{GREEN}">{bt[0]:%H:%M}–{bt[1]:%H:%M}</div></div>""", unsafe_allow_html=True)
        elif k <= 1:
            st.markdown('<div class="card soft tip"><div class="k">نصيحتنا</div><div class="place">جرّب ليلة ثانية أو ساعة ثانية — '
                        'هاي الليلة ما بيها وقت زين.</div></div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="card"><div class="row"><div><h4>النويز بمكانك</h4>
          <div class="muted">شنو يخرّب السما وشلون تقلله</div></div>
          <div class="big" style="color:{nz_col}">{total_noise}%</div></div></div>""", unsafe_allow_html=True)
        st.button("شوف النويز ‹", key="to_noise", on_click=to("النويز"))
        st.markdown(f"""<div class="card"><div class="row"><div><h4>التلوث الضوئي بمكانك</h4>
          <div class="muted">{lp_level} · يخفي {hidden_pct:.0f}% من النجوم</div></div>
          <div class="big" style="color:{lp_col}">{lp_pct:.0f}%</div></div>
          <div class="bar"><div style="width:{lp_pct:.0f}%; background:{lp_col}"></div></div>
          <div class="muted" style="font-size:.9rem">تشوف ~{stars_here:,.0f} نجمة بدل ~{stars_dark:,.0f} بسما مظلمة ·
          السما أضوى من الطبيعي بـ {brighter:,.0f} مرة</div></div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="card"><h4>منين اجت الـ {a['score']}؟</h4><div class="muted">4 عوامل وأوزانها</div></div>""",
                    unsafe_allow_html=True)
        st.button("شلون انحسبت؟ ‹", key="to_calc", on_click=to("شلون انحسبت؟"))

# ------------------------------------------------------------------ شلون انحسبت؟
elif sec == "شلون انحسبت؟":
    c_ = a["components"]
    w_ = engine.REGIONS["mosul"]["weights"][obs]
    cell_ = engine.nearest(engine.cells(), lat, lon)
    st.markdown(f'<div class="sec-title disp">منين اجت الـ {a["score"]}؟</div>'
                '<div class="sec-sub">كل عامل إله درجة من 100 ووزن. نضرب كل درجة بوزنها ونجمع.</div>', unsafe_allow_html=True)
    cards = [
        ("ظلام السما", c_["light"], w_[0],
         f"من صور ناسا الليلية + الذكاء الاصطناعي ({cell_.ai_score:.0f}) والنموذج الفيزيائي ({cell_.physics_score:.0f}، "
         f"سطوع {cell_.sqm:.1f})"),
        ("الجو", c_["weather"], w_[1], f"غيوم {a['cloud_cover']}% · غبار {a['aod']} · رطوبة {a['humidity']}%"),
        ("القمر", c_["moon"], w_[2],
         f"مضاء {a['moon_illum']:.0%} · " + ("غايب تحت الأفق" if a["moon_alt"] <= 0 else f"ارتفاعه {a['moon_alt']:.0f}°")),
        ("الارتفاع", c_["elevation"], w_[3], f"{a['elevation_m']} م فوق البحر"),
    ]
    st.markdown('<div class="fgrid">' + "".join(
        f"""<div class="fcard"><div class="t">{n}</div><div class="v">{s}<small class="ltr">/100</small></div>
        <div class="bar"><div style="width:{max(s, 2)}%; background:{GOLD}"></div></div>
        <div class="d">{d}</div><div class="f"><span>وزن {wt:.0%}</span><b class="ltr">+{s * wt:.1f}</b></div></div>"""
        for n, s, wt, d in cards) + "</div>", unsafe_allow_html=True)
    raw_ = sum(s * wt for _, s, wt, _ in cards)
    note_ = ""
    if a["sun_alt"] > -12:
        note_ = f"المجموع ≈ {raw_:.0f}، بس لأن الشمس بعدها ما غابت كفاية الدرجة تنقص تلقائياً لـ 10 كحد أقصى."
    elif a["cloud_cover"] >= 80:
        note_ = f"المجموع ≈ {raw_:.0f}، بس لأن الغيوم فوق 80% الدرجة تنقص تلقائياً لـ 20 كحد أقصى."
    scale = "".join(f'<div class="{"on" if i == k else ""}"><i style="background:{c}"></i>{lbl}</div>'
                    for i, (c, lbl) in enumerate([(RED, "0–25 ما تصلح"), (GOLD, "26–50 متوسطة"),
                                                  (GREEN, "51–75 زينة"), (BLUE, "76–100 ممتازة")]))
    st.markdown(f"""<div class="total"><div><span style="font-size:1.5rem;font-weight:700">المجموع</span>
      <span class="n">&nbsp;{a['score']}</span> <span class="muted"><span class="ltr">/100</span> · {short}</span></div>
      <div class="scale">{scale}</div></div>""", unsafe_allow_html=True)
    st.markdown(f'<div class="foot">{note_ + " " if note_ else "الأرقام مقرّبة، فممكن الجمع اليدوي يفرق درجة وحدة. "}'
                'الأوزان تتغيّر حسب نوع الرصد، وحالياً مختارة من الفريق — راح نتعلّمها من قياسات حقيقية بالمرحلة الجاية.</div>',
                unsafe_allow_html=True)

# ------------------------------------------------------------------ النويز
elif sec == "النويز":
    side, main = st.columns([1, 1.9], gap="large")
    with side:
        st.markdown(f"""<div class="sec-title disp">النويز بمكانك هسه</div>
          <div class="nzbig" style="--c:{nz_col}"><div class="n">{total_noise}%</div><div class="w">نويز {nz_word}</div></div>
          <p class="muted" style="margin:18px 0">النويز = كل شي يخرّب السما: ضوّ، قمر، غبار، غيوم، اضطراب هوا، رطوبة.
          كل ما يقل أحسن.</p>""", unsafe_allow_html=True)
        st.button("وين أروح حتى يقل؟", key="to_where", on_click=to("وين أروح؟"), use_container_width=True)
    with main:
        st.markdown('<p class="muted" style="margin-top:22px">اضغط على أي مصدر حتى تعرف شلون تقلله.</p>', unsafe_allow_html=True)
        html = ""
        for i, (e, name, val, lvl, tip) in enumerate(src):
            word, col = noise_word(lvl)
            html += (f'<details class="nz"{" open" if i == 0 else ""}><summary><b>{name}</b>'
                     f'<div class="bar" style="margin:0"><div style="width:{max(lvl, 1.5):.0f}%; background:{col}"></div></div>'
                     f'<span style="color:{col}">{lvl:.0f}%</span></summary>'
                     f'<div class="tipx"><span class="muted">{val} · نويز {word} — </span>{tip}</div></details>')
        st.markdown(html, unsafe_allow_html=True)
        st.markdown("""<details class="nz"><summary><b>📷 نويز الكاميرا</b><div></div><span class="muted">نصايح</span></summary>
          <div class="tipx"><ul>
          <li><b>ضوضاء الفوتونات:</b> صوّر صور كثيرة واجمعها (Stacking) بدل صورة وحدة، أو زيد وقت التعريض.</li>
          <li><b>التيار المظلم:</b> برّد الكاميرا، وصوّر Dark frames (نفس الإعدادات بس الغطاء مسدود) وانطرحها.</li>
          <li><b>ضوضاء القراءة:</b> صور أقل بتعريض أطول أحسن من صور كثيرة قصيرة كلش.</li>
          <li><b>البكسلات الساخنة:</b> صور المعايرة Dark و Flat تصلحها، و Dithering يساعد.</li>
          <li><b>الأشعة الكونية وخطوط الأقمار الصناعية:</b> Stacking بطريقة Sigma-clipping يشيلها، وصوّر بعد الظلام الكامل.</li>
          <li><b>الأجهزة:</b> ثبّت الحامل، ضبّط التتبّع والتركيز (Focus)، ونظّف العدسات قبل الرصد.</li></ul></div></details>""",
                    unsafe_allow_html=True)

# ------------------------------------------------------------------ وين أروح؟
elif sec == "وين أروح؟":
    st.markdown(f'<div class="sec-title disp">وين أروح حتى يقل النويز؟</div>'
                f'<div class="sec-sub">أحسن {len(rec) or ""} أماكن ضمن {radius} كم: سماها أظلم، والجو زين، وقريبة من شارع رئيسي.</div>',
                unsafe_allow_html=True)
    if len(rec):
        cols = st.columns(len(rec), gap="medium")
        for i, (col_, r) in enumerate(zip(cols, rec.itertuples())):
            lp_there = float(np.clip((22 - r.sqm) / 6, 0, 1) * 100)
            see_there = float(skyq.seeing(r.wind10, r.wind250))

            def line(label, here, there, fmt, tol):
                c = GREEN if there < here - tol else RED if there > here + tol else "#98a1b7"
                return (f'<div class="row"><span>{label}</span><span><span class="ltr">{fmt(here)}</span> ← '
                        f'<b class="ltr" style="color:{c}">{fmt(there)}</b></span></div>')
            with col_:
                st.markdown(f"""<div class="pcard{' first' if i == 0 else ''}">
                  <div class="row" style="align-items:flex-start"><div>
                    <span class="pbadge">{["المكان الأول", "المكان الثاني", "المكان الثالث"][i] if i < 3 else f"المكان {i + 1}"}</span>
                    <div class="pname">{place_name(r.lat, r.lon)}</div></div>
                    <div style="text-align:left"><div class="pscore">{int(r.score)}</div>
                    <div class="unit"><span class="ltr">/100</span> {LEVEL[int(r.cls)][2]}</div></div></div>
                  <div class="cmpbox">
                    {line("النويز", total_noise, 100 - int(r.score), lambda v: f"{v:.0f}%", 1)}
                    {line("التلوث الضوئي", lp_pct, lp_there, lambda v: f"{v:.0f}%", 3)}
                    {line("الغبار", a["aod"], r.aod, lambda v: f"{v:.2f}", 0.03)}
                    {line("الغيوم", a["cloud_cover"], r.cloud_cover, lambda v: f"{v:.0f}%", 5)}
                    {line("اضطراب الهواء", a["seeing_arcsec"], see_there, lambda v: f"{v:.1f}″", 0.1)}
                  </div>
                  <div class="meta"><span>{r.km:.0f} كم عنك</span><span>{r.road_km:.1f} كم من الشارع</span>
                    <span>ارتفاع {r.elevation_m:.0f} م</span></div>
                  <a class="gobtn" target="_blank"
                     href="https://www.google.com/maps/dir/?api=1&origin={lat},{lon}&destination={r.lat},{r.lon}">خذني هناك</a>
                </div>""", unsafe_allow_html=True)
        st.markdown('<div class="foot">الأسهم تقارن مكانك ← المكان المقترح: الأخضر = أقل نويز · الأحمر = أكثر.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="card">ماكو مكان أحسن من مكانك ضمن هاي المسافة. اضغط «غيّر» وجرّب 100 كم أو ليلة ثانية.</div>',
                    unsafe_allow_html=True)

# ------------------------------------------------------------------ كل العوامل
elif sec == "كل العوامل":
    t_, l_ = st.columns([2, 1], vertical_alignment="bottom")
    t_.markdown('<div class="sec-title disp">كل العوامل</div><div class="sec-sub" style="margin-bottom:6px">'
                'اضغط على أي عامل حتى تقرا الشرح.</div>', unsafe_allow_html=True)
    l_.markdown(f'<div class="legend2"><span><span class="dot" style="background:{GREEN}"></span>زين</span>'
                f'<span><span class="dot" style="background:{RED}"></span>سيء</span>'
                f'<span><span class="dot" style="background:{BLUE}"></span>نصيحة</span></div>', unsafe_allow_html=True)
    gcols = st.columns(3, gap="medium")
    for gcol, (title, rows) in zip(gcols, factor_groups(a, lp_pct)):
        dotc = {"🟢": GREEN, "🟠": GOLD, "🔴": RED}
        gcol.markdown(f'<div class="gtitle">{title}</div>' + "".join(
            f'<details class="fx"><summary><b><span class="dot" style="background:{dotc.get(stt, BLUE)}"></span>{name}</b>'
            f'<span>{val}</span></summary><p>{why_txt}</p></details>' for name, val, stt, why_txt in rows),
            unsafe_allow_html=True)
    st.markdown("""<div class="card" style="margin-top:14px"><h4>🌒 النجوم مو مثل الهلال</h4><div class="muted">
      للنجوم الخافتة أهم شي <b>ظلام السما</b>. للهلال أهم شي <b>تباينه ويّا الشفق</b> و<b>ارتفاعه فوق الأفق</b> — لهذا نحسبهم
      بطريقتين مختلفتين (قسم «المزيد ← الهلال»).</div></div>
      <div class="card"><h4>🛰️ لو التلسكوب بالفضاء؟</h4><div class="muted">تختفي الغيوم والغبار واضطراب الهواء، بس تبقى
      ضوضاء الحساس والأشعة الكونية والضوء الشارد والخلفية من الغبار بين الكواكب.</div></div>""", unsafe_allow_html=True)
    st.markdown('<div class="foot">المصادر: ناسا (VNP46A2) · Globe at Night · Open-Meteo · OpenStreetMap · ESO · STScI · USNO. '
                'اضطراب الهواء تقدير من سرعة الريح العالية، مو قياس مباشر.</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ المزيد
else:
    t1, t2, t3, t4, t5 = st.tabs(["✨ شنو أشوف؟", "🌒 الهلال", "💡 خلّي سما مدينتك أحلى", "🙋 شاركنا", "📖 شنو يعني؟"])
    SQM = sqm_grid()
    sqm = skyq.site_sqm(SQM, lat, lon)
    with t1:
        st.caption("النجوم والكواكب اللي تبين من مكانك بهاي الليلة، ووين تباوع.")
        tg, tg_notes = targets(lat, lon, night, sqm)
        if tg_notes:
            st.warning("⚠️ توقع الجو ما اشتغل لهاي القائمة، فهي تفترض جو طبيعي (غالباً صافي). اعتمد على «النتيجة» للغيوم.")
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
            st.warning("من مكانك ما راح يبين شي واضح بهاي الليلة — روح لواحد من الأماكن بقسم «وين أروح؟».")
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
                st.markdown(f'<div class="spot" style="--c:{RED}"><div class="n">🌑 مساء {d}: {HILAL_AR["D"]}</div>{facts}</div>',
                            unsafe_allow_html=True)
        st.caption("شنو يقرر رؤية الهلال؟ الاستطالة، ارتفاعه فوق الأفق، مدة بقائه بعد الغروب، سماكته وسطوعه مقابل الشفق، "
                   "وصفاء الأفق وخبرة الراصد ونوع الجهاز. عمر الهلال وحده ما يكفي، وحساب موقعه بدقة ما يعني ضمان رؤيته.")
    with t3:
        st.markdown("لو المدينة تقلل الإضاءة اللي ما إلها داعي (ضوّ يطلع للسما بدل الشارع)، شگد تصير السما أحلى؟")
        cut = st.segmented_control("شكد نقلل؟", [0.2, 0.4, 0.6], default=0.4, format_func=lambda v: f"نقلل {v:.0%}") or 0.4
        em = skyq.emission_map(skyq.fetch_black_marble())
        after_ = skyq.site_sqm(skyq.sqm_map(skyq.intervention(em, lat, lon, 30, 0, cut, False)), lat, lon)
        n0, n1 = skyq.visible_stars(skyq.limiting_mag(sqm, 7)), skyq.visible_stars(skyq.limiting_mag(after_, 7))
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
- **النويز:** كل شي يخرّب السما (ضوّ، قمر، غبار، غيوم، اضطراب هوا، رطوبة). النويز الكلي = 100 − الدرجة.
- **درب التبانة 🌌:** مجرّتنا! تبين كشريط أبيض ضبابي بس بمكان مظلم كلش.
- **الغبار / العجاج 🌫️:** يخلي السما باهتة ويطمس النجوم الخافتة.
- **القمر 🌕:** لما يكون بدر يضوّي السما هواية، فأحسن رصد يكون بليالي المحاق (بلا قمر).
- **شلون نحسبها؟** نموذج ذكاء اصطناعي (XGBoost) تعلّم من **15 ألف قياس** حقيقي للسما، ويّا توقعات الجو والغبار والقمر.
""")

if a["notes"] or rec_notes:
    st.markdown('<div class="foot">ℹ️ بعض توقعات الجو ما متوفرة لهذا الوقت، فاستخدمنا المعدل الطبيعي لجو الموصل.</div>',
                unsafe_allow_html=True)
