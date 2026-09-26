"""Main Arabic crescent planning experience for Iraq."""
from datetime import datetime, timedelta, timezone
import json

import pandas as pd
import streamlit as st
import crescent
import hijri

st.markdown("""<style>
[data-testid="stMainBlockContainer"] {direction:rtl;text-align:right;max-width:1150px}
.stApp {background:radial-gradient(ellipse at top right,#183734 0%,#0b1020 65%)}
h1,h2,h3 {color:#f3e5bf!important}
.hilal-intro {border-right:4px solid #e5bd70;padding:12px 20px;margin:10px 0 24px;
background:#132925;border-radius:12px;font-size:1.1rem;line-height:1.9}
[data-testid="stMetric"] {background:#172438;padding:16px;border-radius:14px}
[data-testid="stMetricValue"] {font-size:1.6rem}
[data-testid="stForm"] {background:#111e30}
[data-testid="stDownloadButton"] button {width:100%}
@media(max-width:640px){h1{font-size:1.9rem!important}.hilal-intro{padding:10px;font-size:1rem}}
</style>""", unsafe_allow_html=True)

st.caption("سما الموصل / هلال العراق")
st.title("🌙 نرصد الهلال… ونستعد لرمضان")
st.markdown('<div class="hilal-intro">وين نرصد؟ شوكت نطلع؟ وهل الغبار يخفي الهلال؟<br>'
            'خطّط لتحري هلال رمضان والعيد حسب مدينتك، وسجّل مشاهدتك حتى نحسّن الرصد بالعراق.</div>',
            unsafe_allow_html=True)
st.info("توقع إمكانية الرؤية مو إعلان لبداية رمضان أو العيد. ما عدنا إعلان رسمي موثّق مربوط بالنظام حاليًا؛ "
        "تابع الجهة الدينية التي تعتمدها.")

today = datetime.now(timezone(timedelta(hours=3))).date()
current_year, current_month, _ = hijri.gregorian_to_hijri(today)
with st.form("campaign_controls"):
    a, b, c = st.columns(3)
    event = a.selectbox("المناسبة", list(crescent.EVENTS), format_func=lambda k: crescent.EVENTS[k][1])
    year = b.number_input("السنة الهجرية", min_value=1400, max_value=1500,
                          value=current_year + (1 if current_month > 9 else 0), step=1)
    city = c.selectbox("المدينة", list(crescent.CITIES))
    custom = st.checkbox("أحدد موقع الرصد بإحداثيات خاصة")
    d, e = st.columns(2)
    custom_lat = d.number_input("خط العرض", 29.0, 38.0, 36.34, format="%.5f")
    custom_lon = e.number_input("خط الطول", 38.0, 49.0, 43.13, format="%.5f")
    st.caption("الإحداثيات تُستخدم عند تفعيل الموقع الخاص. كل الأوقات بتوقيت بغداد UTC+3.")
    submitted = st.form_submit_button("اعرض خطة تحري الهلال", type="primary", use_container_width=True)


@st.cache_data(ttl=1800, show_spinner=False)
def calculate(event, year, lat, lon, as_of):
    conjunction, first = crescent.campaign(event, year)
    return conjunction, crescent.plan(lat, lon, first, today=as_of)


if submitted:
    st.session_state.pop("crescent_report_download", None)
    lat, lon = (custom_lat, custom_lon) if custom else crescent.CITIES[city]
    try:
        with st.spinner("نحسب أمسيات التحري وموقع الهلال…"):
            conjunction, frame = calculate(event, year, lat, lon, today)
        st.session_state["crescent_result"] = dict(event=event, year=year, lat=lat, lon=lon,
            city="موقع خاص" if custom else city, conjunction=conjunction, frame=frame,
            calculated_at=datetime.now(timezone(timedelta(hours=3))).isoformat(timespec="minutes"))
    except Exception:
        st.session_state.pop("crescent_result", None)
        st.error("تعذر حساب الخطة الآن. حاول مرة ثانية؛ لن نعرض موعدًا غير محسوب.")

result = st.session_state.get("crescent_result")
if result:
    frame = result["frame"]
    st.subheader(f'{crescent.EVENTS[result["event"]][1]} {result["year"]} · {result["city"]}')
    st.caption(f'الخطة المحسوبة للموقع {result["lat"]:.4f}, {result["lon"]:.4f} · آخر طلب حساب: '
               f'{result["calculated_at"]} بتوقيت بغداد. تغيير الخيارات يحتاج ضغط زر عرض الخطة.')
    candidates = frame[frame.zone.isin(["A", "B", "C"])]
    if len(candidates):
        first = candidates.iloc[0]
        st.success(f'أول مساء مرشح فلكيًا ضمن الخطة: {first["date"]:%Y-%m-%d} · {crescent.ZONES[first["zone"]]}')
    else:
        st.warning("ماكو مساء مرشح ضمن هذه الخطة حسب النموذج. هذا لا يثبت بداية الشهر أو إكماله.")
    if result["event"] == "adha":
        st.info("هنا نخطط لتحري هلال ذي الحجة. عيد الأضحى يوم 10 ذي الحجة، وموعده المعتمد يحتاج إعلان الجهة المختصة.")
    st.caption(f'الاقتران الفلكي: {result["conjunction"] + timedelta(hours=3):%Y-%m-%d %H:%M} بتوقيت بغداد. '
               'الاقتران وحده لا يعني أن الهلال قابل للرؤية.')
    for _, row in frame.iterrows():
        with st.container(border=True):
            st.markdown(f'### مساء {row["date"]:%Y-%m-%d}')
            st.write(f'**بسماء صافية:** {crescent.ZONES[row["zone"]]}')
            st.caption(row["weather_status"])
            if pd.notna(row.get("local_zone")):
                st.write(f'**بعد تقدير أثر الطقس:** {crescent.ZONES[row["local_zone"]]}')
                st.caption(f'الغيوم: {row["cloud_cover"]:.0f}% · العمق البصري للهباء AOD: {row["aod"]:.2f}')
            cols = st.columns(4)
            for col, label, key in zip(cols, ["غروب الشمس", "أفضل وقت هندسي", "غروب القمر", "المكث بعد الغروب"],
                                       ["sunset", "best_time", "moonset", "lag_min"]):
                value = row.get(key)
                text = "غير متاح" if pd.isna(value) else (f"{value:.0f} دقيقة" if key == "lag_min" else value.strftime("%H:%M"))
                col.metric(label, text)
            if pd.notna(row.get("moon_az")):
                st.write(f'🧭 اتجاه الهلال: **{row["moon_az"]:.1f}°** من الشمال باتجاه عقارب الساعة · '
                         f'الارتفاع: **{row["moon_alt"]:.1f}°** · الاستطالة: **{row["arcl"]:.1f}°**')
                st.caption("الاتجاه والارتفاع عند أفضل وقت هندسي. الأفق المحلي والمباني غير محسوبة؛ تأكد من انكشافه.")
    st.download_button("نزّل جدول الرصد CSV", frame.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f'hilal-{result["event"]}-{result["year"]}.csv', mime="text/csv")
    with st.expander("📍 الموقع وتجهيز الرصد"):
        st.map(pd.DataFrame([dict(lat=result["lat"], lon=result["lon"])]), zoom=8)
        st.write("اختَر مكانًا مسموحًا وآمن الوصول، بأفق مكشوف باتجاه الهلال. افحص الغبار والغيوم قبل الخروج، "
                 "وثبّت الجهاز وجهّز الساعة والبوصلة. الموقع المختار ليس توصية ميدانية مؤكدة.")
        st.warning("لا توجّه الناظور أو التلسكوب نحو الشمس. ابدأ البحث البصري بعد غروب الشمس.")
    with st.expander("📝 وثّق محاولة رصدك", expanded=False):
        st.caption("السجل يُجهّز للتنزيل على جهازك فقط؛ لا يُرسل لجهة رسمية ولا يُنشر كمشاهدة مؤكدة.")
        with st.form("crescent_report"):
            observed_date = st.date_input("تاريخ الرصد", value=min(today, frame.iloc[0]["date"]), max_value=today)
            observed_time = st.time_input("وقت الرصد · بغداد")
            outcome = st.selectbox("النتيجة", ["لم أرَ الهلال", "رأيت جسمًا محتملاً وأحتاج مراجعة", "رأيت الهلال بحسب تقديري"])
            equipment = st.selectbox("وسيلة الرصد", ["العين المجردة", "ناظور", "تلسكوب", "كاميرا"])
            horizon = st.selectbox("حالة الأفق", ["صافٍ ومكشوف", "غبار أو ضباب", "غيوم", "عوائق أرضية"])
            notes = st.text_area("ملاحظات", max_chars=2000)
            save = st.form_submit_button("جهّز سجل المشاهدة")
        if save:
            st.session_state["crescent_report_download"] = json.dumps(dict(
                event=result["event"], hijri_year=result["year"], latitude=result["lat"], longitude=result["lon"],
                observed_at=f"{observed_date}T{observed_time.isoformat()}+03:00", outcome=outcome,
                equipment=equipment, horizon=horizon, notes=notes, verification="unverified_observer_report"),
                ensure_ascii=False, indent=2)
        if "crescent_report_download" in st.session_state:
            st.download_button("نزّل سجل المشاهدة JSON", st.session_state["crescent_report_download"],
                               "hilal-observation.json", "application/json")
else:
    st.caption("اختَر مدينتك والمناسبة واضغط عرض الخطة. نعرض ثلاث أمسيات حول الاقتران بدل موعد ثابت لبداية الشهر.")

with st.expander("شلون نحسب؟ وشنو يختلف عن التقويم؟"):
    st.write("التقويم يعطي تاريخًا؛ هلال العراق يساعدك تخطط لمحاولة الرصد: موقع الهلال، وقت غروبه، "
             "أفضل وقت هندسي، والغيوم والغبار عندما تتوفر توقعات قريبة. لا نستخدم تقييم رصد النجوم للحكم على الهلال.")
    st.write("نستخدم حسابات Astropy ومعيار عودة للرؤية. تصنيف الرؤية ليس نسبة احتمال. تعديل الطقس تجريبي، "
             "وخارج أربعة أيام نعرض الهندسة فقط. تسمية الشهر تعتمد تقويمًا حسابيًا لتحديد دورة القمر، وليس لإعلان بدايته.")
    st.markdown("[عوامل رؤية الهلال — المرصد البحري الأمريكي](https://aa.usno.navy.mil/faq/crescent)")
    st.write("الخطوة القادمة: مقارنة التوقعات بمحاولات رصد موثقة من العراق، بما فيها المحاولات التي لم يُرَ فيها الهلال، "
             "ومراجعتها مع راصدين مختصين قبل قياس الدقة أو إطلاق تنبيهات عامة.")
