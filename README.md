# SkyQ Mosul — light pollution & observation-quality prototype (study area: Mosul / Nineveh)

ASI Hackathon · Challenge 1: التلوث الضوئي وصعوبة الرصد الفلكي

**Problem.** Iraqi observers (e.g. University of Baghdad, Dept. of Astronomy & Space) lose nights to city sky glow,
dust/haze, clouds, moonlight and turbulence. The usual way to find out a night is bad is to drive out and try.

**Two pages:** **سما العراق** (public, Iraqi Arabic): pick your city (or paste Google Maps coordinates), nothing else.
You get your sky quality, tonight's verdict and the reason, what you can see and when/where to look, the best night this
week, the next hilal, and the nearest dark site. **تفاصيل النموذج (للحكام)**: every control plus all metrics and validation.

**What it does**

| Tab | Answers | How |
|---|---|---|
| Light-pollution map & site finder | *Where* is the sky dark? | NASA VIIRS Black Marble night lights → Walker-law (d^-2.5) sky-glow model → SQM (mag/arcsec²) + Bortle for every 2 km pixel; ranks the darkest sites within driving distance |
| Observation planner | *When* should I observe target X? | Hourly Open-Meteo weather + CAMS aerosol (AOD/dust) + Astropy Sun/Moon/target geometry → RandomForest quality score → recommended window or "don't go" |
| Planner → **Hilal** target | *Can the new crescent be seen, when and where?* | Next conjunction → sunset/moonset, lag, age → Odeh (2004) criterion at best time (Yallop q cross-check) → downgraded for forecast cloud and dust extinction → zone A–D, look-direction (alt/az) |
| Responsible-lighting scenario | *How much* would better lighting help? | DarkSky principles (shielding, dimming, warm CCT) applied around the site → before/after SQM, Bortle, naked-eye limit, visible stars |
| Model & data | How good and how trustworthy is it? | Held-out MAE/R²/accuracy, feature importances, sources, units, limits; retrain with field logs |

Field data goes in two ways: an **SQM reading** in the sidebar overrides the satellite estimate (and shows the model's
error), and **observation logs** (planner → "Log a real observation") are added to training at 50× weight.

## Business flow (Iraq Sky AI — Mosul)

**Input:** place (list, map tap or Google Maps coordinates) · max distance (25/50/100 km) · date + hour ·
observation type (stargazing / Milky Way / astrophotography / meteors).

**Data:** NASA Black Marble **VNP46A2** daily night lights (median of the last ~7 weeks, via NASA GIBS) ·
Open-Meteo weather (cloud, humidity, visibility) + CAMS dust (AOD) · Open-Meteo elevation (Copernicus DEM) ·
OpenStreetMap major roads · Astropy Sun/Moon · Globe at Night ground truth.

**Scoring engine (`engine.py`):** Sky Score 0–100 = weighted light pollution + weather + Moon + elevation
(weights depend on observation type), with automatic caps for daylight/twilight and overcast.
Classes: 0–25 Poor 🔴 · 26–50 Moderate 🟠 · 51–75 Good 🟢 · 76–100 Excellent ⭐.
Light pollution = mean of the XGBoost classifier and the calibrated physics sky-glow model, both on the recent VNP46A2 lights.

**Recommendation engine:** every ~4 km cell within the radius that is ≤ 5 km from a major road (reachable) →
the 25 darkest well-spread candidates → one multi-point forecast call → ranked by Sky Score at the chosen time → top 3,
each with a Google Maps route.

**Output:** interactive map (tap any point) + result card (score, class, light pollution, cloud, Moon impact, elevation,
best time window, suitable-for ✓/✗) + hourly chart + best spots.

**Why the recent lights matter:** in the 2016 Black Marble composite, Hamdaniya (Qaraqosh) and Tel Afar are dark
(0.00 / 0.01 on the model's light scale, when they were ISIS-held or displaced); in the 2026 VNP46A2 composite both are fully lit (1.0).
Central Mosul changes little (0.97 → 1.0). The recent data is mapped onto the model's training scale by quantile matching.

**API (`api.py`, FastAPI):** `uvicorn api:app --port 8000` → http://localhost:8000/docs

| Endpoint | Example |
|---|---|
| `GET /sky-quality` | `/sky-quality?lat=36.34&lon=43.13&date=2026-09-25&time=22:00&type=milky_way` → score, class, light_pollution, cloud_cover, moon_impact, elevation, best_time, suitable_for, recommended |
| `GET /best-locations` | `/best-locations?lat=36.34&lon=43.13&radius=100&time=22:00&type=milky_way` → ranked spots |
| `GET /pollution-map` | all Nineveh cells with light-pollution score + class |
| `GET /observation-conditions` | `/observation-conditions?lat=36.34&lon=43.13&date=2026-09-25` → hourly score + best window |

Prototype scope: Streamlit is the frontend and distances are computed in Python (no Next.js/PostGIS/deploy).

## Run

```bash
pip install -r requirements.txt
python skyq.py        # self-check: physics asserts, map calibration, trains model, plans tonight
python skyclass.py    # builds satellite+ground dataset, compares LR/RF/XGBoost (spatial CV), saves best (~2 min)
python engine.py      # builds the Nineveh cell table (recent VNP46A2, elevation, roads) + engine self-check
uvicorn api:app --port 8000   # optional REST API
streamlit run app.py
```

First run downloads the Black Marble tile (cached in `data/`). Weather/aerosol need internet; if unavailable or the
date is beyond the forecast horizon (16 d weather, 5 d aerosol) the app falls back to climatology and says so.

## Sky-quality classifier (AI) — satellite → ground (`skyclass.py`)

- **Task:** 11 VIIRS Black Marble light-environment features (pixel light, mean light within 2–100 km, max light
  within 10/25 km, Walker sky-glow, distance to nearest lit area) → **Poor / Moderate / Good / Excellent**, plus a
  0–100 score (expected class). Tap any point on the home-page map to see it; the best spots within 100 km are ranked.
- **Ground truth:** 15,598 Globe at Night observations from 2012–2021 across the Mediterranean and Middle East
  (3,165 SQM-meter readings, which are preferred, and 12,433 naked-eye counts). Classes: SQM <18 / <19.5 / <21 / ≥21.
- **Validation:** spatial GroupKFold over 2°×2° blocks (neighbouring pixels are never split between train and test),
  plus a separate Iraq hold-out.

| Model (spatial CV) | Macro-F1 | Accuracy | Within ±1 class | Iraq hold-out Macro-F1 |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.369 | 0.42 | 0.73 | 0.30 |
| Random Forest | 0.361 | 0.39 | 0.73 | 0.26 |
| **XGBoost (selected)** | **0.383** | 0.40 | **0.80** | 0.26 (24 pts) |
| Physics-only rule (Walker glow thresholds) | 0.380 | 0.44 | 0.76 | – |
| Always "Poor" | 0.161 | 0.47 | 0.70 | – |

**Honest reading:** the target was Macro-F1 ≥ 0.80, and we reached 0.38. XGBoost ties the physics rule on Macro-F1 and
beats it on "within one class" (0.80 vs 0.76). The limit is the inputs, not the model choice: the free Black Marble
layer is a 2016 visual composite that saturates in cities, and naked-eye citizen labels are noisy. Known quirk: the
empty western desert scores "Good" rather than "Excellent", because the training data has almost no sites that far
from any light. Next steps: VNP46A2 radiance from Earthdata, plus SQM readings from Iraq, which the in-app
"ساعدنا نحسّن النموذج" form collects and the next `python skyclass.py` run uses.

## Model results (held-out / cross-validated; full JSON via Model tab → "Report")

**Observation-quality model** (Random Forest, 16,000 train / 4,000 test)

| Metric | Value | | Metric | Value |
|---|---|---|---|---|
| Accuracy (GOOD vs not) | 0.955 | | MAE (score) | 0.046 |
| Precision | 0.929 | | RMSE | 0.072 |
| Recall | 0.860 | | R² | 0.943 |
| **F1** | **0.893** | | 3-class accuracy | 0.910 |
| ROC-AUC | 0.990 | | Macro-F1 (poor/fair/good) | 0.875 |
| 5-fold CV F1 | 0.88 ± 0.01 | | Weighted-F1 | 0.912 |

Baselines on the same test set: *"clear sky + dark + target up"* F1 0.59; *"+ no bright Moon"* F1 0.52 → the model's
F1 0.89 shows combining dust, humidity, light pollution, Moon geometry and airmass beats the rules observers use today.

**Light-pollution map vs real field data** (45 Globe at Night citizen observations 2009–2025 in the region + 6 SQM-meter readings)

| Metric | Value |
|---|---|
| Leave-one-out MAE, naked-eye limiting mag | 1.46 (was 1.82 before calibration) |
| Bias | −0.09 mag · SQM meters: 0.00 |
| Within ±1 mag | 47% |
| Spearman ρ | 0.40 |
| Dark-sky (NELM ≥ 4) accuracy / F1 | 0.69 / 0.46 |

**Hilal:** 4/4 clear-cut historical evenings correct (computed for Mosul) (Ramadan 1444, Shawwal 1445: evening before, evening seen).

Scenario: 70% shielding + 30% dimming + 3000 K within 40 km of Baghdad → +1.1 mag/arcsec² (sky 2.7× darker, 15.8 → 16.9); the core stays Bortle 9, so dark-site trips are still needed.

## Limits (be honest in the pitch)

- Training labels are **physics-generated** (limiting-magnitude model with cloud, humidity and seeing penalties), because
  no labelled Iraqi observation dataset exists yet. The accuracy numbers above measure how well the model fits that
  physics, not reality. Real logs replace it over time.
- Sky glow has one fitted parameter (`GLOW_SCALE`, from Globe at Night). Single-site error is about ±1.5 mag, because the
  field data are coarse citizen estimates and the Black Marble PNG saturates in city cores. Upgrade: VNP46A2 radiance
  from Earthdata, plus more SQM-meter readings from Iraqi sites.
- The quality-model scores (F1 0.89 etc.) measure how well it reproduces the physics label on unseen nights, **not**
  agreement with real observers yet. The planner's field-log loop is how that gets measured.
- Hilal: Odeh's criterion is validated by its own 737-observation fit. The 4 historical checks are sanity checks. The
  cloud/dust downgrade is our addition and still needs calibrating against Iraqi sighting reports.
- Seeing is estimated from jet-stream wind. Next step: measure star FWHM on the telescope's own frames and feed it as a feature.
- Black Marble is the 2016 composite, so it does not show newer lighting.
