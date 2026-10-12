# SkyQ Mosul

A light-pollution and observation-quality planning tool for Mosul/Nineveh, built for the ASI Hackathon (Challenge 1: light pollution and astronomical observation difficulty). It helps Iraqi stargazers figure out whether a given night is worth driving out for, instead of finding out the hard way.

## Quick start

```bash
git clone https://github.com/baraa0abd/Space.git
cd Space
pip install -r requirements.txt
streamlit run app.py
```

Run the crescent-observation test suite:

```bash
python -m unittest test_crescent -v
```

## What it does

**The problem:** observers (e.g. University of Baghdad's Dept. of Astronomy & Space) lose observing nights to city sky glow, dust/haze, clouds, moonlight, and turbulence — and the usual way to find out a night is bad is to drive out and try.

Two experiences:
- **سما الموصل** (public, Iraqi Arabic, for beginners) — answer three questions (where, when, what you want to see) and get one clear answer: go or not, why, the best hour, and the best nearby spots.
- **تفاصيل النموذج** (for judges/power users) — every control, plus the full metrics and validation behind the recommendation.

**Primary feature — هلال العراق (Ramadan/Eid crescent planner):** an Arabic crescent-observation planner for Ramadan, Shawwal/Eid al-Fitr, and Dhu al-Hijjah/Eid al-Adha. Pick an Iraqi city (or raw coordinates) and a Hijri year to get three candidate observing evenings, sunset/moonset times and pointing direction, a CSV export of the plan, and a downloadable (unverified) observation record.

This is explicitly a **visibility planning tool, not an official month-announcement service** — atmospheric adjustments are experimental and only shown for complete near-term inputs; there's no connection to an official announcement feed or a verified site-ranking database. See [`plans/ramadan-feature.md`](plans/ramadan-feature.md) for the full pitch, how it differs from alternatives, and the validation roadmap.

## Tech stack

Python, Streamlit (UI), with a light ML component (`skyclass.joblib` — a trained sky-condition classifier) and real satellite/light-pollution reference data (NASA Black Marble, Globe at Night) for the Mosul/Nineveh and reference regions.

## Honest status

The data and the underlying sky-condition model are real and trained on real measurements — what hasn't yet happened is validating the final recommendation score against actual observed nights in Mosul. That validation step is what would turn this from a hackathon prototype into something an observing club could actually rely on.

## License

No license file is currently included.
