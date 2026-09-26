# هلال العراق — Ramadan and Eid as the primary experience

## Problem and pitch
**Hook:** قبل ما نطلع نتحرى الهلال: وين نروح، شوكت نرصد، وهل الغبار يسمح؟

Iraqi crescent observers struggle to plan Ramadan and Shawwal sighting attempts because low-altitude crescent geometry, twilight, dust, clouds and local horizon obstructions interact. The consequence is wasted trips and uncertain observations. Quantify failed trips and hours lost through interviews; no measured reduction is claimed yet. Difficulty observing is one source of uncertainty: religious criteria and the authority followed also affect month announcements.

**Solution:** an Arabic, Iraq-first observation planner that brings geometry, near-term atmospheric conditions and structured observation records into one workflow. It supports observation and does not determine religious dates.

## Shipped in this change
- Primary navigation and landing page for Ramadan, Shawwal/Eid al-Fitr and Dhu al-Hijjah/Eid al-Adha.
- City presets across Iraq, custom coordinates, Hijri year and event selection.
- Three calculated evenings from the local conjunction date; sunset, moonset, lag, best geometric time, altitude and azimuth.
- Clear-sky criterion separated from experimental atmospheric adjustment. Beyond four days, or with fallback data, no weather-adjusted verdict is presented.
- Explicit unavailable official-announcement state, with no invented date or automatic 30-day completion.
- CSV plan export, location map and local JSON observation report export. Reports are unverified and are not submitted or published.
- Existing stargazing and expert pages remain accessible. No hosting or database service has been added.

## User journey / web states
Open app → choose occasion/year/city → calculate → compare evenings → inspect location/horizon → export plan → attempt observation → export report.
Changing form fields requires recalculation; the result identifies the parameters actually used. Loading, calculation failure, absent geometry, unavailable weather and no eligible evening are distinct states. Baghdad time applies throughout. Layout uses RTL and responsive Streamlit columns.

## Difference from alternatives (positioning, not an exclusivity claim)
| Alternative | What this workflow adds |
|---|---|
| Fixed Hijri calendar | A local observing plan rather than a date alone |
| General weather forecast | Crescent geometry and relevant evening times alongside atmosphere |
| General stargazing / dark-sky planner | Crescent-specific criterion; twilight and horizon matter differently |
| Existing crescent visibility maps | Iraq-focused Arabic workflow and structured field records; many maps already use scientific visibility criteria |

Do not claim to be the first or uniquely accurate without a separate competitor study and field validation.

## Evidence and value
Existing astronomy calculations are reused; model weather penalties remain experimental. The repository's four historical examples are sanity checks, not local accuracy measurements. No interviews, user tests or successful Iraqi field trials are claimed for this new flow.
Measure planning time, wasted journeys, correctly identified usable windows, false positive sightings and report completeness. Establish baselines before assigning improvement percentages. Separate naked-eye, optical-aid and camera observations.

## Next validation and build phases
1. Interview 5–10 Iraqi observers about decisions, equipment and which authority they follow. Observe prototype use; record task completion and confusion between prediction and announcement.
2. Pilot across Mosul, Baghdad and a third region over multiple lunar months (not Ramadan alone). Log negative attempts, time, coordinates, equipment, horizon/weather and reviewer-confirmed outcomes. Obtain consent before collecting precise locations or images centrally.
3. Validate with held-out dates and sites, compare against geometry-only criteria, report precision/recall and uncertainty by equipment and region. Do not train and test on the same observing campaign.
4. Add a moderated database and image evidence with access control, retention/deletion policy, duplicate review and upload validation. Current downloadable reports are not a shared database.
5. Add verified authority announcements as separate records carrying authority, original source URL, publication time and reviewer. Support differing announcements without merging them into one supposedly official answer.
6. Add opt-in forecast-change reminders only after reliable data freshness tracking. Add terrain/horizon screening and site comparison after field access verification; do not rank random dark locations as safe or accessible.

## Scope and religious-date handling
Ramadan: plan the crescent opening Ramadan. Fitr: plan the Shawwal crescent. Adha: plan Dhu al-Hijjah; Eid is its tenth day after an accepted month start. No countdown to a confirmed holiday until a sourced authority announcement exists. No automatic inference of month completion from a failed local observation or three simulated evenings. Prayer times are outside this feature.

## Scientific reference
https://aa.usno.navy.mil/faq/crescent
