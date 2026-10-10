# Tides at the Santa Cruz Wharf and in Monterey Bay — September 2026

*UCSC Ocean Sciences · prepared for the 2026–27 El Niño report series ·
analysis script `scripts/santa_cruz_tides.py`, numbers in `tide_stats.json`*

*Revised 20 September 2026 in response to comments from David Barcelo and Mike
Beck: the City of Santa Cruz's own wharf gauge, the perigean as well as the nodal
cycle, water levels in feet and inches rather than only against NOAA's flood
lines, and a summary up front.*

---

## Summary

- **The water is standing high, but the tide is not the reason.** The
  astronomical tide is near the top of its 18.6-year cycle, and that is worth
  only about **1.4 inches** on the year's biggest tide. Everything else is the
  ocean itself.
- **Over the past year the high tide has arrived about 4½ inches above the tide
  tables** (+11.6 cm), against a normal of under an inch. A quarter of days ran
  6 inches or more above prediction.
- **2026 already holds the record for days at flood level** — 10 days above
  NOAA's minor high-tide-flooding level, against 42 days in total since 1980 —
  and three of them were in **June and July**, which is not something summer
  normally does here.
- **3 January 2026 produced the third-highest high water in 47 years** (2.362 m
  above MLLW), behind only the two days of the 1982–83 El Niño.
- **A strong El Niño adds about another 4 inches on a typical winter day, and
  about a foot on the worst days.** Mike's foot is the right number for the bad
  days: in past strong El Niño winters 1 day in 25 carried a foot or more of
  extra water, and a quarter of days carried 6 inches or more.
- **The date to mark is 24 December 2026.** The season's highest astronomical
  tide — a perigean spring tide — lands that afternoon. On its own it reaches
  **1.7 ft above the average daily high tide**; with a typical El Niño winter
  underneath it, **2.0 ft**; with a 1997–98-scale winter, **2.7 ft**, higher than
  anything in the 47-year record.
- **We do not yet have the City's own wharf gauge.** NOAA has no gauge at Santa
  Cruz; the City's Hohonu sensor does, and getting its record is the single
  biggest improvement available to this analysis (§1).

---

## 1. Which gauge, and the City's sensor on the wharf

**NOAA has no tide gauge at the Santa Cruz Wharf.** Station 9413745 "Santa Cruz,
Monterey Bay" is a *subordinate* station: predictions only, derived from the
Monterey gauge (9413450, 25 km across the bay) with published offsets — high
waters ×0.97 and 6 minutes earlier, low waters ×0.99 and 11 minutes earlier. It
has no observations, no tidal datums of its own and no flood thresholds. So every
*observation* in this report is Monterey's, and Santa Cruz values are those
scaled by NOAA's offsets.

**David is right that the City of Santa Cruz has its own gauge on the wharf**, on
the [Hohonu network](https://dashboard.hohonu.io/map-page/hohonu-12/SantaCruzWharf).
We could not pull the bulk record: the dashboard reads from
`https://dashboard.hohonu.io/api/v1/stations/hohonu-12/statistic/`, which answers
`401 Unauthorized` without an API key. The endpoint and its `from` / `to` /
`datum` parameters are confirmed — a 401 rather than a 404 says the route is
right — so only the key is missing.

The analysis script now carries that path ready to run: `hohonu_fetch()` reads a
key from `$HOHONU_API_KEY` or `~/.hohonu_token`, caches the record alongside the
NOAA data, and `compare_wharf_to_monterey()` then does the thing worth doing with
it — **fit the wharf-to-Monterey height ratio and time lag from data and test
NOAA's 0.97 / −6 min offsets**, which have never been checked against a real
Santa Cruz record. If those offsets are off by even a few per cent, every Santa
Cruz number in this report moves with them.

**The ask:** a read key, or a CSV export from the sensor's installation to the
present, from David's contact at the City or from Hohonu directly.

The Monterey datums used throughout (1983–2001 epoch, metres above MLLW) are
MHHW 1.63 m, MSL 0.86 m, MLLW 0.00 m; NOAA's flooding levels there are minor
2.19 m, moderate 2.47 m, major 2.86 m.

---

## 2. The two lunar cycles: how big can the tide get on its own?

David and Mike have been looking at the nodal and perigean cycle peaks, so this
section takes them head on.

![Two panels: the highest predicted tide of each year at Santa Cruz 1990–2045 with a fitted nodal-plus-perigee curve, and the closest perigee-at-syzygy of each year](figs/fig_nodal_cycle.png)

**Figure 1.** *Top:* the highest predicted tide of each year at Santa Cruz, with
harmonics at the 18.613-year lunar nodal period and the 8.847-year perigee
precession fitted jointly. *Bottom:* the closest perigee falling within 24 hours
of a new or full moon each year — where the perigee cycle actually shows up.

| cycle | amplitude in the year's biggest tide | modelled peak |
|---|---|---|
| 18.613-yr lunar nodal | **±3.5 cm (±1.4 in)** | mid-2024 |
| 8.847-yr perigee precession | **±0.2 cm (±0.1 in)** | — |

The nodal cycle is real but small, and **we are one to two years past its peak**.
The perigee-precession cycle barely registers in the annual maximum, for the
reason the bottom panel makes plain: *every* year gets at least one good
perigee–syzygy alignment, and the closest one varies only between about 356.5 and
357.6 thousand km — a 0.3 % spread. Across 1990–2045 the year's highest predicted
tide ranges only from 1.94 to 2.10 m, **16 cm end to end**. That is the entire
astronomical story, and it is smaller than what El Niño and sea-level rise are
doing.

What the perigean cycle *does* control is timing, and this winter it is emphatic:

![Predicted daily high tides at Santa Cruz for the coming winter with the Moon–Earth distance overlaid and perigees marked](figs/fig_perigee_winter.png)

**Figure 2.** The coming winter's predicted high tides against the Moon–Earth
distance (plotted with closer moon upward). Red lines mark perigees falling
within 36 hours of a new or full moon — perigean spring tides.

**The 24 December 2026 tide is a textbook perigean spring tide.** Perigee falls
at 08:40 UTC that morning at **356,649 km — the closest approach of the season
and the second-closest alignment of the 2020s** — just **11 hours from syzygy**,
and the season's highest tide follows 8.7 hours later at 17:21. The 21 January
2027 alignment (357,288 km, 17 hours from syzygy) is the season's second.

---

## 3. The ocean underneath is high

![Monthly sea-level anomaly at Monterey 1973–2026 against NOAA's ONI](figs/fig_sealevel_oni.png)

**Figure 3.** Monthly mean sea level at Monterey, de-trended and referenced to
the 1991–2020 seasonal cycle, with NOAA's ONI. The 1982–83, 1997–98 and 2015–16
El Niños are the three tallest excursions in the record.

- Sea level in **August 2026 stood +8.9 cm (3.5 in)** above normal — the
  second-highest August of 53 years, behind August 1983 (+13.1 cm).
- The **past 12 months average +6.4 cm (2.5 in)**.
- Fitting the 53 complete winters against the ONI gives **+4.5 cm of sea level
  per °C of ONI, r = 0.80**. At the +2 °C threshold of the "very strong" event
  the CPC gives better than a 90 % chance of, that is about **+9 cm (3.5 in)**;
  the empirical composite of the five strong El Niño winters in the record (1983,
  1987, 1992, 1998, 2016) gives **+10 cm (4 in)**.
- The station's long-term rise is **+1.8 mm/yr**, which against the 1983–2001
  datum epoch the tide tables use is already **+6.4 cm (2.5 in)** of extra water
  in every high tide today.

---

## 4. The high tides are running above the tide tables

![Monthly mean of the observed-minus-predicted higher-high water at Monterey, last three years against the 1991–2020 climatology](figs/fig_residual.png)

**Figure 4.** The daily high tide at Monterey relative to its prediction, monthly
means for the last three years against the 1991–2020 10th–90th percentile band.

Over the last 12 months the daily high tide has run **+11.6 cm (4.6 in) above
prediction**, against a climatological **+1.8 ± 7.7 cm**, and **222 of 366 days**
were more than 10 cm high.

![Mean November–February residual at Monterey by winter, 1981–2026](figs/fig_winter_residual.png)

**Figure 5.** Winter (Nov–Feb) mean residual by winter, bars to the 90th
percentile; red marks the strong El Niño winters. **1997–98 leads at +17.6 cm**
(90th percentile +29 cm), then **1982–83 at +15.1 cm**. The winter just past,
2025–26, reached **+11.3 cm** — fourth of 47 winters, and reached *without* an
El Niño, on the marine heatwave alone. The all-winter mean is +2.0 cm.

---

## 5. How much extra water, in inches — and is it a foot?

Mike put the expectation as "a foot of extra water on every high tide and storm".
The record lets us be precise, so here is the distribution of the non-tidal
residual rather than a single number:

| | typical day | 6 in or more | a foot or more | worst day |
|---|---|---|---|---|
| All winters, 1981–2026 | +0.8 in | 9 % of days | 1 % (60 days in 47 winters) | +20.7 in |
| **Strong El Niño winters** | **+3.9 in** | **27 % of days** | **4 % (1 day in 25)** | **+20.7 in** |
| Last 12 months | +4.6 in | 26 % of days | 3 days | +14.9 in |

**The reading:** a foot is the right number for the *bad days* of a strong
El Niño winter — roughly one winter day in 25, which over a season is several
events, each landing on whatever tide happens to be running. It is not what every
high tide will carry: the typical elevated day is **4 to 6 inches**. And 2.5
inches of that is already permanent, being sea-level rise since the tide tables'
datum epoch.

Because NOAA's minor/moderate/major lines are not a planning standard everyone
uses — Mike's point, and a fair one — the winter scenarios in §7 are also given
as **feet above MHHW, the average daily high tide**, a reference anyone with a
site elevation can use directly.

---

## 6. How often the water actually reaches flood level

![Days per year above NOAA's minor high-tide-flooding level at Monterey](figs/fig_flood_days.png)

**Figure 6.** Days per calendar year above NOAA's minor high-tide-flooding level
at Monterey, from NOAA's derived-product API.

**2026 already holds the record with 10 days**, against 42 days in total over
1980–2026 — more than 1982 and 1983 combined (3 and 4 days) and more than the
previous worst year, 2005 (6 days). They came in three clusters:

| dates | highest high water (m, MLLW) | extra water |
|---|---|---|
| 1–4 January 2026 | 2.362 (3 Jan) | +7 to +10 in |
| 14–16 June 2026 | 2.286 (15 Jun) | +7 in |
| 13–15 July 2026 | 2.290 (14 Jul) | +5 to +7 in |

**3 January 2026 produced the third-highest high water of the 47-year record,
2.362 m**, behind only 27 and 28 January 1983 (2.401 m and 2.398 m, on residuals
of +36 and +33 cm) at the height of the great El Niño. The June and July clusters
are the telling ones: summer king tides do not normally reach flood level here,
and did this year only because the background ocean was already high.

*Two checks on the method:* the highest water level this analysis finds in 47
years — 2.401 m above MLLW on 27 January 1983 — is exactly the station record
NOAA publishes in its datums metadata; and counting days above the minor flooding
level in the downloaded high waters reproduces NOAA's own high-tide-flooding
counts year by year.

### A caution: the tide is not what broke the wharf

On **23 December 2024**, when a 150-foot section of the Santa Cruz Wharf
collapsed during a high-surf event, the day's higher-high water at Monterey was
**1.49 m — 0.70 m (2.3 ft) below** the minor flooding level, with only +5 cm of
extra water. The damage was done by long-period swell. Conversely the highest
tide of 2024 (14 December, above the minor flooding level) passed without
incident. Everything in this report is still-water level; wave setup and runup,
which are what damage structures at the wharf, are a separate problem and are not
analysed here.

---

## 7. The winter ahead

![Predicted high tides at Monterey for October 2026 – March 2027 with El Niño residual scenarios](figs/fig_winter_outlook.png)

**Figure 7.** Predicted daily high tides for the coming season (Monterey datum on
the left, Santa Cruz equivalent on the right), with two empirical scenarios: the
average strong-El Niño winter residual (+10 cm) and the 1997–98 analogue at its
90th-percentile day (+29 cm).

| scenario | peak water level | above MHHW | days above NOAA minor, Oct–Mar |
|---|---|---|---|
| Tide prediction alone | 2.14 m | **1.7 ft** | 0 |
| + typical strong El Niño winter (+4 in) | 2.24 m | **2.0 ft** | 3 |
| + 1997–98 analogue, worst days (+11.5 in) | 2.43 m | **2.7 ft** | 20 |
| *(for reference)* record high water, 27 Jan 1983 | 2.40 m | 2.5 ft | — |

Reading that table:

- The predicted tide by itself never reaches flood level this winter.
- A typical strong El Niño residual puts the **Christmas-week tides (23–24
  December) over NOAA's minor flooding level**, with other spring-tide runs
  within a couple of centimetres of it.
- A 1997–98-scale response would put water levels near **2.43 m on 20 days** —
  above the January 2026 high water, above the 1983 record, and within 4 cm of
  NOAA's *moderate* flooding level.

The ten highest predicted tides at Santa Cruz this season:

| date | predicted high water (m, MLLW) |
|---|---|
| 24 Dec 2026 | 2.08 |
| 23 Dec 2026 | 2.07 |
| 25 Nov 2026 | 2.03 |
| 25 Dec 2026 | 2.02 |
| 21 Jan 2027 | 2.02 |
| 22 Dec 2026 | 2.01 |
| 24 Nov 2026 | 2.00 |
| 26 Nov 2026 | 2.00 |
| 22 Jan 2027 | 2.00 |
| 20 Jan 2027 | 1.97 |

---

## 8. What to watch, and when

1. **October.** The coastal Kelvin wave propagating up from the tropics is
   expected at San Francisco in early-to-mid October. It should appear at
   Monterey as a step up in the monthly mean and in the daily residual. If the
   residual jumps from ~4½ inches to ~8 inches, the winter is tracking 1997–98
   rather than 2015–16.
2. **23–25 December.** The season's highest astronomical tides, the 24th the
   peak. Watch the residual in the days before.
3. **20–22 January 2027.** The second-highest run, and the calendar window in
   which both 1983 and 2026 set their record high waters.
4. **Monthly.** Whether the 12-month mean sea level, now +2.5 in, climbs toward
   the +5 to +5.5 in of the 1983 El Niño peak.
5. **Whenever the City's data arrive.** Re-run with the wharf record and replace
   NOAA's assumed 0.97 offset with a measured one.

---

## 9. Data and methods

* **Tide gauge:** NOAA CO-OPS [Monterey, station
  9413450](https://tidesandcurrents.noaa.gov/stationhome.html?id=9413450) —
  monthly means from 1973, verified high/low waters from 1980 (the high/low
  record begins in August 1979), all on MLLW, in local standard time.
* **Predictions:** CO-OPS harmonic predictions for [Santa Cruz, station
  9413745](https://tidesandcurrents.noaa.gov/noaatidepredictions.html?id=9413745)
  (subordinate) and for Monterey, 1990–2045.
* **City of Santa Cruz wharf gauge:** [Hohonu station
  hohonu-12](https://dashboard.hohonu.io/map-page/hohonu-12/SantaCruzWharf) —
  code path ready, awaiting an API key (§1).
* **Datums, offsets and flood levels:** CO-OPS metadata API; high-tide-flooding
  day counts from the derived-product API.
* **Lunar astronomy:** `astropy` built-in ephemeris — Moon–Earth distance and
  Moon–Sun elongation; perigees located as minima of the distance and refined
  parabolically.
* **ONI:** NOAA CPC, via `cugn.indices.load_oni`.
* **Sea-level anomalies:** `cugn.indices.sea_level_anomaly` — 12 monthly means
  and a linear trend fitted jointly, climatology 1991–2020.
* **Residuals:** each observed high water is matched to the *nearest predicted
  high water* within 3 hours, not to the highest prediction of the same calendar
  day. Calendar-day matching differences two different tide cycles when the
  higher-high falls near midnight, which manufactured apparent residuals of up to
  0.9 m in July and August; fixing it removed those without moving any headline
  number by more than 2 mm.
* **Script:** `reports/El_Nino_2026/scripts/santa_cruz_tides.py` (run in
  `ocean14`); every number quoted here is in
  `reports/El_Nino_2026/tides/tide_stats.json`.

**Caveats.** (i) Santa Cruz values are Monterey values scaled by NOAA's published
offsets; until the City's gauge is in hand, that scaling is untested. (ii)
Predictions and datums are on the 1983–2001 tidal epoch; when NOAA updates the
epoch, the datums and thresholds shift and the residuals quoted here drop by
roughly the sea-level rise between epochs. (iii) NOAA's flooding levels exist
only for Monterey, and are a reference here rather than a planning standard.
(iv) Everything is still water level: no wave setup, runup or overtopping.
(v) 2026 observations after mid-summer include preliminary data.

*Preliminary results, September 2026. Next update October 2026, after the Kelvin
wave arrives.*
