# Sources for the CUGN Line 66.7 El Niño 2026–27 report

**Purpose.** Running bibliography for the report on the 2026–27 El Niño as observed by the
California Underwater Glider Network (CUGN) on Line 66.7 off Monterey Bay. Entries are
appended each month as the event evolves; nothing is removed. Every entry gives the
BibTeX key used in the companion file `sources.bib`.

**Access date for all entries: 2026-09-03.**

Conventions: "DOI verified" means `curl -sI https://doi.org/<doi>` returned a redirect
(HTTP 302) on 2026-09-03 and Crossref/DataCite metadata was retrieved. "not re-fetched"
means the page returned HTTP 403 to automated fetchers on 2026-09-03; the details listed
come from the report notes, search-engine snippets, or an earlier read.

---

## Agency products and monitoring pages

1. **ENSO Diagnostic Discussion (El Niño Advisory)** — NOAA/NWS Climate Prediction Center, issued 13 Aug 2026.
   URL: https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso_advisory/ensodisc.shtml
   DOI: none. Fetched 2026-09-03 (text confirmed: July Niño-3.4 = +1.4 °C; ">90% chance of a very strong event during NH fall and winter 2026-27"; "69% chance of a historic event that would exceed the strength of previous El Niño events dating back to 1950" for OND 2026).
   Supports: ENSO status, Niño-3.4 value and strength probabilities quoted in the Introduction. Note: the page is overwritten monthly; the September discussion will replace this text.
   Key: `cpc2026aug`

2. **Official NOAA CPC ENSO Strength Probabilities** — NOAA Climate Prediction Center, continuously updated (fetched 2026-09-03).
   URL: https://cpc.ncep.noaa.gov/products/analysis_monitoring/enso/roni/strengths.php
   DOI: none.
   Supports: probabilistic strength categories (weak/moderate/strong/very strong) by season used in the ENSO-context figure.
   Key: `cpc2026strengths`

3. **IRI ENSO Forecast (model plume and technical discussion)** — International Research Institute for Climate and Society, Columbia University, issued 19 Aug 2026.
   URL: https://iri.columbia.edu/our-expertise/climate/forecasts/enso/current/
   DOI: none. Fetched 2026-09-03 (text confirmed: "25 of 26 models place Niño 3.4 in the very strong category" in OND 2026; almost all models >+2.0 °C from ASO onward).
   Supports: multi-model consensus on a very strong event peaking in OND 2026.
   Key: `iri2026aug`

4. **"El Niño forms, expected to strengthen, say NOAA forecasters"** — NOAA news release, 11 Jun 2026.
   URL: https://www.noaa.gov/news-release/el-nino-forms-expected-to-strengthen-say-noaa-forecasters
   DOI: none. HTTP 403 to fetchers; **not re-fetched**. Date confirmed from secondary coverage (weather.com, MPR News 11–12 Jun 2026).
   Supports: date on which the El Niño Advisory was first issued (onset of the event).
   Key: `noaa2026elninoforms`

5. **The California Current Marine Heatwave Tracker – Blobtracker** — NOAA CCIEA (California Current Integrated Ecosystem Assessment), update dated 19 Aug 2026.
   URL: https://www.integratedecosystemassessment.noaa.gov/regions/california-current/california-current-marine-heatwave-tracker-blobtracker
   DOI: none. Fetched 2026-09-03 (confirmed: NEP25A, onset May 2025, "located near the coast of central and southern California"; NEP26A, onset May 2026, far west of the region 37–47°N).
   Supports: naming and history of the coastal marine heatwave (NEP25A) preceding the El Niño.
   Key: `cciea2026blobtracker`

6. **"2025-26 California Current Ecosystem Status Report: Upwelling Fueled Productive West Coast Ocean, Holding Warm Waters Offshore in 2025"** — NOAA CCIEA news, 5 Mar 2026.
   URL: https://www.integratedecosystemassessment.noaa.gov/news/2025-26-california-current-ecosystem-status-report-upwelling-fueled-productive-west-coast
   DOI: none. Fetched 2026-09-03.
   Supports: ecosystem baseline for 2025 (strong upwelling kept warm water offshore) against which the 2026 coastal warming is contrasted.
   Key: `cciea2026esr`

7. **"Monitoring El Niño with Gliders"** — Scripps Institution of Oceanography, Instrument Development Group (SprayData), page updated 25 Jun 2026.
   URL: https://spraydata.ucsd.edu/products/el-nino/
   DOI: none. Fetched 2026-09-03.
   Supports: Southern California Temperature Index (SCTI) vs. ONI comparison; precedent for glider-based El Niño monitoring.
   Key: `spraydata2026elnino`

8. **CUGN Climatology (product page)** — Scripps IDG SprayData.
   URL: https://spraydata.ucsd.edu/products/cugn-climatology/
   DOI: 10.21238/S8SPRAY7292 (DOI verified; DataCite title "A climatology using data from the California Underwater Glider Network", Rudnick, Zaba, Todd & Davis, IDG/SIO, 2016; DataCite landing URL is this page).
   Supports: climatological mean/anomaly fields used to compute Line 66.7 anomalies.
   Key: `spraydata_cugnclim`

9. **California Underwater Glider Network (project page)** — Scripps IDG SprayData.
   URL: https://spraydata.ucsd.edu/projects/CUGN/ (redirects to https://spraydata.ucsd.edu/projects/cugn/)
   DOI: none.
   Supports: description of the network, lines, and sampling (surface to 500 m, repeat transects).
   Key: `spraydata_cugn`

10. **"NASA spots giant ocean swell that could signal El Niño's return"** — NASA Jet Propulsion Laboratory via ScienceDaily, 15 Jun 2026.
    URL: https://www.sciencedaily.com/releases/2026/06/260614012002.htm
    DOI: none. Fetched 2026-09-03 (quotes from Josh Willis and Severine Fournier, JPL; Sentinel-6 Michael Freilich altimetry of the equatorial Kelvin wave).
    Supports: satellite detection of the downwelling Kelvin wave that preceded coastal sea-level rise.
    Key: `nasa2026kelvin`

## Data sets

11. **CUGN Line 66.7 binned glider data (`binnedCUGN66`)** — Scripps IDG SprayData ERDDAP.
    URL: https://spraydata.ucsd.edu/erddap/tabledap/binnedCUGN66 (HTTP 200 on 2026-09-03)
    DOI: 10.21238/S8SPRAY7292 (CUGN data DOI; DOI verified).
    Supports: primary observational data set (T, S, velocity, etc. binned in depth and along-track distance) for all Line 66.7 analyses.
    Key: `cugn66_binned`

12. **CUGN data DOI record** — Rudnick, D. L., Zaba, K. D., Todd, R. E., Davis, R. E. (2016), Instrument Development Group, Scripps Institution of Oceanography.
    DOI: 10.21238/S8SPRAY7292 (DOI verified via DataCite).
    Supports: formal data citation for CUGN.
    Key: `rudnick2016cugndata`

13. **IOOS Glider DAC, mission sp025-20260611T1755 (real-time Level 2)** — U.S. IOOS National Glider Data Assembly Center ERDDAP.
    URL: https://gliders.ioos.us/erddap/tabledap/sp025-20260611T1755 (HTTP 200 on 2026-09-03)
    DOI: none.
    Supports: real-time profiles from Spray 025 deployed 11 Jun 2026 on Line 66.7, used for the most recent months before binned data are released.
    Key: `ioosdac_sp025`

14. **IOOS Glider DAC overview / documentation** — U.S. IOOS.
    URL: https://ioos.github.io/glider-dac/
    DOI: none.
    Supports: description of DAC processing levels and access.
    Key: `ioosdac_overview`

15. **Southern California Temperature Index, 10-day (`socal_index_10day_v1`)** — Scripps IDG SprayData ERDDAP.
    URL: https://spraydata.ucsd.edu/erddap/tabledap/socal_index_10day_v1 (HTTP 200 on 2026-09-03)
    DOI: none.
    Supports: SCTI time series compared with Line 66.7 anomalies and ONI.
    Key: `spraydata_scti`

16. **NOAA OISST v2.1 (Daily Optimum Interpolation SST)** — Huang, B., Liu, C., Banzon, V., Freeman, E., Graham, G., Hankins, B., Smith, T., Zhang, H.-M. (2021), "Improvements of the Daily Optimum Interpolation Sea Surface Temperature (DOISST) Version 2.1", J. Climate 34(8), 2923–2939.
    DOI: 10.1175/JCLI-D-20-0166.1 (DOI verified).
    Access: NCEI product page https://www.ncei.noaa.gov/products/optimum-interpolation-sst ; CoastWatch ERDDAP dataset `ncdcOisst21Agg_LonPM180` https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180
    Supports: SST anomaly maps and coastal SST time series; marine-heatwave categorization.
    Keys: `huang2021oisst` (paper), `oisst_ncei` (NCEI access), `oisst_coastwatch` (ERDDAP access)

17. **Oceanic Niño Index (ONI) table** — NOAA Climate Prediction Center.
    URL: https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt (HTTP 200)
    DOI: none.
    Supports: ONI values for 2026 and for the historical comparison events (1982–83, 1997–98, 2015–16).
    Key: `cpc_oni`

18. **Detrended Niño-3.4 index (relative/detrended)** — NOAA Climate Prediction Center.
    URL: https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/detrend.nino34.ascii.txt (HTTP 200)
    DOI: none.
    Supports: trend-independent ENSO amplitude comparison.
    Key: `cpc_nino34detrend`

19. **CUTI and BEUTI upwelling indices** — Jacox, M. G., Edwards, C. A., Hazen, E. L., Bograd, S. J. (2018), "Coastal Upwelling Revisited: Ekman, Bakun, and Improved Upwelling Indices for the U.S. West Coast", J. Geophys. Res. Oceans 123(10), 7332–7350.
    DOI: 10.1029/2018JC014187 (DOI verified). Data: https://mjacox.com/upwelling-indices/ (site returned HTTP 406 to curl on 2026-09-03; loads in a browser).
    Supports: upwelling forcing at 36–37°N during the 2026 warm season.
    Keys: `jacox2018cuti` (paper), `jacox_indices_web` (data page)

20. **NOAA CO-OPS tide gauges: Monterey (9413450) and San Francisco (9414290)** — NOAA Center for Operational Oceanographic Products and Services.
    URLs: https://tidesandcurrents.noaa.gov/stationhome.html?id=9413450 ; https://tidesandcurrents.noaa.gov/stationhome.html?id=9414290 ; portal https://tidesandcurrents.noaa.gov/
    DOI: none.
    Supports: coastal sea-level anomaly associated with the Kelvin wave and the El Niño steric signal.
    Keys: `coops_monterey`, `coops_sf`

## Peer-reviewed papers and preprints

21. Rudnick, D. L., Zaba, K. D., Todd, R. E., Davis, R. E. (2017). "A climatology of the California Current System from a network of underwater gliders." *Progress in Oceanography* 154, 64–106.
    DOI: 10.1016/j.pocean.2017.03.002 (DOI verified).
    Supports: CUGN sampling design and the climatology from which anomalies are computed; Line 66.7 mean structure.
    Key: `rudnick2017climatology`

22. Zaba, K. D., Rudnick, D. L. (2016). "The 2014–2015 warming anomaly in the Southern California Current System observed by underwater gliders." *Geophysical Research Letters* 43(3), 1241–1248.
    DOI: 10.1002/2015GL067550 (DOI verified).
    Supports: glider view of the 2014–15 warm anomaly (surface-intensified, upper 50 m) used as the comparison for 2025–26 warming.
    Key: `zaba2016warming`

23. Jacox, M. G., Hazen, E. L., Zaba, K. D., Rudnick, D. L., Edwards, C. A., Moore, A. M., Bograd, S. J. (2016). "Impacts of the 2015–2016 El Niño on the California Current System: Early assessment and comparison to past events." *Geophysical Research Letters* 43(13), 7072–7080.
    DOI: 10.1002/2016GL069716 (DOI verified).
    Supports: how the 2015–16 El Niño manifested in the CCS (including glider lines) versus 1997–98; template for this report's comparisons. (No separate Rudnick et al. 2017 GRL El Niño glider paper was found; the CUGN El Niño assessment in the literature is this paper plus Zaba & Rudnick 2016.)
    Key: `jacox2016elnino`

24. Bond, N. A., Cronin, M. F., Freeland, H., Mantua, N. (2015). "Causes and impacts of the 2014 warm anomaly in the NE Pacific." *Geophysical Research Letters* 42(9), 3414–3420.
    DOI: 10.1002/2015GL063306 (DOI verified).
    Supports: the original "Blob" marine heatwave; precedent for the NEP25A/NEP26A naming.
    Key: `bond2015blob`

25. Hobday, A. J., et al. (2016). "A hierarchical approach to defining marine heatwaves." *Progress in Oceanography* 141, 227–238.
    DOI: 10.1016/j.pocean.2015.12.014 (DOI verified).
    Supports: marine heatwave definition (90th percentile, ≥5 days) applied to Line 66.7 and OISST.
    Key: `hobday2016mhw`

26. Hobday, A. J., Oliver, E. C. J., Sen Gupta, A., Benthuysen, J. A., Burrows, M. T., Donat, M. G., Holbrook, N. J., Moore, P. J., Thomsen, M. S., Wernberg, T., Smale, D. A. (2018). "Categorizing and naming marine heatwaves." *Oceanography* 31(2).
    DOI: 10.5670/oceanog.2018.205 (DOI verified).
    Supports: MHW category scheme (Moderate/Strong/Severe/Extreme).
    Key: `hobday2018naming`

27. Lian, T., Hu, R., Feng, J., Liu, T., Tang, Y., Chen, D. (2026). "Extreme Spring Pacific Annular Warming Elevates the 2026/27 El Niño." *Ocean-Land-Atmosphere Research* 5, 0153. Published online 30 Apr 2026.
    DOI: 10.34133/olar.0153 (DOI verified; authors/date from Crossref).
    Supports: dynamical explanation for the rapid spring 2026 onset and its forecast amplitude.
    Key: `lian2026olar`

28. Ludescher, J., Meng, J., Fan, J., Bunde, A., Schellnhuber, H. J. (2026). "Climate network and complexity based ENSO forecast for 2026." arXiv:2602.14773, submitted 16 Feb 2026.
    DOI: 10.48550/arXiv.2602.14773 (DOI verified). URL: https://arxiv.org/abs/2602.14773
    Supports: an early-2026 statistical forecast that favored ENSO-neutral (or a mild ~0.8 °C event) — cited as an example of the forecast spread before the spring onset.
    Key: `ludescher2026arxiv`

29. Beniche, M., Vialard, J., Taschetto, A. S., Lengaigne, M. (2026). "Reduced Distinctiveness of Extreme El Niño Teleconnections in Warmer Climates." *Geophysical Research Letters* 53(12), e2025GL121189. Published 12 Jun 2026.
    DOI: 10.1029/2025GL121189 (DOI verified).
    Supports: expected teleconnection behavior of extreme El Niño in a warmer climate (context for California impacts).
    Key: `beniche2026grl`

30. Wang, Y., Widlansky, M. J., Stuecker, M. F., Zhao, S., Jin, F.-F. (2026). "ENSO Predictability From Combined Wyrtki and Hasselmann Memory in a Cyclostationary Linear Inverse Model." *Geophysical Research Letters* 53(8), e2025GL119694. Published 14 Apr 2026.
    DOI: 10.1029/2025GL119694 (DOI verified). This is the UH Mānoa "15-month ENSO prediction from SST and sea-surface height" study (UH News, 30 Apr 2026: https://www.hawaii.edu/news/2026/04/30/el-nino-15-months/).
    Supports: long-lead prediction of the 2026 event from upper-ocean heat content (Wyrtki memory) and extratropical SST persistence.
    Key: `wang2026wyrtki`

31. Xu, T., Newman, M., Shin, S.-I., Capotondi, A., Vimont, D. J., Alexander, M. A., Di Lorenzo, E. (2026). "Persistent Northeast Pacific marine heatwaves are sensitive to the seasonality of tropical and North Pacific dynamics." *Communications Earth & Environment* 7, 528. Published 15 Apr 2026.
    DOI: 10.1038/s43247-026-03442-x (DOI verified).
    Supports: mechanism linking persistent NE Pacific MHWs (e.g., NEP25A) to ENSO seasonality; prolonged MHWs begin months before the ENSO peak.
    Key: `xu2026nepmhw`

32. Cai, C., Thompson, L., Maroon, E. A., Deppenmeier, A.-L., Cohen, J. T., Staneva, V. (2026). "Drivers of Marine Heat Waves in the North Pacific Ocean." *Journal of Climate* 39(8), 1799–1820. Published 15 Apr 2026.
    DOI: 10.1175/JCLI-D-25-0308.1 (DOI verified).
    Supports: mixed-layer heat-budget drivers of North Pacific MHWs, used when interpreting the Line 66.7 upper-ocean heat content.
    Key: `cai2026drivers`

    *Search note (2026-09-03):* no peer-reviewed paper or indexed preprint specifically describing the 2025–26 NEP25A heatwave or the California coastal response to the 2026 El Niño was found yet; items 31–32 are the closest 2026 process papers. Re-check ESS Open Archive / EarthArXiv / GRL monthly.

## News and outreach

33. Times of San Diego (Thomas Murphy), 25 Aug 2026. "'Super' El Niño and San Diego – what scientists know now and what else they hope to learn."
    URL: https://timesofsandiego.com/environment/2026/08/25/super-el-nino-san-diego-science/
    Note: 42 record-SST days at Scripps Pier in 2026; Adam Young (SIO) on El Niño sea-level rise.
    Key: `timesofsandiego2026super`

34. KQED Science (Ezra David Romero), 26 May 2026. "Scientists Worry El Niño Could Supercharge Marine Heat Wave Roiling Coastal California."
    URL: https://www.kqed.org/science/2001047/scientists-worry-el-nino-could-supercharge-marine-heat-wave-roiling-coastal-california
    Note: Monterey Bay Aquarium (Anita Giraldo Ospina) on kelp; coastal waters 3–4 °F above normal; seventh MHW in seven years.
    Key: `kqed2026mhw`

35. San Francisco Chronicle (Anthony Edwards), 2 Sep 2026. "El Niño-driven wave set to raise California sea levels by a foot."
    URL: https://www.sfchronicle.com/weather/article/el-nino-california-sea-level-22413150.php
    Note: coastal Kelvin wave propagating north at ~6 mph; ~1 ft sea-level rise expected by late September.
    Key: `sfchronicle2026kelvin`

36. Gizmodo (Ellyn Lapointe), 7 Aug 2026. "This Super El Niño Could Trigger the Highest Sea Levels Ever Recorded on the California Coast."
    URL: https://gizmodo.com/this-super-el-nino-could-trigger-the-highest-sea-levels-ever-recorded-on-the-california-coast-2000795970
    Note: Mark Merrifield (SIO) — "likely the highest sea levels ever recorded on the California coast"; eastern Pacific already >15 cm above average.
    Key: `gizmodo2026sealevel`

37. The Spokesman-Review, 13 Aug 2026 (Los Angeles Times syndication; Rong-Gong Lin II). "A 'super' El Niño is now super likely, leaving California bracing for impact."
    URL: https://www.spokesman.com/stories/2026/aug/13/a-super-el-nino-is-now-super-likely-leaving-califo/
    Note: coverage of the 13 Aug CPC update; Michelle L'Heureux quote on unprecedented strength since 1950.
    Key: `spokesman2026super`

38. **CBS8 San Diego (Earth 8), 5 May 2026** (date from the station's YouTube upload of the segment; article page returns HTTP 403, **not re-fetched**). "Autonomous ocean gliders reveal unusual marine heat wave."
    URL: https://www.cbs8.com/article/news/local/outreach/earth8/autonomous-ocean-gliders-show-unusual-marine-heat-wave/509-2cb64042-0a07-4640-9294-bb1637dc8925 (video: https://www.youtube.com/watch?v=kqv80UD1PUg)
    Note: Dan Rudnick — temperatures on the Pt. Conception and Monterey glider lines the warmest in 20 years.
    Key: `cbs8_gliders_mhw`

39. Scripps Institution of Oceanography news explainer, c. Aug 2026 (date not confirmed; HTTP 403, **not re-fetched**; snippet cites the Aug 2026 NOAA advisory). "What Is El Niño and What Makes This a Very Strong El Niño?"
    URL: https://scripps.ucsd.edu/news/what-el-nino-and-what-makes-very-strong-el-nino
    Note: definition (0.5 °C threshold), NOAA >90 % very-strong probability.
    Key: `scripps2026whatis`

40. Scripps Institution of Oceanography news explainer, c. Aug 2026 (earliest Wayback capture 24 Aug 2026; HTTP 403, **not re-fetched**). "How Does El Niño Impact the California Coast?"
    URL: https://scripps.ucsd.edu/news/how-does-el-nino-impact-california-coast
    Note: winds push warm offshore water coastward in fall/winter; up to 13 in. temporary sea-level rise (Adam Young); king tides near Christmas.
    Key: `scripps2026coast`

41. Euronews, 31 Mar 2026. "A 'super El Niño': Inside the weather phenomenon that could send temperatures soaring."
    URL: https://www.euronews.com/2026/03/31/a-super-el-nino-inside-the-weather-phenomenon-that-could-send-temperatures-soaring (HTTP 406 to curl; loads in browser)
    Note: early-spring explainer; "super" is informal, strength defined by ONI.
    Key: `euronews2026super`

42. Newsweek, 30 Jul 2026. "Super El Niño Threat Grows: Map Shows At-Risk States This Winter."
    URL: https://www.newsweek.com/super-el-nino-2026-forecast-prediction-experts-12241456
    Note: national impact map; CPC probabilities as of July.
    Key: `newsweek2026map`

43. Newsweek, 21 Aug 2026. "New Super El Niño Maps Show Potential Rainfall, Temperatures Across US."
    URL: https://www.newsweek.com/new-super-el-nino-maps-show-potential-rainfall-temperatures-across-us-12353463
    Note: winter 2026–27 precipitation/temperature outlook maps (wet Southwest/California).
    Key: `newsweek2026maps_aug`

44. CBS News Bay Area / KPIX (Zoe Mintz), 30 Aug 2026 (republished by KION Central Coast). "Marine heat wave impacting California coastal water temperatures."
    URLs: https://www.cbsnews.com/sanfrancisco/news/marine-heat-wave-california-coastal-water-temperatures/ ; https://kioncentralcoast.com/news/top-stories/2026/08/30/marine-heat-wave-impacting-california-coastal-water-temperatures/
    Note: Monterey Bay National Marine Sanctuary under MHW conditions 194 days in 2026 to date (130 days in all of 2025); water 5–10 °F above normal.
    Key: `cbssf2026mhw`

45. University of Hawaiʻi News, 30 Apr 2026. "Simple ocean model predicts El Niño 15 months in advance."
    URL: https://www.hawaii.edu/news/2026/04/30/el-nino-15-months/
    Note: press release for Wang et al. (2026); Wyrtki-CSLIM predicted a strong 2026 El Niño (>2 °C).
    Key: `uhnews2026wyrtki`
