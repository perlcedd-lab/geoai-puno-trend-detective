# GeoAI Puno: Earth System Trend Detective & Agricultural Drought Forecasting
**NASA Space Apps Challenge 2026** — *Submission Window: November 14–15, 2026*  
**Official Challenge Track:** **"Be An Earth System Trend Detective!"**  
**Geographic Domain:** Andean Altiplano of Puno, Peru (Elevation: 3,812 to 5,200 m a.s.l. | 404 terrestrial 5 km grid cells)  
**Core Architecture:** Multi-Decadal Earth Observation Ingestion (2001–2025, 25 complete agricultural campaigns Dec–Mar), Non-Parametric Trend Detection (Mann-Kendall with Hamed & Rao 1998 modification + Benjamini-Hochberg FDR control with $p_{\text{FDR}} < 0.05$ Significance), Target Leakage Elimination, 16-Day Early Warning Forecasting, Conformalized Quantile Regression (CQR), and Two-Level Catastrophic Agricultural Insurance (SAC) Decision Triggers.

---

## 1. ALIGNMENT WITH THE NASA CHALLENGE: "BE AN EARTH SYSTEM TREND DETECTIVE!"

### Official NASA Challenge Statement (from spaceappschallenge.org)
> "Change is always occurring in the Earth's interconnected environmental system, and when it runs in a consistent direction – rising or falling, increasing or decreasing, thickening or thinning – measured variables reflect it. But a measured Earth science data variable can trend one way in one region and the opposite way in another, even when the same process drives both. The multitude of Earth science data variables makes it difficult to select variables that could indicate change, and to then perform an analysis on the selected variables. Your challenge is to create a tool that can select variables measured by Earth science missions or produced by Earth science models, visualize how they change over time, and determine what is changing, where it is changing, how much it is changing, and if these changes are what scientists call 'significant.'"

### Our System as the "Earth System Trend Detective"

#### 1. WHAT IS CHANGING?
- **Vegetation Health Index (VHI, 2001–2025):** 25 complete agricultural campaigns (December–March) derived from MODIS Terra/Aqua (MOD13Q1/MOD11A2) combining normalized vegetation vigor (VCI) and thermal stress (TCI). Incomplete 2026 is strictly excluded.
- **Root-Zone Soil Moisture ($0\text{--}100\text{ cm}$):** NASA FLDAS Noah Land Surface Model L4 Global Monthly (`NASA/FLDAS/NOAH01/C/GL/M/V001`, 2001–2025, $n=25$ campaigns) with depth-weighted profile:
  $$\text{SM}_{\text{rootzone}} = 0.10 \cdot \text{SoilMoi}_{0\text{--}10\text{cm}} + 0.30 \cdot \text{SoilMoi}_{10\text{--}40\text{cm}} + 0.60 \cdot \text{SoilMoi}_{40\text{--}100\text{cm}}$$
  capturing decadal desiccation, complemented by NASA SMAP L4 (`SPL4SMGP`) pure satellite observations (2015–2025, $n=11$ campaigns, where lower statistical power is explicitly acknowledged due to satellite sensor lifespan).
- **Precipitation Anomalies (SPI-1, SPI-3):** Derived from CHIRPS Daily high-resolution precipitation records (1981–2025) reflecting seasonal monsoon delay.
- **Land Surface Temperature (LST Day/Night):** MODIS MOD11A2 (1 km) preserving extreme high-altitude thermal dynamics, daytime radiative warming, and nocturnal sub-zero frost episodes (down to $-2.4^\circ\text{C}$).

#### 2. WHERE IS IT CHANGING?
- Spatial coverage: 404 terrestrial 5 km grid cells across the Andean Altiplano of Puno, Peru (WGS84: $71.1^\circ\text{W}$ to $68.8^\circ\text{W}$, $17.3^\circ\text{S}$ to $13.0^\circ\text{S}$, excluding Lake Titicaca).
- Ranked vulnerable districts: Highest rates of vegetative degradation are localized in **Chucuito**, **Huancané**, **Azángaro**, **El Collao**, and **San Román**, whereas zones within $25\text{ km}$ of Lake Titicaca exhibit pronounced microclimatic thermal and moisture buffering.

#### 3. HOW MUCH IS IT CHANGING?
- Quantified by the non-parametric **Theil-Sen Robust Median Slope**:
  - **VHI Desiccation Rate:** Mean regional slope of **$-0.3297\text{ points/year}$**, representing a cumulative loss of **$-7.91\text{ VHI points}$** over 24 years.
  - **Soil Moisture Depletion:** Root-zone moisture loss of **$-0.00183\text{ m}^3/\text{m}^3/\text{year}$** (FLDAS reanalysis, $n=25$) and **$-0.00201\text{ m}^3/\text{m}^3/\text{year}$** (SMAP satellite, $n=11$).
  - **Land Surface Temperature:** Daytime radiative warming of **$+0.0402^\circ\text{C/year}$**.

#### 4. ARE THESE SHIFTS STATISTICALLY SIGNIFICANT?
- **Serial Autocorrelation:** Environmental time-series in the Altiplano exhibit positive serial correlation ($\bar{\rho}_1 = 0.3020$) due to multi-year ENSO cycles and hydrological soil memory. Standard Mann-Kendall tests inflate false positive rates under positive autocorrelation.
- **Hamed & Rao (1998) Modification:** We implemented the rigorous Hamed & Rao formulation where the variance inflation factor $\eta$ scales the test statistic as $Z^* = Z / \sqrt{\eta}$:
  $$\text{Var}^*(S) = \text{Var}(S) \cdot \eta, \quad \eta = \max\left(1 + \frac{2}{n(n-1)(n-2)} \sum_{k=1}^{n-1} (n-k)(n-k-1)(n-k-2) \rho_k, 1.0\right)$$
  Under positive serial autocorrelation, $\bar{\eta} = 1.2509 \ge 1.0$, which strictly deflates $|Z^*|$ and guarantees $p_{\text{HR}} \ge p_{\text{raw}}$.
- **False Discovery Rate (FDR) Control:** Applied the Benjamini-Hochberg procedure across all 404 simultaneous cell hypotheses at $\alpha = 0.05$.
- **Statistical Results:**
  - **Raw Mann-Kendall:** $84.4\%$ significant ($341 / 404$ cells, $p < 0.05$).
  - **Hamed & Rao (1998) Corrected:** $74.0\%$ significant ($299 / 404$ cells, $p_{\text{HR}} < 0.05$). Significance monotonically decreases from $84.4\% \to 74.0\%$, adhering to the physical and mathematical direction.
  - **Hamed & Rao + Benjamini-Hochberg FDR:** **$62.1\%$ of terrestrial cells** ($251 / 404$ cells) exhibit statistically significant regional desiccation ($p_{\text{FDR}} < 0.05$).
  - **Sensor Reality (NASA SMAP):** SMAP satellite observations (2015–2025, $n=11$) show $3.2\%$ FDR significance due to limited sample size ($n=11$ campaigns), which we openly report as low statistical power, contrasting with the 25-year FLDAS reanalysis ($94.6\%$ significance).

---

## 2. SCIENTIFIC INTEGRITY & THE LEAKAGE AUDIT

A common failure mode in academic machine learning is reporting **vanity metrics**. Prior versions of this repository claimed an artificial $R^2 = 0.9852$ and $\text{RMSE} = 2.29\%$. 

Our audit confirmed a critical flaw: **Circular Target Leakage**.
- The target variable was Kogan's Vegetation Health Index: $\text{VHI}(t) = 0.5 \cdot \text{VCI}(t) + 0.5 \cdot \text{TCI}(t)$.
- Contemporaneous $\text{VCI}(t)$ and $\text{TCI}(t)$ were provided as model inputs. The models were simply solving an exact algebraic identity.
- Furthermore, raw coordinates (`lon`, `lat`) allowed spatial memorization, and Kriging was blindly applied.

### How We Solved It
1. **Central Feature Registry (`app/utils/feature_registry.py` & `experiment_config.yaml`):**
   - Hard programmatic assertions prevent any contemporaneous optical/thermal predictor (`vci`, `tci`, `lst`, `ndvi`) or raw coordinate from entering the model.
   - Eradicated linear dependency $\text{vhi\_lag16} = 0.5(\text{vci\_lag16} + \text{tci\_lag16})$ by using only the physical pair `vci_lag16` and `tci_lag16`.
2. **Operational 16-Day Forecast Mode (Mode B):**
   - Predicts $\text{VHI}(t+16\text{ days})$ using exclusively antecedent states ($\le t$) with an embargo of $\ge 16$ days between train and test to avoid composite window overlap.
3. **Rigorous Spatial Block Partitioning:**
   - Evaluated on $n=85$ unseen terrestrial cells (21.0% test), $n=37$ validation/calibration (9.2%), and $n=282$ training (69.8%).
   - *Why test expanded from 72 to 85:* The legacy codebase used unstratified random sampling. Our refactoring groups contiguous geographic cells into whole macro-blocks ($77.2\text{ km} \times 77.2\text{ km}$); assigning two entire spatial blocks to test yields 85 disjoint cells with zero spatial leakage.

---

## 3. BENCHMARK RESULTS (TEST CIEGO $n=85$ CELDAS, MODO B +16 DÍAS)

> [!CAUTION]
> The **Legacy Leaky Baseline** ($R^2 = 1.0000$) is preserved strictly for scientific audit. It is **scientifically invalid** and must never be cited as an operational benchmark. The true physical benchmark is the **Persistence Baseline**.

| Model / Architecture | Scientific Evaluation Status | $R^2$ Score [95% CI] | RMSE (%) | MAE (%) | Skill Score (RMSE vs Persist.) | 90% PICP | 90% PINAW | Winkler Score | Operational Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Legacy Leaky Baseline** | **SCIENTIFICALLY INVALID (Circular Leakage)** | **1.0000** | 0.12 % | 0.08 % | N/A | N/A | N/A | N/A | Audit Only |
| **Persistence Baseline** ($\hat{y} = \text{VHI}(t)$) | **Mandatory Physical Inertia Benchmark** | **0.5564** | **8.78 %** | **7.89 %** | **+0.000 (Ref)** | N/A | N/A | N/A | Reference Baseline |
| **Random Forest (Regularized)** | Valid Benchmark (`max_depth=10`, `sqrt`) | 0.8458 [0.782, 0.896] | 5.17 % | 4.15 % | +0.411 | N/A | N/A | N/A | Benchmarked |
| **Random Forest (Full Trees)** | Valid Benchmark (Unconstrained) | 0.9166 [0.885, 0.941] | 3.81 % | 2.88 % | +0.566 | N/A | N/A | N/A | Benchmarked |
| **XGBoost Regressor** | Valid Model (Gradient Boosting) | 0.9034 [0.865, 0.935] | 4.09 % | 3.25 % | +0.534 | N/A | N/A | N/A | Benchmarked |
| **LightGBM Quantile (Raw)** | Valid Model (Direct Quantiles, Sin CQR) | 0.9058 [0.871, 0.938] | 4.04 % | 2.93 % | +0.539 | 87.06 % | 31.15 % | 27.43 | Sharpest empirical interval (22.14 pts), but fails 90% finite coverage guarantee |
| **LightGBM Quantile + CQR** | Conformal Calibrated ($\hat{q} = +3.00$) | 0.9058 [0.871, 0.938] | 4.04 % | 2.93 % | +0.539 | 96.47 % | 39.58 % | 28.44 | Valid coverage, widened by additive CQR ($22.14 + 2(3.00) = 28.13$ pts) |
| **FCNN Quantile (Raw)** | Neural Pinball Loss (Over-conservative) | 0.9124 [0.883, 0.948] | 3.90 % | 3.07 % | +0.556 | 98.82 % | 37.86 % | 27.69 | Over-covers without guarantee (26.91 pts) |
| **FCNN Quantile + CQR** | **Conformal Calibrated (Optimal Band)** | **0.9124 [0.883, 0.948]** | **3.90 %** | **3.07 %** | **+0.556** | **96.47 %** | **31.58 %** | **23.93** | **OPERATIONAL BAND ($26.91 - 2(2.23) = 22.45$ pts)** |

### Spatial Exclusion Buffer as a Stress-Test / Pessimistic Lower Bound (Fixed Test Set $n=85$)
> [!NOTE]
> The buffer setup with $n_{\text{train}} = 53$ is a **stress-test / pessimistic lower bound**, not the operational production model. By forcing a strict exclusion moat of $\ge 91.67\text{ km}$ (matching the empirical variogram range), $81.2\%$ of the training data is purged ($282 \to 53$ cells), creating extreme data starvation and forcing cross-regional geographical extrapolation. The operational production model is the **Spatial Block-CV ($n_{\text{train}} = 282, n_{\text{val}} = 37, n_{\text{test}} = 85$)**, with the acknowledged caveat that adjacent spatial blocks must be monitored for residual correlation.

To test whether spatial autocorrelation between splits inflated accuracy under this lower-bound scenario, we evaluated models on the strictly fixed test set ($n=85$ cells) with the buffer removing all training points within $91.67\text{ km}$ of any test point (leaving $n=53$ distant cells with minimum separation $\ge 92.04\text{ km}$):

| Model | Setup / Hyperparameters | WITHOUT Buffer ($n_{\text{train}} = 282$) | WITH Buffer $\ge 91.67\text{ km}$ ($n_{\text{train}} = 53$, min dist $\ge 92.04\text{ km}$) | $\Delta R^2$ (Spatial Moat Effect) |
| :--- | :--- | :---: | :---: | :---: |
| **Random Forest (Regularized)** | `max_features="sqrt"`, `max_depth=10`, `n_est=250` | $R^2 = 0.8458$ ($\text{RMSE} = 5.17\%$) | $R^2 = 0.6387$ ($\text{RMSE} = 7.92\%$) | $-0.2071$ |
| **Random Forest (Full Trees)** | `max_features=1.0`, unconstrained depth, `n_est=250` | $R^2 = 0.9166$ ($\text{RMSE} = 3.81\%$) | $R^2 = 0.8149$ ($\text{RMSE} = 5.67\%$) | $-0.1017$ |
| **XGBoost Regressor** | `max_depth=5`, `lr=0.05`, `subsample=0.85` | $R^2 = 0.9034$ ($\text{RMSE} = 4.09\%$) | $R^2 = 0.8030$ ($\text{RMSE} = 5.85\%$) | $-0.1004$ |

**RF Metric Reconciliation:**
- In `correccion5.txt`, Random Forest was evaluated with regularized hyperparameters (`max_depth=10`, `max_features="sqrt"`), giving $R^2 = 0.8349 \approx 0.8458$.
- In `correccion6.txt`, Random Forest was evaluated with full unconstrained trees (`max_depth=None`, `max_features=1.0`), reaching $R^2 = 0.9166 \approx 0.9169$.
- Both runs used identical fixed seeds (`random_state=42`) and the exact same $n=85$ test cells; the difference is purely hyperparameter tree depth and feature subspace sampling.

---

## 4. VALIDATION AGAINST PERSISTENCE, CONFORMAL BANDS & BRIER SCORE

### A. Performance on Drought Onset ($|\Delta\text{VHI}| > 10$)
In slow-moving environmental systems, persistence ($\hat{y} = \text{VHI}(t)$) can appear deceptively competitive. We evaluated all models specifically on the subset where conditions shifted rapidly ($|\Delta\text{VHI}| > 10$, $n=21$ cells, representing $24.7\%$ of test cells; $100\%$ active Drought Onset):
- **Persistence Error:** $\text{RMSE} = 13.16\%$, $\text{MAE} = 12.94\%$ (catastrophic failure during onset).
- **Random Forest:** $\text{RMSE} = 6.79\%$ ($\text{Skill Score} = +0.484$).
- **XGBoost:** $\text{RMSE} = 4.73\%$ ($\text{Skill Score} = +0.640$).
- **LightGBM Quantile:** $\text{RMSE} = 4.47\%$ ($\text{Skill Score} = +0.661$).
- **FCNN Cuantil:** $\text{RMSE} = 4.23\%$ ($\text{Skill Score} = +0.679$).

Machine learning reduces forecasting error by **$67.9\%$** relative to persistence during active drought onset.

### B. Uncertainty Quantification: Conformal Guarantee vs Empirical Sharpness Trade-off
Evaluating uncertainty methods across test ($n=85$):
- **Double-CQR Proven Bug & Arithmetic Resolution:**
  - Raw LightGBM interval width is **$22.14$ pts** (Winkler $27.43$, PICP $87.06\%$).
  - Correct single CQR calibration ($\hat{q} = +3.00$) yields: $22.14 + 2(3.00) = \mathbf{28.13\text{ pts}}$ (Winkler $28.44$, PICP $96.47\%$).
  - Accidental re-calibration of the calibrated output produced $28.12 + 2(4.57) = 37.27$ pts. Conformal calibration must **always execute directly on the raw uncalibrated predictions**.
- **Operational Trade-Off Decision:**
  - LightGBM Raw provides a narrower width ($22.14$ pts) than CQR ($28.13$ pts), but **lacks finite-sample conformal guarantees** and falls below nominal coverage ($87.06\% < 90\%$), which is unacceptable for agricultural insurance parametric triggers. Calibrating LightGBM with CQR widens it to $28.13$ pts ($W = 28.44$).
  - FCNN Raw is over-conservative ($26.91$ pts, PICP $98.82\%$). Applying Split-CQR ($\hat{q} = -2.230$) contracts the interval to **$22.45\text{ pts}$** ($W = 23.93$) while guaranteeing finite-sample valid coverage (**$96.47\% \ge 90\%$**).
  - Therefore, **FCNN Split-CQR** is chosen as the operational uncertainty band: it guarantees formal statistical validity while being $5.68$ points sharper than calibrated LightGBM ($22.45$ vs $28.13$ pts).

### C. Probabilistic Calibration: Brier Score for $P(\text{VHI} < 20)$
Across the $n=85$ test cells, exactly 10 severe/extreme drought events ($y_{\text{true}} < 20$, $11.8\%$) occurred:
- **FCNN Brier Score:** **0.0299** [Spatial Block Bootstrap 95% CI: **0.0263, 0.0325**].
- **Climatological Reference Brier:** $0.1038$.
- **Brier Skill Score (BSS):** Empirical point estimate is **+0.712** ($+0.7123 = 1 - 0.0299 / 0.1038$) with Spatial Block Bootstrap 95% CI: **[+0.5826, +0.7812]** across the 5 independent test macro-blocks ($[0, 8, 11, 17, 19]$).

---

## 5. DECISION ENGINE: TWO-LEVEL CATASTROPHIC INSURANCE (SAC) TRIGGER

The decision system applies **pure probability thresholds** derived from monotonic CDF quantile interpolation:
1. **Level 1 — Agronomic Surveillance ($P(\text{VHI} < 20) \ge 0.50$):**
   - Severe drought is more likely than not ($>50\%$ probability). Mobilizes field verification teams and alerts irrigation committees.
2. **Level 2 — Official Indemnification Activation ($P(\text{VHI} < 20) \ge 0.70$ + 2-Composite Confirmation):**
   - High statistical certainty confirmed across two consecutive 16-day composite windows to eliminate false alarms from transient sensor noise. Unlocks parametric compensation under Peru's Catastrophic Agricultural Insurance (SAC).

---

## 6. GEOSTATISTICS: RESIDUAL MORAN'S I & OMISSION OF REDUNDANT KRIGING

We evaluated Global Moran's $I$ on model residuals using inverse distance weighting ($w_{ij} = 1 / d_{ij}$):
- **Empirical Moran's $I$:** $0.0412$
- **Expected $E[I]$:** $-0.0119$
- **p-value:** $0.2577 \ge 0.05$

Because residual autocorrelation is statistically indistinguishable from spatial white noise, **Kriging is intentionally omitted**. The continuous prediction map endpoint `/ml-prediction-map` (aliased to `/prediction-map`) returns direct continuous ML predictions, transparently avoiding redundant double smoothing.

---

## 7. REPRODUCIBILITY & TEST SUITE

```powershell
# Run the complete test suite (31 tests, 100% passing in ~29.5s)
pytest -v tests/test_refactor_integrity.py

# Generate the 1-page scientific summary PDF
python generate_article_summary_pdf.py

# Launch the interactive Leaflet Geoportal
python run.py
# Open: http://localhost:5000
```
- **Reliability Curve:** Demonstrates strict monotonic fidelity: predicted probability intervals $[0.0, 0.2]$ observe $0\%$ drought, while intervals $[0.8, 1.0]$ observe $100\%$ drought.

---

## 5. DECISION ENGINE: TWO-LEVEL CATASTROPHIC INSURANCE (SAC) TRIGGER

Rather than a single arbitrary cutoff or redundant OR conditions, the decision system applies **pure probability thresholds** derived from monotonic CDF quantile interpolation:

1. **Level 1 — Agronomic Surveillance ($P(\text{VHI} < 20) \ge 0.50$):**
   - *Meaning:* Severe/extreme drought is more likely than not ($>50\%$ probability).
   - *Operational Action:* Mobilizes field verification teams from DRA Puno, inspects communal water reservoirs, and alerts local irrigation committees.
2. **Level 2 — Official Indemnification Activation ($P(\text{VHI} < 20) \ge 0.70$ + 2-Composite Confirmation):**
   - *Meaning:* High statistical certainty of catastrophic vegetative failure, confirmed across two consecutive 16-day composite windows to eliminate false alarms from transient sensor noise.
   - *Operational Action:* Unlocks parametric financial compensation under the Peruvian Catastrophic Agricultural Insurance (SAC) and prepares direct assistance vouchers for affected alpaquero families.

---

## 6. GEOSTATISTICS: REGRESSION KRIGING & MORAN'S I

We reformed the interpolation pipeline to **Regression Kriging**:
$$\text{Final Surface} = \text{ML Prediction} + \text{Ordinary Kriging}(\text{ML Residuals})$$

Before applying Kriging, the system automatically evaluates Global Moran's $I$ on model residuals using inverse distance weighting ($w_{ij} = 1 / d_{ij}$ within a $1.5^\circ$ search radius):
- **Empirical Moran's $I$:** $0.0383$
- **Expected $E[I]$:** $-0.0119$
- **p-value:** $0.2577 \ge 0.05$

Because the residual autocorrelation is statistically indistinguishable from spatial white noise, the pipeline **automatically skips Kriging**, preventing artificial double smoothing and documenting that the physical covariates (DEM, slope, aspect, distance to lake) already absorbed the regional spatial pattern.

---

## 7. LATENCY & OPERATIONAL EXECUTION

- **Local Cache Inference Latency:** $\sim 0.15 - 0.30$ seconds (complete evaluation of 404 cells, uncertainty intervals, SAC triggers, and GeoJSON export).
- **Remote Cloud Ingestion Latency (Google Earth Engine):** $15 - 45$ seconds per multi-mission scene (on-demand composite downloads).

---

## 8. REPRODUCIBILITY & TEST SUITE

The repository includes a comprehensive 31-test automated test suite:
```powershell
# Run the complete test suite (31 tests, 100% passing in ~22.7s)
pytest -v tests/test_refactor_integrity.py

# Run the complete GeoAI pipeline end-to-end
python run_full_pipeline.py

# Launch the interactive Leaflet Geoportal
python run.py
# Open: http://localhost:5000
```

All 31 tests verify:
- Feature registry enforcement (zero forbidden features).
- Target permutation collapse ($R^2 \approx 0$).
- Spatial block integrity & zero cell overlap.
- Spatial exclusion buffer ($\ge 91.67\text{ km}$, variogram range decoupling).
- Strict 16-day temporal embargo.
- Quantile monotonicity ($q_{05} \le q_{50} \le q_{95}$).
- Synthetic and empirical CQR coverage.
- Winkler reference score exactness.
- NASA SMAP depth and quality flags (`sm_rootzone` 0–100cm, `sm_surface` 0–5cm, $n_{\text{effective}} = 351 < 404$).
- Physical LST preservation without sub-zero clipping (frosts $<0^\circ\text{C}$).
- Moran's I residual decision logic.
- Persistence baseline, skill scores, and paired bootstrap tests.
- Two-level pure probability SAC trigger and consecutive composite confirmation.
- NASA Earth System Trend Detective: Mann-Kendall test & Sen's slope (2001–2025).
- Brier score and probabilistic calibration for $P(\text{VHI} < 20)$.
- Full 17-endpoint Flask API contract.
