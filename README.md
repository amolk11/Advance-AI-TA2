# Bayesian Air Quality Risk and Mitigation Decision System
### Temporal Dynamic Bayesian Network for Air Quality Forecasting & Municipal Decision Support
**Location:** Revenue Colony–Shivajinagar, Pune, India (`IITM Station 11613`)  
**Methodology:** 1st-Order Markov Dynamic Bayesian Network (DAG), Dirichlet Parameter Smoothing (`BDeu`), Exact Variable Elimination, Calibration & Walk-Forward Time-Series Evaluation.

---

## Executive Summary

Urban air quality management requires decision-support tools that provide **well-calibrated probabilistic risk distributions** rather than black-box point forecasts. A single deterministic forecast (e.g., "AQI will be 145") fails to quantify the tail risk of severe pollution spikes, preventing public health officials from taking preemptive action.

This project implements an end-to-end, scientifically validated **Temporal Dynamic Bayesian Network (DBN)** that forecasts next-day AQI states:
$$P(\text{AQI}_t \mid \text{AQI}_{t-1}, \text{Pollutant}_{t-1}, \text{Season}_t)$$

The system explicitly captures:
1. **Atmospheric Inertia:** 1-day autoregressive persistence ($\text{AQI}_{t-1} \to \text{AQI}_t$).
2. **Chemical Precursor Inertia:** Dominant pollutant species at $t-1$ ($\text{Pollutant}_{t-1} \to \text{AQI}_t$).
3. **Exogenous Meteorological Forcing:** Astronomical calendar seasons capturing thermal inversions and monsoon washout ($\text{Season}_t \to \text{AQI}_t$).

---

## Mathematical & Graphical Model Formulation

```text
┌──────────────────────────────┐
│  Previous AQI State (t-1)    │──────────┐
│  {Low, Moderate, High}       │          │
└──────────────────────────────┘          │
                                          ▼
┌──────────────────────────────┐     ┌──────────────────────────────┐
│  Dominant Pollutant (t-1)    │────>│   Target AQI State (t)       │
│  {CO, NO2, O3}               │     │   {Low, Moderate, High}      │
└──────────────────────────────┘     └──────────────────────────────┘
                                          ▲
┌──────────────────────────────┐          │
│  Astronomical Season (t)     │──────────┘
│  {Monsoon, Post-M, Win, Sum} │
└──────────────────────────────┘
```

### 1. State-Space Discretization
* **Low ($\le 100$):** Good to Satisfactory air quality.
* **Moderate ($101 - 200$):** Moderate health concern for sensitive demographics.
* **High ($> 200$):** Poor, Unhealthy, or Hazardous air quality requiring immediate curbs.

### 2. Bayesian Dirichlet Smoothing (`BDeu` Prior)
Empirical data inevitably contains unobserved parent-child configurations (e.g., High AQI during heavy Monsoon rain). To prevent **zero-frequency probability traps** (where $P(E) = 0$ breaks inference), parameters are estimated using a **Bayesian Dirichlet Prior (`BDeu`)** with equivalent sample size $\alpha = 10$:
$$P(Y_t = k \mid \text{Pa}(Y_t) = j) = \frac{N_{j,k} + \frac{\alpha}{r \cdot q_j}}{N_j + \frac{\alpha}{q_j}}$$
This guarantees that all Conditional Probability Tables (CPDs) are strictly non-zero, normalized, and mathematically sound.

---

## Data Leakage & Temporal Continuity Audit

### 1. The 957-Day Sensor Blackout Resolution
An audit of the raw historical records (`2020–2026`) revealed a **957-day sensor blackout (missing data gap)** between **July 7, 2022** ($\text{AQI}=73.2$) and **February 18, 2025** ($\text{AQI}=96.4$). 
* **The Fix:** The modeling pipeline isolates the modern, continuous **2025–2026 series** ($N=294$) and strictly enforces $\Delta\text{Date} = 1\text{ day}$. This discards non-consecutive transitions and yields **286 valid 1-day Markov pairs** ($t-1 \to t$).

### 2. Zero-Leakage Feature Audit

| Feature | Temporal Index | Data Source | Leakage Status |
|:---|:---:|:---|:---:|
| `previous_aqi_state` | $t - 1$ | Observed AQI state at day $t-1$ | **SAFE** (Strictly Lagged) |
| `previous_dominant_pollutant` | $t - 1$ | Dominant pollutant at day $t-1$ | **SAFE** (Strictly Lagged) |
| `season` | $t$ | Astronomical Calendar Date at day $t$ | **SAFE** (Pre-known Exogenous) |
| `target_aqi_state` | $t$ | Ground truth state at day $t$ | **TARGET** (Held out during inference) |

---

## Experimental Benchmark Results

Evaluated on the **unseen chronological holdout test set** ($N=58$ samples, Nov 2025 – Jan 2026; 80% train / 20% test split):

### 1. Comparative Model Performance Table

| Model | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 | Brier Score | Log Loss | Key Characteristic |
|:---|---:|---:|---:|---:|---:|---:|:---|
| **Naive Persistence** | 86.21% | 67.51% | 0.4500 | 0.8621 | 0.2759 | 32.1568 | Standard point baseline ($\hat{Y}_t = Y_{t-1}$) |
| **Logistic Regression** | 87.93% | 68.49% | 0.4645 | 0.8753 | 0.1934 | 5.5311 | Linear probabilistic classifier |
| **Random Forest** | 86.21% | 67.51% | 0.4500 | 0.8621 | 0.1949 | 29.4175 | Non-linear ensemble |
| **Bayesian Network (BDeu)** | 86.21% | 67.51% | 0.4500 | 0.8621 | 0.2187 | **2.6681** | **Best probabilistic Log Loss & calibrated uncertainty** |

* **Expected Calibration Error (ECE):** **`0.0805`** (Demonstrating tight alignment between predicted confidence and empirical accuracy).

> **Scientific Disclosure:** The test period contained 0 observations of the "High" AQI state. Consequently, High-state predictive sensitivity cannot be empirically measured on this specific test holdout.

### 2. Walk-Forward Expanding Window Cross-Validation (3 Folds)

| Fold | Training Period | Test Period | Train $N$ | Test $N$ | Accuracy | Macro F1 | Brier Score | Log Loss |
|:---:|:---|:---|:---:|:---:|---:|---:|---:|---:|
| **Fold 1** | Feb 19, 2025 – Jun 17, 2025 | Jun 18, 2025 – Sep 22, 2025 (Monsoon) | 114 | 57 | **87.72%** | **0.5534** | 0.2240 | 3.8809 |
| **Fold 2** | Feb 19, 2025 – Sep 22, 2025 | Sep 23, 2025 – Nov 21, 2025 (Post-Monsoon) | 171 | 57 | **42.11%** | **0.2729** | 0.5875 | 1.6862 |
| **Fold 3** | Feb 19, 2025 – Nov 21, 2025 | Nov 22, 2025 – Jan 27, 2026 (Winter) | 228 | 57 | **85.96%** | **0.4183** | 0.2108 | 2.6956 |

---

## Prototype Decision Support Layer

Rather than forcing binary deterministic actions, the Bayesian posterior distributions feed directly into an operational risk matrix:

| Risk Tier | Posterior Probability Thresholds | Recommended Municipal Action |
|:---|:---|:---|
| **Routine / Low Risk** | $P(\text{Moderate} \cup \text{High}) < 0.30$ | Normal outdoor activity; standard baseline monitoring. |
| **Elevated Advisory** | $0.30 \le P(\text{Moderate}) < 0.70$ and $P(\text{High}) < 0.20$ | Advise vulnerable demographic groups (asthma/elderly) to reduce prolonged exertion. |
| **High Vigilance** | $P(\text{Moderate}) \ge 0.70$ or $0.20 \le P(\text{High}) < 0.50$ | Deploy mechanical road sweeping; traffic police manage high-congestion corridors. |
| **Severe Alert** | $P(\text{High}) \ge 0.50$ | Issue public health emergency warning; activate industrial emission curbs. |

---

##  Repository Structure

```text
advanceai/
├── data/
│   └── station_1_Revenue_Colony-Shivajinagar_Pu.csv   # Raw monitoring dataset (IITM Station 11613)
├── notebooks/
│   └── temporal_air_quality_modeling.ipynb
├── requirements.txt                                   # Python dependencies
└── README.md                                          # Project documentation & interview guide
```

---

##  Quickstart & Execution

```bash
# 1. Clone or open the workspace and install requirements
pip install -r requirements.txt

# 2. Run the complete mathematical audit & benchmark experiments
python scripts/run_audit_and_experiments.py

# 3. Re-build and pre-render the master Jupyter notebook
python scripts/generate_end_to_end_notebook.py
```

---

## 🎓 Technical Viva & Interview FAQ

<details>
<summary><b>1. Why use a Bayesian Network instead of a standard classifier like XGBoost?</b></summary>
Standard classifiers output uncalibrated, point-estimate class scores that lack causal interpretability. A Bayesian Network factors the joint distribution over a Directed Acyclic Graph (DAG), enabling <b>exact evidential and interventional querying</b> ($P(Y_t \mid E)$), uncertainty representation, and principled parameter smoothing through Dirichlet priors without overfitting small sample regimes.
</details>

<details>
<summary><b>2. Why is a chronological split mandatory for this task?</b></summary>
Air pollution exhibits strong autocorrelation across consecutive days. A random train/test split leaks future information into past predictions (temporal lookahead bias), artificially inflating accuracy while failing in real-world deployment. Chronological splitting ensures the model is evaluated strictly on unseen future time horizons.
</details>

<details>
<summary><b>3. What role does the BDeu prior play in parameter estimation?</b></summary>
In discrete Bayesian networks, Conditional Probability Tables (CPDs) are estimated from sample counts. Rare combinations (such as High AQI in Monsoon) yield zero counts. Maximum Likelihood Estimation (MLE) assigns these zero probability, causing division-by-zero or infinite log-loss during inference. BDeu smoothing applies equivalent sample size Dirichlet pseudo-counts ($\alpha = 10$), ensuring smooth, non-zero probabilities.
</details>

<details>
<summary><b>4. How does multi-step rollout differ from true multi-day weather forecasting?</b></summary>
A true 7-day weather forecast requires dynamic daily numerical weather prediction (NWP) inputs. In this graphical model, the 7-day simulation represents an <b>iterative Markov state propagation under fixed/conditioned environmental regimes</b>, showing the asymptotic steady-state distribution without teacher forcing.
</details>

---

## License & Attribution
* **Dataset:** Central Pollution Control Board (CPCB) & Indian Institute of Tropical Meteorology (IITM) air monitoring station, Revenue Colony–Shivajinagar, Pune.
