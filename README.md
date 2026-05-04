# FPL Predictor — ML-Driven Fantasy Premier League Team Selection

An end-to-end machine learning system that predicts Fantasy Premier League player points and uses integer linear programming to select an optimal squad. Evaluated across a full Premier League season, the model-selected squad outperformed the average FPL manager by **39 points** over 32 gameweeks.

---

## Overview

Most FPL managers rely on intuition, form, and fixture difficulty when picking their team. This project replaces that with a data-driven pipeline: pull live FPL data, engineer predictive features, train a gradient boosting model, and solve a constrained optimisation problem to pick the best possible squad within FPL's rules.

The system is exposed via a Streamlit app with three components: a live team picker, a model performance dashboard, and a methodology explainer.

---

## Results

Evaluated using walk-forward cross-validation across GW2–GW34 of the 2024/25 season. At each gameweek, the model was trained only on prior gameweeks (no lookahead), then used to pick a squad for the following week.

| | Points (GW2–34) |
|---|---|
| Model-selected squad | **1,615** |
| Average FPL manager | 1,576 |
| Outperformance | **+39 pts** |

> **Note on predicted vs actual:** The model's raw point predictions are systematically higher than actuals — a known characteristic of gradient boosting on noisy targets like FPL points. The predictions are used ordinally (to rank players) rather than cardinally, so the optimiser still selects a strong squad even when absolute accuracy is limited. Calibration is an area for future work.

---

## Architecture

```
FPL API
   │
   ▼
Data Engineering (Kedro)
   │  ├── Player & team metadata
   │  ├── Gameweek-by-gameweek player history (parallelised)
   │  └── Fixture data
   │
   ▼
Feature Engineering
   │  ├── Rolling window averages (configurable windows)
   │  └── Exponential weighted means (EWM) to weight recent form
   │
   ▼
Model Training (LightGBM / XGBoost via sklearn Pipeline)
   │  ├── Walk-forward cross-validation (no data leakage)
   │  └── Experiment tracking via MLflow
   │
   ▼
Team Optimisation (PuLP — Integer Linear Programming)
   │  ├── Maximise predicted XI points + captain bonus
   │  ├── Constraints: £100m budget, 3-per-club, position rules
   │  └── Flexible formation selection (3-4-3 through 5-3-2)
   │
   ▼
Streamlit App
   ├── Live team picker (next GW)
   ├── Model performance dashboard
   └── Methodology
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Pipeline orchestration | [Kedro](https://kedro.org/) |
| Gradient boosting | LightGBM / XGBoost (sklearn API) |
| Optimisation | [PuLP](https://coin-or.github.io/pulp/) (CBC solver) |
| Experiment tracking | MLflow |
| Feature engineering | Custom rolling/EWM pipeline (extensible strategy pattern) |
| Data source | [FPL REST API](https://fantasy.premierleague.com/api/) |
| App | Streamlit |

---

## Key Technical Details

**Walk-forward validation** — The model is never trained on future data. For each gameweek `t`, it trains on GW1 to `t-1` and predicts GW`t`. This mirrors real deployment conditions and gives honest out-of-sample performance numbers.

**Feature engineering** — Player history features are constructed using a configurable strategy pattern (`FeatureEngineeringPipeline`) supporting both rolling window means and exponential weighted means, with automatic shift-by-one to prevent leakage.

**Optimisation formulation** — Squad selection is modelled as an ILP: binary variables for squad inclusion, starting XI, captain, and formation. The solver maximises expected XI points (with captain double) subject to FPL's hard constraints. Formation is selected endogenously — the solver picks whichever of the five valid formations maximises points.

**Parallelised data ingestion** — Player gameweek history is fetched concurrently using `ThreadPoolExecutor`, reducing ingestion time significantly across the ~700 FPL players.

---

## Project Structure

```
fpl-modelling/
└── src/fpl_modelling/
    ├── FPL_API.py                  # FPL API client
    ├── pipeline_registry.py        # Kedro pipeline registry
    └── pipelines/
        ├── data_engineering/       # Data ingestion & storage
        ├── data_processing/        # Preprocessing & expanding window
        ├── data_science/           # Feature engineering, training, evaluation
        └── optimisation/           # ILP team selection (PuLP)
```

---

## Running the Project

**Prerequisites:** Python 3.10+, Kedro, PuLP, MLflow

```bash
pip install kedro pulp mlflow lightgbm xgboost streamlit

# Pull latest FPL data
kedro run --pipeline update_tables

# Train model for a given gameweek
kedro run --pipeline train_model --params predicting_gameweek:35,model_num:1

# Generate predictions and pick optimal team
kedro run --pipeline gameweek_prediction --params predicting_gameweek:35,model_num:1

# Run historical evaluation
kedro run --pipeline eval_model

# Launch Streamlit app
streamlit run app/app.py
```

---

## Future Work

- **Calibration:** Apply isotonic regression or Platt scaling to reduce systematic overestimation in raw predictions
- **Fixture difficulty:** Incorporate opponent strength and home/away as model features
- **Transfer optimisation:** Extend the ILP to recommend weekly transfers given an existing squad (framework already built in `TransferOptimizer.py`)
- **Chip strategy:** Model optimal timing for Wildcard, Free Hit, and Triple Captain
