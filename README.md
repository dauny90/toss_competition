# Toss NEXT ML Challenge: CTR Prediction

<p>
  <img src="https://static.toss.im/icons/svg/logo-toss-blue.svg" alt="Toss" height="36">
  <img src="https://dacon.io/_nuxt/img/main-logo.b9ffbb6.svg" alt="DACON" height="36">
</p>

This repository is a Python-based ML project for the Dacon competition
"Toss NEXT ML Challenge: CTR prediction".

Competition page:
https://dacon.io/en/competitions/official/236575/overview/description

## Project Goal
Predict ad click-through rate (CTR) from Toss app ad impression and click logs.
Features are anonymized, so feature meanings are not provided.

## Competition Notes
- Models should be suitable for real-time serving.
- Multiple models with different inference cycles are allowed.
- Two stages: preliminary selection by private leaderboard and a final stage
  that also evaluates a development report and source code.
- Eligibility includes full-time availability from mid-December 2025 and excludes
  Toss employees and affiliates (see the competition page for details).

## Repository Structure
- `data/`: datasets (`train`, `test`, `submission`)
- `notebooks/`: exploratory and experiment notebooks
- `src/`: pipeline code (data, features, models, training, evaluation, utils)
- `models/`: checkpoints and artifacts
- `reports/`: figures and metrics
- `scripts/`: CLI utilities
- `tests/`: tests

## Setup (Python)
1. Create a virtual environment: `python3 -m venv .venv`
2. Activate it: `source .venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
