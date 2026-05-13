# STARTCrew

A simple Streamlit web app for managing teams at events such as hackathons.
Built for the HSG St. Gallen Computer Science project.

## Problem statement

Event organizers struggle to manage participants, assign them to teams, and
get a quick overview of team composition. This app gives organizers a single
place to create events, build teams, add participants, visualise team
composition, and use a small machine learning model to recommend the
unassigned candidates that best fill each team's skill gaps.

## Features

- **Events, teams, participants** — full CRUD with cascading delete buttons.
- **Skill ratings & gap-weighted kNN recommendations** — rate participants on 12 skills (strength, driving, design, social, construction, english, german, photography, leadership, communication, experience, problem solving) on a 1–5 scale, set per-team minimum thresholds, and let scikit-learn's `NearestNeighbors` recommend the unassigned candidates whose strengths best fill each team's skill gaps.
- **Event-scoped participant pool** — the Participants page lists everyone in an event regardless of team, so you can add people first and assign them later.
- **Dashboard** — bar chart of team sizes and a skill-gap radar (Netzdiagramm) comparing team requirements to actual team averages.

## Tech stack

- **UI:** Streamlit
- **Database & Auth:** Supabase (Postgres + REST API + Auth)
- **ML:** scikit-learn `NearestNeighbors` (gap-weighted kNN recommender)
- **Data:** pandas, numpy
- **Charts:** matplotlib + Streamlit built-ins
- **Tests:** pytest

## Project structure

```
.
├── app.py            # Streamlit UI and page routing
├── auth.py           # Supabase Auth helpers
├── database.py       # Supabase CRUD operations
├── ml.py             # Gap-weighted kNN recommender
├── tests/            # pytest unit tests for the ML module
├── requirements.txt  # Python dependencies
├── .env.example      # Template for credentials
└── README.md
```

## Run locally

1. **Clone the repo** and enter the project directory.
2. **Create a virtual environment** and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Set up Supabase**:
   - Create a project at https://supabase.com.
   - Apply the schema by running every file in `supabase/migrations/` in order. Either:
     - Run `supabase link --project-ref <your-ref>` then `supabase db push` (requires the Supabase CLI and your DB password), **or**
     - Open each `.sql` file under `supabase/migrations/` and paste the contents into the Supabase SQL editor in chronological order.
   - In Authentication, enable Email/Password sign-in and create a user.
4. **Configure credentials**:
   ```bash
   cp .env.example .env
   # Edit .env and paste in your Supabase URL and anon key.
   ```
5. **Run the app**:
   ```bash
   streamlit run app.py
   ```

## How the 8 grading requirements are met

1. **Real-world problem solved** — Manages event participants, teams, and team-fit recommendations for organizers.
2. **Web-based UI** — Streamlit pages with sidebar navigation (`app.py`).
3. **Database integration** — Supabase Postgres with three relational tables (events, teams, participants), accessed via `database.py`.
4. **User authentication** — Supabase Auth email/password login via `auth.py`.
5. **Data visualisation** — Bar chart of team sizes and skill-gap radar chart (team requirements vs. actual team averages) on the Dashboard page.
6. **Source code documentation** — Every function has a docstring; every section has comments.
7. **Machine learning component** — `ml.py` builds a 9-dimensional skill-gap vector per team and uses scikit-learn's `NearestNeighbors` to recommend the unassigned candidates whose strengths best fill those gaps. Recommendations are surfaced on both the Teams and ML Insights pages.
8. **Reproducibility & deployment** — `requirements.txt`, `.env.example`, single-command run (`streamlit run app.py`), and `pytest` unit tests for the ML module.

## Contribution matrix

| Team member | Contributions |
|---|---|
| _Name 1_ | _e.g. database schema, `database.py`, README setup_ |
| _Name 2_ | _e.g. `app.py` UI, navigation, dashboard charts_ |
| _Name 3_ | _e.g. `auth.py`, Supabase setup, deployment_ |
| _Name 4_ | _e.g. `ml.py`, classifier design, evaluation_ |

> Replace placeholders before submission.

## Demo video

_Link to be added before submission: `<paste-video-url>`_
