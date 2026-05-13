# Lückengewichteter kNN-Empfehler: schlägt für jedes Team Teilnehmer vor, deren Fähigkeiten die Skill-Lücken des Teams füllen

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

SKILL_COLUMNS = [
    "strength",
    "driving",
    "design",
    "social",
    "construction",
    "english",
    "german",
    "photography",
    "leadership",
    "communication",
    "experience",
    "problem_solving",
]

# Deutsche Anzeigebezeichnungen für die Skill-Spalten (UI-Labels)
SKILL_LABELS_DE = {
    "strength": "Stärke",
    "driving": "Fahren",
    "design": "Design",
    "social": "Soziale Kompetenz",
    "construction": "Bau",
    "english": "Englisch",
    "german": "Deutsch",
    "photography": "Fotografie",
    "leadership": "Führung",
    "communication": "Kommunikation",
    "experience": "Erfahrung",
    "problem_solving": "Problemlösung",
}


def skill_label(skill: str) -> str:
    # Liefert das deutsche Anzeige-Label; fällt auf Title-Case zurück, falls kein Eintrag existiert
    return SKILL_LABELS_DE.get(skill, skill.replace("_", " ").title())


def _backfill_missing_skills(df: pd.DataFrame, default: int) -> pd.DataFrame:
    # Falls eine in SKILL_COLUMNS deklarierte Skill-Spalte im DataFrame fehlt
    # (z.B. weil die DB-Migration noch nicht ausgeführt wurde), wird sie mit
    # `default` eingefügt, damit nachgelagerte Operationen keinen KeyError
    # werfen. default=5 für Teamdaten (gilt als vollständig abgedeckt, gap=0)
    # und default=3 für Kandidaten (neutraler Mittelwert). Bei gap=0 ist das
    # Gewicht null, der konkrete Kandidatenwert beeinflusst das Ranking also
    # ohnehin nicht.
    missing = [c for c in SKILL_COLUMNS if c not in df.columns]
    if not missing:
        return df
    df = df.copy()
    for c in missing:
        df[c] = default
    return df


def team_gap_vector(team_participants: pd.DataFrame) -> np.ndarray:
    # Liefert einen Vektor der Dimension len(SKILL_COLUMNS) mit max(5 - team_max_skill, 0);
    # leeres Team -> nur 5er. Fehlende Skill-Spalten gelten als bereits abgedeckt
    # (gap=0) und beeinflussen somit keine Empfehlung.
    if team_participants.empty:
        return np.full(len(SKILL_COLUMNS), 5.0)
    df = _backfill_missing_skills(team_participants, default=5)
    team_max = df[list(SKILL_COLUMNS)].max(axis=0).to_numpy(dtype=float)
    return np.maximum(5.0 - team_max, 0.0)


def recommend_complementary(
    team_participants: pd.DataFrame,
    candidates_df: pd.DataFrame,
    k: int = 5,
) -> pd.DataFrame:
    # Bewertet Kandidaten per lückengewichtetem euklidischem kNN; abgedeckte Skills tragen 0 zur Distanz bei
    out_cols = ["id", "name", "distance", "gap_score", *SKILL_COLUMNS]
    if candidates_df.empty:
        return pd.DataFrame(columns=out_cols)

    g = team_gap_vector(team_participants)
    if not np.any(g):
        return pd.DataFrame(columns=out_cols)

    candidates_df = _backfill_missing_skills(candidates_df, default=3)
    skill_matrix = candidates_df[list(SKILL_COLUMNS)].to_numpy(dtype=float)
    target = np.where(g > 0, 5.0, 0.0)  # ideale Ergänzung: Skill 5 in jeder Lücken-Dimension, sonst 0

    # Vorab-Skalierung mit sqrt(g): einfache euklidische Distanz auf skalierten Vektoren
    # entspricht gewichteter euklidischer Distanz mit Gewichten g
    weights = np.sqrt(g)
    scaled_matrix = skill_matrix * weights
    scaled_target = target * weights

    n_neighbors = min(k, len(candidates_df))
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    knn.fit(scaled_matrix)
    distances, indices = knn.kneighbors([scaled_target])

    ordered = candidates_df.iloc[indices[0]].copy()
    ordered["distance"] = distances[0]
    ordered["gap_score"] = skill_matrix[indices[0]] @ g
    return ordered[out_cols].reset_index(drop=True)
