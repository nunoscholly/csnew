# Einmal-Seed-Runner: parst eine Teilnehmer-Seed-SQL-Datei und fügt die Zeilen
# über supabase-py mit dem Service-Role-Key ein (umgeht RLS).
#
# Verwendung:
#   SUPABASE_SERVICE_ROLE_KEY=... python scripts/apply_seed.py supabase/seed_50_more_extreme_participants.sql

import os
import re
import sys
from pathlib import Path

from supabase import create_client

URL = "https://grdewbuenxzqtqerqbhv.supabase.co"
EVENT_NAME = "START Summit"

key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
if not key:
    raise SystemExit("Umgebungsvariable SUPABASE_SERVICE_ROLE_KEY ist erforderlich.")

sql_path = Path(sys.argv[1])
text = sql_path.read_text()

client = create_client(URL, key)

events = client.table("events").select("id,name").eq("name", EVENT_NAME).execute()
if not events.data:
    raise SystemExit(f"Veranstaltung {EVENT_NAME!r} nicht gefunden.")
event_id = events.data[0]["id"]

pattern = re.compile(r"\('([^']+)',\s*([\d,\s]+)\)")
rows = []
for m in pattern.finditer(text):
    name = m.group(1)
    nums = [int(x.strip()) for x in m.group(2).split(",") if x.strip()]
    if len(nums) != 9:
        continue
    rows.append({
        "event_id": event_id,
        "team_id": None,
        "name": name,
        "status": "pending",
        "strength":     nums[0],
        "driving":      nums[1],
        "design":       nums[2],
        "social":       nums[3],
        "construction": nums[4],
        "english":      nums[5],
        "german":       nums[6],
        "photography":  nums[7],
        "leadership":   nums[8],
    })

print(f"{len(rows)} Teilnehmerzeilen aus {sql_path.name} geparst")
res = client.table("participants").insert(rows).execute()
print(f"{len(res.data)} Teilnehmer in Veranstaltung {event_id} eingefügt")
