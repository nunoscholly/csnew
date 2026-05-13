# One-off seed runner: parses a participants seed SQL file and inserts rows
# via supabase-py using the service role key (bypasses RLS).
#
# Usage:
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
    raise SystemExit("SUPABASE_SERVICE_ROLE_KEY env var is required.")

sql_path = Path(sys.argv[1])
text = sql_path.read_text()

client = create_client(URL, key)

events = client.table("events").select("id,name").eq("name", EVENT_NAME).execute()
if not events.data:
    raise SystemExit(f"Event {EVENT_NAME!r} not found.")
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

print(f"Parsed {len(rows)} participant rows from {sql_path.name}")
res = client.table("participants").insert(rows).execute()
print(f"Inserted {len(res.data)} participants into event {event_id}")
