-- Seed 10 additional unassigned participants into the START Summit event.
-- Designed to give the skill-imputation classifier balanced signal:
--   * Each of the 9 skills has 3+ "high" (>=4) and 3+ "low" (<4) cases.
--   * Skills follow soft correlations (social + english tend to track leadership)
--     so kNN can recover them on a held-out test set.
-- Run via: SUPABASE_SERVICE_ROLE_KEY=... python scripts/apply_seed.py supabase/seed_10_signal_participants.sql

with ev as (
  select id from public.events where name = 'START Summit' limit 1
)
insert into public.participants
  (event_id, team_id, name, status,
   strength, driving, design, social, construction, english, german, photography, leadership)
select ev.id, null, v.name, 'pending',
       v.strength, v.driving, v.design, v.social, v.construction,
       v.english, v.german, v.photography, v.leadership
from ev,
(values
  ('Lena Bauer',      2, 2, 3, 5, 2, 5, 4, 3, 5),
  ('Mateo Fischer',   5, 4, 2, 2, 5, 2, 2, 1, 1),
  ('Yara Hoffmann',   3, 2, 5, 4, 2, 4, 3, 5, 4),
  ('Noah Wagner',     4, 5, 1, 1, 4, 2, 1, 1, 1),
  ('Amelie Roth',     2, 2, 4, 5, 1, 5, 5, 4, 5),
  ('Felix Krause',    4, 3, 2, 2, 4, 3, 4, 2, 2),
  ('Iris Lehmann',    3, 2, 5, 4, 2, 4, 3, 5, 4),
  ('Oskar Becker',    3, 3, 3, 1, 3, 2, 2, 2, 1),
  ('Sophia Lange',    2, 3, 3, 4, 2, 5, 4, 3, 4),
  ('Linus Berger',    5, 5, 2, 2, 5, 2, 3, 1, 2)
) as v(name, strength, driving, design, social, construction, english, german, photography, leadership);
