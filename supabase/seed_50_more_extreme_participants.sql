-- Seed 50 more unassigned participants with EXTREME skill profiles (lots of 5s and 0s)
-- to give the ML recommender clearer skill gaps to fill.
-- Run in the Supabase SQL editor while logged in (RLS requires authenticated user).

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
  -- Strength specialists
  ('Max Bauer',        5, 1, 0, 0, 1, 0, 1, 0, 0),
  ('Klaus Wagner',     5, 0, 0, 1, 2, 0, 1, 0, 1),
  ('Felix Hartmann',   5, 1, 0, 0, 1, 1, 0, 0, 0),
  ('Bjorn Schmitt',    5, 0, 1, 0, 0, 0, 1, 0, 1),
  ('Brent Cole',       5, 0, 0, 0, 0, 0, 0, 0, 0),
  -- Design specialists
  ('Lena Fischer',     0, 0, 5, 1, 0, 2, 1, 4, 0),
  ('Mila Wolf',        1, 0, 5, 0, 0, 1, 0, 4, 0),
  ('Yuki Sato',        0, 0, 5, 1, 0, 3, 0, 5, 1),
  ('Anya Petrov',      1, 1, 5, 0, 1, 1, 0, 3, 0),
  -- Photography specialists
  ('Ravi Patel',       0, 1, 2, 1, 0, 2, 0, 5, 0),
  ('Lukas Klein',      1, 0, 3, 0, 1, 1, 1, 5, 0),
  ('Pia Sommer',       0, 0, 2, 1, 0, 1, 0, 5, 0),
  ('Yusuf Demir',      0, 0, 0, 0, 0, 0, 0, 5, 0),
  -- Leadership specialists
  ('Sven Larsen',      1, 1, 0, 3, 0, 4, 0, 0, 5),
  ('Diego Cruz',       0, 1, 0, 4, 0, 3, 0, 0, 5),
  ('Mei Tanaka',       1, 0, 1, 3, 0, 4, 0, 1, 5),
  ('Tariq Hassan',     0, 0, 0, 2, 0, 3, 0, 0, 5),
  -- Social/communicator specialists
  ('Nina Reiter',      0, 1, 1, 5, 0, 3, 0, 1, 2),
  ('Olga Sokolov',     1, 0, 0, 5, 0, 2, 0, 0, 1),
  ('Ahmed Karim',      0, 0, 0, 5, 0, 4, 0, 0, 2),
  ('Bea Friedrich',    1, 0, 1, 5, 0, 0, 0, 0, 1),
  -- English speakers
  ('Connor Walsh',     0, 1, 0, 1, 0, 5, 0, 0, 1),
  ('Liam Byrne',       0, 0, 1, 0, 0, 5, 0, 0, 0),
  ('Grace Murphy',     1, 0, 0, 2, 0, 5, 0, 0, 1),
  -- German-only speakers
  ('Heinz Baumann',    1, 1, 0, 0, 1, 0, 5, 0, 0),
  ('Greta Vogel',      0, 0, 1, 1, 0, 0, 5, 0, 0),
  ('Wolfgang Krause',  1, 1, 0, 0, 0, 0, 5, 0, 0),
  ('Ingrid Holm',      0, 0, 0, 0, 0, 0, 5, 0, 0),
  -- Construction specialists
  ('Tobias Berg',      3, 1, 0, 0, 5, 1, 1, 0, 0),
  ('Marek Novak',      4, 1, 0, 0, 5, 0, 1, 0, 1),
  ('Pablo Reyes',      3, 2, 1, 1, 5, 0, 0, 1, 0),
  ('Sami Korhonen',    4, 0, 0, 0, 5, 0, 1, 0, 0),
  -- Driving specialists
  ('Erik Solberg',     2, 5, 0, 0, 1, 0, 1, 0, 0),
  ('Aleks Markov',     1, 5, 0, 1, 1, 1, 0, 0, 0),
  ('Bo Andersson',     1, 5, 1, 0, 2, 0, 0, 0, 0),
  -- Design+photography combos
  ('Iris Hahn',        0, 0, 5, 0, 0, 1, 0, 5, 0),
  ('Naomi Klein',      0, 0, 5, 1, 0, 2, 1, 5, 0),
  ('Tess Vogel',       1, 0, 4, 0, 1, 0, 0, 5, 1),
  ('Ren Yamada',       0, 0, 5, 0, 0, 4, 0, 5, 0),
  -- Leader+communicator combos
  ('Victor Lange',     0, 0, 0, 5, 0, 4, 0, 0, 5),
  ('Cleo Schwarz',     1, 0, 1, 5, 0, 3, 0, 1, 4),
  ('Omar Nasser',      0, 1, 0, 4, 0, 5, 0, 0, 5),
  -- Strength+construction combos
  ('Hans Becker',      5, 1, 0, 0, 4, 0, 1, 0, 0),
  ('Kai Fischer',      5, 2, 0, 1, 5, 0, 0, 0, 1),
  ('Ole Hansen',       4, 1, 0, 0, 5, 0, 1, 0, 0),
  -- English+leadership combos
  ('Charlotte Reid',   0, 0, 1, 2, 0, 5, 0, 0, 5),
  ('James Whitman',    1, 0, 0, 1, 0, 5, 0, 1, 4),
  ('Aisha Brown',      0, 0, 0, 2, 0, 5, 1, 0, 5),
  -- Misc extreme combos
  ('Karim Aziz',       4, 5, 0, 0, 3, 1, 0, 0, 0),
  ('Sigrid Lund',      0, 0, 1, 0, 0, 0, 5, 0, 5)
) as v(name, strength, driving, design, social, construction, english, german, photography, leadership);
