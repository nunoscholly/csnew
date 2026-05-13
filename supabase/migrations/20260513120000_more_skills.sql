-- Adds three more skill dimensions (communication, experience, problem_solving)
-- to participants, plus their per-team minimum thresholds on teams.
-- Existing rows get default values (3 for participants, 0 for team thresholds).

alter table public.participants
  add column if not exists communication   int default 3,
  add column if not exists experience      int default 3,
  add column if not exists problem_solving int default 3;

alter table public.teams
  add column if not exists req_communication   int default 0,
  add column if not exists req_experience      int default 0,
  add column if not exists req_problem_solving int default 0;
