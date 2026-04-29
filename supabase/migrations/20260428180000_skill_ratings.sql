-- Skill ratings on participants + minimum-skill thresholds on teams.
-- Also makes participants.team_id nullable and adds participants.event_id
-- so participants can exist as an event-level pool independent of any team.

alter table public.participants
  add column if not exists strength     int default 3,
  add column if not exists driving      int default 3,
  add column if not exists design       int default 3,
  add column if not exists social       int default 3,
  add column if not exists construction int default 3,
  add column if not exists english      int default 3,
  add column if not exists german       int default 3,
  add column if not exists photography  int default 3,
  add column if not exists leadership   int default 3,
  add column if not exists event_id     uuid references public.events(id) on delete cascade;

alter table public.participants
  alter column team_id drop not null;

-- Backfill event_id for any pre-existing rows from their team.
update public.participants p
   set event_id = t.event_id
  from public.teams t
 where p.team_id = t.id and p.event_id is null;

-- After backfill, enforce that every participant belongs to an event.
alter table public.participants
  alter column event_id set not null;

alter table public.teams
  add column if not exists req_strength     int default 0,
  add column if not exists req_driving      int default 0,
  add column if not exists req_design       int default 0,
  add column if not exists req_social       int default 0,
  add column if not exists req_construction int default 0,
  add column if not exists req_english      int default 0,
  add column if not exists req_german       int default 0,
  add column if not exists req_photography  int default 0,
  add column if not exists req_leadership   int default 0;
