-- Event Team Manager: initial schema
-- Three tables: events, teams (FK -> events), participants (FK -> teams).

create table if not exists public.events (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  date date,
  location text,
  created_at timestamp default now()
);

create table if not exists public.teams (
  id uuid primary key default gen_random_uuid(),
  event_id uuid references public.events(id) on delete cascade,
  name text not null,
  created_at timestamp default now()
);

create table if not exists public.participants (
  id uuid primary key default gen_random_uuid(),
  team_id uuid references public.teams(id) on delete cascade,
  name text not null,
  skill text,
  status text default 'pending',
  created_at timestamp default now()
);

-- Enable Row Level Security so the anon key can't read/write without a policy.
alter table public.events       enable row level security;
alter table public.teams        enable row level security;
alter table public.participants enable row level security;

-- Beginner-friendly RLS: any authenticated user can read and write.
-- Tighten these for production (e.g. organiser-only writes).
create policy "authenticated read events"
  on public.events for select to authenticated using (true);
create policy "authenticated write events"
  on public.events for all to authenticated using (true) with check (true);

create policy "authenticated read teams"
  on public.teams for select to authenticated using (true);
create policy "authenticated write teams"
  on public.teams for all to authenticated using (true) with check (true);

create policy "authenticated read participants"
  on public.participants for select to authenticated using (true);
create policy "authenticated write participants"
  on public.participants for all to authenticated using (true) with check (true);
