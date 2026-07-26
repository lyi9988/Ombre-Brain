create extension if not exists pgcrypto;

alter table public.memories
add column if not exists updated_at timestamptz default now();

alter table public.memories
add column if not exists resolved boolean default false;

alter table public.memories
add column if not exists digested boolean default false;

alter table public.memories
add column if not exists anchor boolean default false;

alter table public.memories
add column if not exists confidence double precision default 0.5;

alter table public.memories
add column if not exists period text;

alter table public.memories
add column if not exists date text;

alter table public.memories
add column if not exists comments jsonb default '[]'::jsonb;

alter table public.memories
add column if not exists comment_count integer default 0;

update public.memories
set resolved = false
where resolved is null;

update public.memories
set digested = false
where digested is null;

update public.memories
set anchor = false
where anchor is null;

update public.memories
set confidence = 0.5
where confidence is null;

update public.memories
set comments = '[]'::jsonb
where comments is null;

update public.memories
set comment_count = case
  when jsonb_typeof(comments) = 'array' then jsonb_array_length(comments)
  else 0
end
where comment_count is null;

create or replace function public.set_memories_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists trg_memories_updated_at on public.memories;

create trigger trg_memories_updated_at
before update on public.memories
for each row
execute function public.set_memories_updated_at();

drop function if exists public.create_memory(
  text, text, text, text[], text[], text, double precision,
  double precision, double precision, boolean, timestamptz
);

drop function if exists public.create_memory(
  text, text, text, text[], text[], text, double precision,
  double precision, double precision, boolean, boolean, boolean, timestamptz
);

drop function if exists public.create_memory(
  text, text, text, text[], text[], text, double precision,
  double precision, double precision, boolean, boolean, boolean, timestamptz, boolean
);

drop function if exists public.create_memory(
  text, text, text, text[], text[], text, double precision,
  double precision, double precision, boolean, boolean, boolean, timestamptz, boolean,
  double precision, text, text, jsonb, integer
);

create or replace function public.create_memory(
  p_id text default null,
  p_title text default '未命名记忆',
  p_type text default 'dynamic',
  p_domain text[] default array['未分类'],
  p_tags text[] default '{}',
  p_content text default '',
  p_valence double precision default 0.5,
  p_arousal double precision default 0.5,
  p_importance double precision default 5.0,
  p_pinned boolean default false,
  p_resolved boolean default false,
  p_digested boolean default false,
  p_time timestamptz default now(),
  p_anchor boolean default false,
  p_confidence double precision default 0.5,
  p_period text default null,
  p_date text default null,
  p_comments jsonb default '[]'::jsonb,
  p_comment_count integer default null
)
returns public.memories
language plpgsql
security definer
set search_path = public
as $$
declare
  result public.memories;
  memory_id text;
  memory_type text;
  memory_comments jsonb;
  memory_comment_count integer;
begin
  memory_id := coalesce(nullif(p_id, ''), 'chatgpt_' || replace(gen_random_uuid()::text, '-', ''));
  memory_type := case
    when p_type in ('dynamic', 'permanent', 'feel', 'archived') then p_type
    else 'dynamic'
  end;
  memory_comments := coalesce(p_comments, '[]'::jsonb);
  if jsonb_typeof(memory_comments) <> 'array' then
    memory_comments := '[]'::jsonb;
  end if;
  memory_comment_count := coalesce(p_comment_count, jsonb_array_length(memory_comments), 0);

  insert into public.memories (
    id, title, type, domain, tags, content,
    valence, arousal, importance, pinned,
    resolved, digested, anchor,
    confidence, period, date, comments, comment_count,
    activation_count, created, last_active, updated_at, source, synced_at
  )
  values (
    memory_id, p_title, memory_type, p_domain, p_tags, p_content,
    greatest(0.0, least(1.0, p_valence)),
    greatest(0.0, least(1.0, p_arousal)),
    greatest(1.0, least(10.0, p_importance)),
    p_pinned,
    p_resolved, p_digested, p_anchor,
    greatest(0.0, least(1.0, p_confidence)),
    p_period,
    p_date,
    memory_comments,
    greatest(0, memory_comment_count),
    1, p_time, p_time, p_time, 'chatgpt', p_time
  )
  on conflict (id) do update set
    title = excluded.title,
    type = excluded.type,
    domain = excluded.domain,
    tags = excluded.tags,
    content = excluded.content,
    valence = excluded.valence,
    arousal = excluded.arousal,
    importance = excluded.importance,
    pinned = excluded.pinned,
    resolved = excluded.resolved,
    digested = excluded.digested,
    anchor = excluded.anchor,
    confidence = excluded.confidence,
    period = excluded.period,
    date = excluded.date,
    comments = excluded.comments,
    comment_count = excluded.comment_count,
    last_active = excluded.last_active,
    updated_at = now(),
    source = 'chatgpt',
    synced_at = excluded.synced_at
  returning * into result;

  return result;
end;
$$;

revoke all on function public.create_memory(
  text, text, text, text[], text[], text, double precision,
  double precision, double precision, boolean, boolean, boolean, timestamptz, boolean,
  double precision, text, text, jsonb, integer
) from public;

grant execute on function public.create_memory(
  text, text, text, text[], text[], text, double precision,
  double precision, double precision, boolean, boolean, boolean, timestamptz, boolean,
  double precision, text, text, jsonb, integer
) to authenticated, service_role;
