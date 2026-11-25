-- JobPilot schema. Idempotent: safe to run on every startup.

CREATE EXTENSION IF NOT EXISTS vector;

-- Users are anonymous at first (the frontend keeps a generated id); email is optional.
CREATE TABLE IF NOT EXISTS users (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    email       text UNIQUE,
    created_at  timestamptz NOT NULL DEFAULT now()
);

-- One profile per user, built from their resume and from what they tell the agent.
CREATE TABLE IF NOT EXISTS profiles (
    user_id     uuid PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    skills      text[] NOT NULL DEFAULT '{}',
    titles      text[] NOT NULL DEFAULT '{}',
    locations   text[] NOT NULL DEFAULT '{}',
    min_salary  integer,
    remote_ok   boolean,
    years_exp   real,
    summary     text,
    embedding   vector(768),
    updated_at  timestamptz NOT NULL DEFAULT now()
);

-- The jobs index. Rows are upserted by (source, external_id); content_hash decides
-- whether a job must be re-embedded; is_active/last_seen_at handle expiry.
CREATE TABLE IF NOT EXISTS jobs (
    id            bigserial PRIMARY KEY,
    source        text NOT NULL,
    external_id   text NOT NULL,
    title         text NOT NULL,
    company       text,
    location      text,
    remote        boolean NOT NULL DEFAULT false,
    salary_min    integer,
    salary_max    integer,
    description   text NOT NULL DEFAULT '',
    url           text,
    posted_at     timestamptz,
    content_hash  text NOT NULL,
    embedding     vector(768),
    tsv           tsvector GENERATED ALWAYS AS (
                      setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
                      setweight(to_tsvector('english', coalesce(company, '')), 'B') ||
                      setweight(to_tsvector('english', coalesce(description, '')), 'C')
                  ) STORED,
    is_active     boolean NOT NULL DEFAULT true,
    first_seen_at timestamptz NOT NULL DEFAULT now(),
    last_seen_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (source, external_id)
);

CREATE INDEX IF NOT EXISTS jobs_embedding_hnsw ON jobs USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS jobs_tsv_gin        ON jobs USING gin (tsv);
CREATE INDEX IF NOT EXISTS jobs_active_posted  ON jobs (is_active, posted_at DESC);
CREATE INDEX IF NOT EXISTS jobs_location       ON jobs (location);

-- Cache of query embeddings: repeated searches (and eval reruns) cost no embedding quota.
CREATE TABLE IF NOT EXISTS query_cache (
    query_key   text PRIMARY KEY,           -- model + normalised query text
    embedding   vector(768) NOT NULL,
    hits        integer NOT NULL DEFAULT 0,
    created_at  timestamptz NOT NULL DEFAULT now()
);

-- Chat history for analytics and the UI. (LangGraph keeps its own checkpoint tables for agent state.)
CREATE TABLE IF NOT EXISTS conversations (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title       text,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS messages (
    id               bigserial PRIMARY KEY,
    conversation_id  uuid NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role             text NOT NULL CHECK (role IN ('user', 'assistant', 'tool')),
    content          text NOT NULL,
    tool_calls       jsonb,
    created_at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS messages_conversation ON messages (conversation_id, created_at);

CREATE TABLE IF NOT EXISTS saved_jobs (
    user_id     uuid   NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    job_id      bigint NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    created_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, job_id)
);

CREATE TABLE IF NOT EXISTS feedback (
    id          bigserial PRIMARY KEY,
    user_id     uuid   NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    job_id      bigint REFERENCES jobs(id) ON DELETE SET NULL,
    message_id  bigint REFERENCES messages(id) ON DELETE SET NULL,
    rating      smallint NOT NULL CHECK (rating IN (-1, 1)),
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS feedback_user ON feedback (user_id);

CREATE TABLE IF NOT EXISTS eval_runs (
    id          bigserial PRIMARY KEY,
    kind        text  NOT NULL,           -- 'retrieval' | 'agent'
    config      jsonb NOT NULL DEFAULT '{}',
    metrics     jsonb NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);
