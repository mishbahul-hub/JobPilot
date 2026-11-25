from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.agent.service import close_agent, open_agent
from app.config import get_settings
from app.db.pool import apply_schema, close_pool, open_pool
from app.log import get_logger, setup_logging
from app.routers import chat, health, jobs, profile

settings = get_settings()
setup_logging(settings.log_level)
log = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build shared resources once, not per request.
    apply_schema(settings.database_url)
    open_pool(settings.database_url)
    if settings.google_api_key:
        open_agent(settings.database_url)
    else:
        log.warning("GOOGLE_API_KEY not set: /chat is disabled")
    log.info("startup complete")
    yield
    close_agent()
    close_pool()


app = FastAPI(
    title="JobPilot API",
    description="AI job-matching agent: hybrid search, tool-using agent with memory, evals.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception):
    # Log the details, never send them to the client.
    log.exception("unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"error": "internal server error"})


app.include_router(health.router)
app.include_router(jobs.router)
app.include_router(chat.router)
app.include_router(profile.router)
