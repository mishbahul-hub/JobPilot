"""Resume PDF -> structured profile. The file is read in memory and never written to disk;
only the extracted profile fields are stored (no name, email or phone)."""

import io

from pydantic import BaseModel, Field
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.llm import get_llm_pair

MAX_BYTES = 5 * 1024 * 1024
MAX_CHARS = 15_000


class ResumeError(ValueError):
    """The upload is not a readable PDF resume."""


class ResumeProfile(BaseModel):
    """What a job search needs from a resume. Leave out personal contact details."""

    skills: list[str] = Field(
        description="Technical skills, tools, languages, frameworks; short names like 'Python'"
    )
    titles: list[str] = Field(
        description="Job titles this person is a good fit for now, e.g. 'Backend Developer Intern'"
    )
    locations: list[str] = Field(
        default_factory=list, description="Cities they live in or say they prefer"
    )
    years_exp: float = Field(
        description="Years of full-time work experience; 0 for students (internships don't count)"
    )
    summary: str = Field(
        description="Two sentences: level (student/junior/...), strongest areas, notable projects"
    )


PROMPT = """Extract a job-search profile from this resume. Use only what the resume says.
Normalise skill names (e.g. 'ReactJS' -> 'React'), max 25 skills, max 5 titles.

Resume:
{text}"""


def extract_text(data: bytes) -> str:
    if len(data) > MAX_BYTES:
        raise ResumeError("file is larger than 5 MB")
    if not data.startswith(b"%PDF-"):
        raise ResumeError("file is not a PDF")
    try:
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join((page.extract_text() or "") for page in reader.pages[:10])
    except (PdfReadError, ValueError, KeyError) as exc:
        raise ResumeError("could not read the PDF") from exc
    text = " ".join(text.split())
    if len(text) < 100:
        raise ResumeError("no text found in the PDF (is it a scanned image?)")
    return text[:MAX_CHARS]


def parse_resume(text: str) -> ResumeProfile:
    main, *rest = [m.with_structured_output(ResumeProfile) for m in get_llm_pair()]
    chain = main.with_fallbacks(rest) if rest else main
    return chain.invoke(PROMPT.format(text=text))
