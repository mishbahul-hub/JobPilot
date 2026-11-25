import pytest

from app.profile import profile_text
from app.resume.parse import MAX_BYTES, ResumeError, extract_text
from app.search.personalize import rocchio


def test_rocchio_combines_and_normalises():
    v = rocchio([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    assert v[0] > v[1] > 0 > v[2]
    assert sum(x * x for x in v) == pytest.approx(1.0)


def test_rocchio_edge_cases():
    assert rocchio(None, None, None) is None
    assert rocchio(None, None, [1.0, 0.0]) is None  # only negatives: no direction to go
    assert rocchio(None, [0.0, 2.0], None) == pytest.approx([0.0, 1.0])


def test_profile_text_leaves_out_locations():
    text = profile_text({"titles": ["Backend Intern"], "skills": ["Python", "SQL"],
                         "years_exp": 0, "locations": ["Pune"], "summary": "Student."})
    assert "Backend Intern" in text and "Python, SQL" in text and "0 years" in text
    assert "Pune" not in text
    assert profile_text({}) == ""


def _pdf(text: str) -> bytes:
    """Minimal one-page PDF with a text line, built by hand (correct xref offsets)."""
    stream = f"BT /F1 10 Tf 20 700 Td ({text}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>"),
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for n, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return bytes(out)


def test_extract_text_reads_pdf():
    line = "Python FastAPI SQL developer with internship experience building REST APIs " * 2
    assert "FastAPI" in extract_text(_pdf(line))


@pytest.mark.parametrize("data, msg", [
    (b"hello, not a pdf", "not a PDF"),
    (b"%PDF-" + b"0" * MAX_BYTES, "larger than 5 MB"),
    (b"%PDF-1.4 garbage", "could not read"),
])
def test_extract_text_rejects_bad_files(data, msg):
    with pytest.raises(ResumeError, match=msg):
        extract_text(data)


def test_extract_text_rejects_empty_pdf():
    with pytest.raises(ResumeError, match="no text"):
        extract_text(_pdf("Hi"))
