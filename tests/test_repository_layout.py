import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".json", ".bat"}
EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    "outputs",
    "runs",
    "third_party",
}


def test_required_entries_exist():
    required = [
        ROOT / "README.md",
        ROOT / "LICENSE",
        ROOT / ".gitignore",
        ROOT / "dimer" / "pipeline.py",
        ROOT / "monomer" / "pipelines" / "pipeline.py",
        ROOT / "multicodes" / "src" / "pipeline.py",
        ROOT / "multicodes" / "config" / "config.example.json",
    ]
    assert all(path.exists() for path in required)


def test_no_developer_absolute_paths():
    pattern = re.compile(r"(?i)\b[cdg]:[\\/](?:my projects|conda|windows)")
    offenders = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        if pattern.search(path.read_text(encoding="utf-8", errors="ignore")):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_no_real_api_key():
    pattern = re.compile(r"sk-[A-Za-z0-9_-]{20,}")
    offenders = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        if pattern.search(path.read_text(encoding="utf-8", errors="ignore")):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_uploaded_text_is_english():
    han_pattern = re.compile(r"[\u3400-\u9fff]")
    offenders = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        if han_pattern.search(path.read_text(encoding="utf-8", errors="ignore")):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []
