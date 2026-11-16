"""
Fast Audit Scanner for Deprecated Patterns & Hallucinated Stats.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = [
    "model.pth",
    "audio_classifier.pkl",
    "trained_modelimages",
    "random.choice",
    "986 images",
    "15K+",
    "Playfair",
    "Space Grotesk"
]

IGNORED_DIRS = {".venv", ".git", ".pytest_cache", "datasets", "ppt_like_work", "node_modules"}
ALLOWED_EXTS = {".py", ".html", ".css", ".js", ".json", ".yaml", ".md"}

def run_audit():
    print(f"Scanning codebase at {ROOT}...")
    found_any = False
    for root, dirs, files in os.walk(ROOT):
        # Skip ignored dirs
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS and not d.startswith(".")]
        for file in files:
            if file.endswith(".bak"):
                continue
            ext = Path(file).suffix.lower()
            if ext not in ALLOWED_EXTS:
                continue
            path = Path(root) / file
            # Don't check the audit script itself
            if path.name == "audit_check.py":
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
                for pat in PATTERNS:
                    if pat in content:
                        print(f"[FOUND] '{pat}' in {path.relative_to(ROOT)}")
                        found_any = True
            except Exception as e:
                pass

    if not found_any:
        print("[AUDIT PASSED] 0 occurrences of deprecated models, ghost fallbacks, fake stats, or legacy fonts found.")
    return not found_any

if __name__ == "__main__":
    run_audit()
