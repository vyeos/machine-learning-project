"""Package the project as a ZIP (excluding data/ CSVs, caches and venv).
Run:  python scripts/package_project.py
"""
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE_DIRS = {"data", ".venv", "venv", "__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".git"}
EXCLUDE_PARTS = {".joblib_memmapping_folder"}
ZIP_NAME = ROOT / f"Cybersecurity_Intrusion_Detection_{datetime.now():%Y%m%d}.zip"

EXCLUDE_EXTS = set()  # data/ CSVs are already excluded by directory


def wanted(path: Path) -> bool:
    rel_parts = path.relative_to(ROOT).parts
    if any(part in EXCLUDE_DIRS or part.endswith("_memmapping_folder") for part in rel_parts[:-1]):
        return False
    if rel_parts[0] == "data":
        return False
    if path.suffix.lower() in EXCLUDE_EXTS:
        return False
    if path.name == ZIP_NAME.name:
        return False
    if "__pycache__" in path.name:
        return False
    return True


def main():
    n = 0
    with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(ROOT.rglob("*")):
            if not path.is_file():
                continue
            if not wanted(path):
                continue
            z.write(path, path.relative_to(ROOT))
            n += 1
    print(f"Packaged {n} files -> {ZIP_NAME}")


if __name__ == "__main__":
    main()
