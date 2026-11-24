"""Fetch the pinned upstream PointPillars code + KITTI weights into third_party/ (no compiling).

    python setup_pp.py              # git clone (cần git + mạng)
    python setup_pp.py --zip FILE   # máy không có git: dùng file zip GitHub của commit đã ghim
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = "https://github.com/zhulf0804/PointPillars"
COMMIT = "620e6b0d07e4cb37b7b0114f26b934e8be92a0ba"
DEST = Path(__file__).resolve().parent / "third_party" / "PointPillars"
WEIGHTS = DEST / "pretrained" / "epoch_160.pth"


def from_zip(path: Path) -> None:
    tmp = DEST.parent / "_unzip"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(path) as z:
        z.extractall(tmp)
    (inner,) = [p for p in tmp.iterdir() if p.is_dir()]
    shutil.rmtree(DEST, ignore_errors=True)
    inner.rename(DEST)
    shutil.rmtree(tmp)


def main() -> int:
    for _s in (sys.stdout, sys.stderr):
        if hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", help="đường dẫn zip đã tải sẵn")
    ap.add_argument("--download-zip", action="store_true", help="tải zip của commit qua HTTPS (không cần git)")
    args = ap.parse_args()
    if WEIGHTS.exists():
        print(f"Đã có {WEIGHTS}. Bỏ qua.")
        return 0
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if args.zip:
        from_zip(Path(args.zip))
    elif args.download_zip or not shutil.which("git"):
        target = DEST.parent / f"PointPillars-{COMMIT[:7]}.zip"
        print(f"Tải {REPO}/archive/{COMMIT}.zip ...")
        urllib.request.urlretrieve(f"{REPO}/archive/{COMMIT}.zip", target)
        from_zip(target)
        target.unlink()
    else:
        shutil.rmtree(DEST, ignore_errors=True)
        subprocess.run(["git", "clone", "--quiet", REPO, str(DEST)], check=True)
        subprocess.run(["git", "-C", str(DEST), "checkout", "--quiet", COMMIT], check=True)
    if not WEIGHTS.exists():
        print(f"Không thấy {WEIGHTS} sau khi tải.", file=sys.stderr)
        return 1
    print(f"Xong: {DEST} @ {COMMIT[:10]}, weights {WEIGHTS.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
