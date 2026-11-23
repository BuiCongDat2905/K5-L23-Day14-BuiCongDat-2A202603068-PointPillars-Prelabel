"""Run pre-trained PointPillars on VF frames and write pre-label boxes.

Examples::

    # buổi giảng: 6 frame chung
    python pp_prelabel.py --data <Data-VF> --frames demo/frames_lec.txt --out out/lec.json --plot out/lec_png

    # lab: frame tập A của bạn, cấu hình đã tune, so với ref_A.json bạn tự sửa trên CVAT
    python pp_prelabel.py --data <Data-VF> --frames d14_<ID>/frames_A.txt --config my_config.json \
        --gt d14_<ID>/ref_A.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore", message="torch.meshgrid")

import vf_d14  # noqa: E402
from vf_pp.core import VF_NAMES, Config, Detector, correction_cost  # noqa: E402


def load_config(path: str | None, overrides: list[str]) -> Config:
    cfg = Config.from_dict(json.loads(Path(path).read_text())) if path else Config()
    for item in overrides:
        key, _, value = item.partition("=")
        if key not in Config.__dataclass_fields__:
            raise SystemExit(f"--set {key}: không có tham số này. Xem vf_pp/core.py:Config.")
        current = getattr(cfg, key)
        if value.lower() in ("none", "null", ""):
            setattr(cfg, key, None)
        elif isinstance(current, str):
            setattr(cfg, key, value)
        elif isinstance(current, int) and not isinstance(current, bool):
            setattr(cfg, key, int(value))
        else:
            setattr(cfg, key, float(value))
    return cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="thư mục Data-VF (đã tải ở Day 13)")
    ap.add_argument("--frames", required=True, help="file danh sách frame hoặc s02f07,s03f15")
    ap.add_argument("--config", help="file JSON cấu hình (mặc định: cấu hình gốc của model)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="ghi đè 1 tham số, ví dụ --set score_thr=0.3 (lặp được)")
    ap.add_argument("--out", help="ghi box ra JSON (đưa vào vf_cvat.py prelabel)")
    ap.add_argument("--gt", help="ref_*.json hoặc thư mục nhãn để tính chi phí sửa")
    ap.add_argument("--plot", help="thư mục ghi ảnh 7 camera + BEV cho từng frame")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.set)
    data = Path(args.data)
    names = vf_d14.read_frame_list(args.frames)
    rows = vf_d14.frame_rows(data, names)
    det = Detector(cfg)
    gt = vf_d14.load_reference(args.gt, data, names) if args.gt else {}
    results, total = {}, Counter()
    print(f"{'frame':8} {'box':>4}  lớp")
    for row in rows:
        start = time.time()
        boxes = det.detect(vf_d14.load_points(row))
        results[row["name"]] = boxes
        classes = Counter(VF_NAMES[b["label"]] for b in boxes)
        warns = sum(1 for b in boxes if b["warn"])
        line = f"{row['name']:8} {len(boxes):4}  {dict(classes)}  cảnh báo kích thước={warns}  {time.time() - start:.1f}s"
        if row["name"] in gt:
            c = correction_cost(gt[row["name"]], boxes)
            total.update({k: v for k, v in c.items()})
            line += f"  | xóa {c['delete']} thêm {c['add']} sửa {c['edit']} → chi phí {c['cost']}"
        print(line)
    if total:
        cost = round(0.3 * total["delete"] + total["add"] + 0.5 * total["edit"], 2)
        print(f"TỔNG ({len(gt)} frame có nhãn): GT {total['gt']}, dự đoán {total['pred']}, khớp {total['match']}, "
              f"xóa {total['delete']}, thêm {total['add']}, sửa {total['edit']} → chi phí {cost}")
    if args.out:
        vf_d14.save_boxes(Path(args.out), results, {"config": cfg.to_dict(), "frames": names})
        print(f"Đã ghi {args.out}")
    if args.plot:
        import vf_qa
        out = Path(args.plot)
        out.mkdir(parents=True, exist_ok=True)
        calibs = {}
        for row in rows:
            if row["seq"] not in calibs:
                calibs[row["seq"]] = vf_qa.load_calib(data / f"sequence_{row['seq']:02d}")
            boxes = [dict(vf_d14.as_vf_box(b), idx=b["pp_id"]) for b in results[row["name"]]]
            sheet, _ = vf_qa.render_frame(row, boxes, calibs[row["seq"]])
            sheet.save(out / f"{row['name']}.jpg", quality=85)
        print(f"Đã vẽ {len(rows)} ảnh vào {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
