---
day: "D14"
title: "Day 14 Lab — Pre-label 3D bằng PointPillars pre-trained (VF LiDAR)"
description: "Chạy model PointPillars pre-trained trên KITTI bằng PyTorch CPU trên dữ liệu VF LiDAR của Day 13, đẩy cuboid pre-label lên CVAT 3D sửa tay làm chuẩn đối chứng trên tập A, commit dự đoán 5b trước khi quét tham số giảm công sửa, hoàn thiện 12 frame tập B và nộp link commit SHA repo GitHub cá nhân lên LMS."
outcomes:
  - "Cài đặt sạch môi trường PyTorch CPU (hoặc Docker fallback) không cần GPU/biên dịch CUDA, tải đúng weights KITTI ghim commit `620e6b0` bằng `scripts/setup_pp.py` và chạy qua `python scripts/pp_selftest.py --data <Data-VF>` với đầu ra `TẤT CẢ OK`."
  - "Chạy `scripts/pp_prelabel.py` trên 6 frame buổi giảng (`demo/frames_lec.txt`), giải thích được phép đổi trục VF ↔ KITTI, vùng phủ 4 lượt xoay (`passes=4`) và lý do mất cân bằng chi phí `0.3·xóa + 1.0·thêm + 0.5·sửa` trong bán kính `50 m`."
  - "Sinh bộ frame cá nhân theo `<ID>` và `<SEED>` qua `scripts/vf_cvat.py day14`, đẩy tập A (`3` frame) lên CVAT `D14-<ID>-A`, sửa tay bằng `Shape` (không bấm `Track`) và kéo về `d14_<ID>/ref_A.json` khớp số `Items`."
  - "Ghi mục 1 báo cáo `report/individual/<MSSV>_<HoTen>.md` (Bước 5b) rồi commit và push trước khi chạy `scripts/pp_sweep.py`, đối chiếu `sweep.csv`, `best_config.json`, leave-one-out và `scripts/pp_compare.py`, sửa đủ 12 frame tập B (`ref_B.json`) và nộp link commit SHA repo `K5-L23-DAY14-HoVaTen-MSSV-PointPillarsPrelabel` lên LMS trước 12:00 trưa hôm sau (GMT+7)."
prerequisites:
  - "Đã hoàn thành thao tác tạo Task 3D, xoay/chỉnh cuboid theo 3 khung Top/Side/Front và xuất nhãn trên CVAT ở Day 13."
  - "Có Python 3.9+, Git (hoặc chạy `scripts/setup_pp.py --download-zip`), bộ dữ liệu `<Data-VF>` (5 sequence từ Day 13) và mã học viên cá nhân."
  - "Nhận `<SEED>` từ giảng viên khi bắt đầu phần lab buổi chiều để sinh danh sách frame."
requiredTools:
  - "Python 3.9+ với `numpy>=1.24`, `torch>=2.1` (bản CPU), `Pillow>=10` và `cvat-sdk>=2.20` (`requirements.txt`)."
  - "CVAT 3D (`http://localhost:8080` khi chạy Docker local như Day 13 hoặc server CVAT của lớp)."
  - "Tùy chọn Docker (`Dockerfile` + `.dockerignore`) khi môi trường Python máy gặp sự cố."
commonErrors:
  - "PowerShell 5.1 báo lỗi khi gõ `&&`, `source`, hoặc chặn `.venv\\Scripts\\Activate.ps1` → Chạy từng dòng riêng, bật `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` và đặt `$env:PYTHONUTF8 = \"1\"` trước khi chạy."
  - "Nếu JSON không đọc được, tạo lại dự đoán bằng `scripts/pp_prelabel.py` hoặc xuất lại nhãn từ CVAT bằng `scripts/vf_cvat.py reference`."
  - "Lệnh `scripts/vf_cvat.py reference` dừng nếu còn Track. Vẽ lại các hộp đó bằng Shape, xóa Track, bấm Save rồi xuất lại. Nếu trùng tên task, kiểm các ID và đổi tên task bạn sẽ không xuất."
  - "Truyền `--org <SLUG>` khi chạy CVAT local Docker gây lỗi 404 không thấy tổ chức → Bỏ hẳn cờ `--org` trên CVAT local (`http://localhost:8080`); chỉ truyền `--org` thống nhất ở cả Bước 3, 5, 7 và 9 trên server lớp có organization."
  - "Docker báo `exit 137` (OOM) hoặc thoát container bị mất `d14_<ID>/` → Tắt CVAT Docker (`docker compose stop`) khi chạy model nếu WSL2 RAM <= 4 GB, và luôn mount `-v \"${PWD}\\work:/work\" -w /work`."
  - "`scripts/pp_sweep.py` báo `--grid ...: không có tham số này` → Luôn viết đủ dấu `=` dạng `--grid score_thr=0.1,0.2,0.3,0.4,0.5` và giữ mức gốc `0.1` trong lưới."
  - "Mất điểm mục 3 hoặc mục 4 dù đủ file → Phải commit/push mục 1 báo cáo (5b) *trước* `scripts/pp_sweep.py`, tự kiểm sửa 12 frame tập B để `ref_B.json` không trùng `pred_B.json`, và nộp đúng link commit SHA (`git rev-parse HEAD`) lên LMS trước 12:00 trưa hôm sau."
requiresSubmission: true
workMode: "individual"
---

# Day 14 Lab — Pre-label 3D bằng PointPillars pre-trained (VF LiDAR)

Bài cá nhân này có **245 phút thực hành**. Bạn dành 45 phút ở cuối buổi lý thuyết sáng (09:00–13:00), gồm 10 phút cài đặt và 35 phút chạy thử. Buổi chiều (16:00–19:30) có 200 phút làm tập A và B. Bạn chạy PointPillars đã huấn luyện trên KITTI trên dữ liệu LiDAR 360° của Day 13. Mô hình dùng CPU và mã PyTorch thay cho hai phép toán CUDA, nên máy không cần GPU. Mã PointPillars gốc giữ nguyên tại commit `620e6b0`.

Tạo cuboid dự đoán, đưa lên CVAT rồi sửa 3 frame tập A để có nhãn tham chiếu. Ghi dự đoán và push báo cáo ở bước 5b trước khi thử nhiều cấu hình. Sau đó dùng cấu hình đã chọn cho 12 frame tập B, sửa nhãn và xuất `ref_B.json`.

Đẩy bài lên repo `K5-L23-DAY14-HoVaTen-MSSV-PointPillarsPrelabel`. Nộp link commit có SHA lên LMS trước **12:00 trưa hôm sau (GMT+7)**. Xem [SUBMISSION.md](SUBMISSION.md) để biết file cần nộp và [RUBRIC.md](RUBRIC.md) để biết cách chấm. Theo dõi tiến độ bằng [CHECKPOINTS.md](CHECKPOINTS.md), đọc [RULES.md](RULES.md) và dùng [mẫu báo cáo cá nhân](report/individual/MSSV_HoTen.md).

## Pha 0 — Chuẩn bị máy, repository và kiểm cài đặt (Buổi lec sáng · 0:00 → 0:10 · 10 phút)

Ở cuối buổi lý thuyết sáng (trong khung `09:00–13:00`, ngay sau khi giảng xong phần lý thuyết, tùy tiến độ của giảng viên, ví dụ tượng trưng `~12:15 → 12:25`), hãy clone repository chính thức, dựng môi trường ảo Python 3.9+ và tải mã upstream PointPillars cùng weights KITTI (~20 MB). Hoàn tất bước này ngay trên lớp sáng giúp bạn kiểm tra CPU, PyTorch và đường dẫn `<Data-VF>` (`sequence_01 … sequence_05`) đều sẵn sàng cho Pha 1 và buổi chiều, đồng thời đọc trước [CHECKPOINTS.md](CHECKPOINTS.md) và [RULES.md](RULES.md).

:::repository{url="https://github.com/VinUni-AI20k/K5-L23-Day14-PointPillars-Prelabel" label="Repo gốc Day 14 — K5-L23-Day14-PointPillars-Prelabel"}
- [Repository gốc Day 14](https://github.com/VinUni-AI20k/K5-L23-Day14-PointPillars-Prelabel)
:::

Script `scripts/setup_pp.py` tải mã upstream đúng commit `620e6b0` và weights `epoch_160.pth` mà không biên dịch CUDA nhờ lớp PyTorch thuần trong `vf_pp/upstream.py`; nếu máy không có `git`, dùng `--download-zip` để tải qua HTTPS. Chọn tab theo hệ điều hành để cài PyTorch CPU (`torch>=2.1`), `requirements.txt` (`numpy`, `Pillow`, `cvat-sdk`), rồi chạy `scripts/setup_pp.py` và `scripts/pp_selftest.py`. Trên PowerShell 5.1, không dùng `&&` hay `source`; chạy từng dòng, cấp quyền tiến trình cho `.venv\Scripts\Activate.ps1` và đặt `$env:PYTHONUTF8 = "1"`.

:::code-tabs{title="Cài đặt môi trường Python CPU và tự kiểm tra cài đặt"}
```powershell {title="Windows (PowerShell)"}
git clone https://github.com/VinUni-AI20k/K5-L23-Day14-PointPillars-Prelabel.git k5-day14-lab
cd k5-day14-lab
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python scripts/setup_pp.py
python scripts/pp_selftest.py --data "<Data-VF>"
```
```bash {title="macOS / Linux (Bash)"}
git clone https://github.com/VinUni-AI20k/K5-L23-Day14-PointPillars-Prelabel.git k5-day14-lab
cd k5-day14-lab
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python scripts/setup_pp.py
python scripts/pp_selftest.py --data "<Data-VF>"
```
:::

1. Chạy các lệnh trên; nếu bản clone cũ bị lỗi read-only trong `third_party/PointPillars/.git`, xóa `third_party` rồi chạy lại `python scripts/setup_pp.py`.
2. Chạy `python scripts/pp_selftest.py --data "<Data-VF>"` và kiểm tra mọi dòng đều in `[OK ]` kèm `TẤT CẢ OK`.
3. Nếu môi trường máy gặp lỗi, dùng `Dockerfile` (tắt CVAT bằng `docker compose stop` nếu WSL2 RAM `<= 4 GB` tránh OOM `exit 137`, gắn `-v "${PWD}\work:/work" -w /work` giữ `d14_<ID>/`, và dùng `--host http://host.docker.internal:8080`).

:::callout{tone="info" title="Chạy qua Docker khi môi trường Python cục bộ lỗi"}
Build một lần bằng `docker build -t k5-day14-lab .`, sau đó chạy `docker run --rm -v "C:\path\Data-VF:/data" -v "${PWD}\work:/work" -w /work k5-day14-lab python /lab/scripts/pp_selftest.py --data /data` và kiểm tra `TẤT CẢ OK`.
:::

:::checklist{title="Tín hiệu sẵn sàng trước Pha 1" tone="info"}
- [ ] Đã clone đúng repo và kích hoạt `.venv` có Python 3.9+.
- [ ] `scripts/setup_pp.py` tải xong `third_party/PointPillars` (`620e6b0`) và `epoch_160.pth`.
- [ ] `scripts/pp_selftest.py --data "<Data-VF>"` in `[OK ]` ở mọi mục và `TẤT CẢ OK`.
- [ ] CVAT 3D (`http://localhost:8080` hoặc server lớp) đăng nhập bình thường.
:::

## Pha 1 — Thực hành 6 frame buổi sáng, hệ trục VF ↔ KITTI và chi phí sửa (0:10 → 0:45 · 35 phút)

### Bản chất vấn đề: vì sao model KITTI không dùng thẳng được trên VF LiDAR?

Một mô hình phát hiện 3D được huấn luyện trên bộ dữ liệu KITTI không thể dự đoán chính xác ngay lập tức khi đưa sang bộ dữ liệu VF của Day 13 vì sự khác biệt cả về hệ tọa độ cảm biến, vùng quét không gian lẫn danh sách lớp đối tượng. Trên bộ KITTI, cảm biến chỉ gán nhãn trong vùng hình nón phía trước xe (`x` từ `0` đến `69.12 m`, `|y| < 39.68 m`) và chỉ phân biệt ba lớp là `Car`, `Cyclist` và `Pedestrian`. Trong khi đó, đám mây điểm VF LiDAR thu thập toàn cảnh 360° quanh xe với hệ trục đặt khác hẳn: trục `x` hướng sang trái xe, trục `y` hướng về phía sau (nên đầu xe là hướng `−y`), trục `z` hướng lên, còn góc `yaw` trong `[0, 2π)` đo ngược chiều kim đồng hồ từ `+x`. Đồng thời, dữ liệu VF có thêm hai lớp phương tiện cỡ lớn là `Truck` và `Bus` mà trọng số KITTI chưa từng được học trực tiếp.

:::image{src="walkthrough_images/diagram_01_vf_axes_kitti.jpg" alt="Sơ đồ đối chiếu hệ trục LiDAR VF và khung tọa độ KITTI của PointPillars" caption="Phép đổi trục trong vf_pp/core.py đưa đám mây điểm VF về hướng nhìn của model KITTI."}
:::

### Giải nghĩa vùng phủ 4 lượt xoay và công thức chi phí sửa

Vì model chỉ nhìn vùng 90° phía trước (`x` trong `0–69.12 m`, `|y| < 39.68 m`), hàm `Detector.raw_detect` trong `vf_pp/core.py` thực hiện **4 lượt xoay tuần tự (`passes = 4`)** ở `0°`, `90°`, `180°`, `270°` (bỏ qua an toàn lượt không có điểm), rồi gộp hộp bằng NMS không phân lớp (`merge_nms_thr = 0.1`). Vòng tròn chấm điểm bán kính `50.0 m` (`MAX_RANGE = 50.0`, cố định bất kể tham số `range_m`) nằm gọn bên trong hợp bốn khung xoay này.

:::image{src="walkthrough_images/diagram_02_model_range_50m.jpg" alt="Sơ đồ vùng nhìn BEV bốn lượt xoay bao trọn vòng tròn đỏ 50 m" caption="Hợp 4 lượt xoay phủ kín vòng tròn chấm điểm 50 m; giảm passes=1 sẽ sót hai bên xe."}
:::

Chất lượng pre-label được đo trong bán kính `50.0 m` bằng **`correction_cost = 0.3·xóa + 1.0·thêm + 0.5·sửa`** sau khi ghép 1–1 theo Hungarian (`BEV IoU >= 0.3`). Hàm `needs_edit` kiểm tra `yaw` theo chu kỳ `π`, nên box ngược đầu đúng 180° không tính là "sửa" nếu lớp, tâm và cạnh đã đạt.

| Thao tác (trọng số) | Điều kiện tính trong `correction_cost` (`MAX_RANGE = 50.0 m`) | Tác động khi sửa trên CVAT |
| --- | --- | --- |
| **Xóa (`0.3`)** | Box dự đoán dư không ghép được cặp với `gt` (`BEV IoU < 0.3`) | Bấm `Remove` hoặc `Delete` rất nhanh nên phạt nhẹ nhất |
| **Thêm (`1.0`)** | Box thật trong `gt` bị model bỏ sót hoàn toàn | Phải vẽ cuboid mới và kéo lại 3 mặt cắt nên phạt nặng nhất |
| **Sửa (`0.5`)** | Cặp khớp (`BEV IoU >= 0.3`) nhưng sai lớp, lệch tâm `> 0.3 m`, lệch cạnh `> 0.2 m`, hoặc lệch `yaw > 10°` (mod `π`) | Kéo khung `Top/Side/Front` hoặc gõ `L/W/H`, đổi nhãn trên `DETAILS` |

### Lựa chọn thiết kế: ba nhóm tham số cấu hình trong `vf_pp/core.py:Config`

Các tham số trong `vf_pp/core.py:Config` được tách thành ba nhóm để bạn can thiệp đúng tầng khi tinh chỉnh. Nhóm **A** đưa điểm VF về lưới KITTI: luôn giữ `voxel_size = 0.16` khớp weights đã huấn luyện và giữ `passes = 4` để phủ đủ `360°`, vì thay đổi hai tham số này làm tụt chất lượng ngay lập tức. Nhóm **B** đặt ngưỡng tin cậy chung (`score_thr`) hoặc theo lớp (`thr_car`, `thr_cyc`, `thr_ped`) cùng NMS nội bộ. Nhóm **C** hậu xử lý trên không gian VF: lọc `min_points`, nới hộp `size_pad_m` và đổi `Car` cỡ lớn thành `Truck`/`Bus` qua `vf_class`. Khi thử nhanh, dùng cờ `--set KEY=VALUE`; khi chốt cấu hình ổn định, lưu file JSON và truyền qua `--config`.

| Nhóm | Tham số chính (mặc định) | Ý nghĩa và phạm vi tác động |
| --- | --- | --- |
| A. Đầu vào model | `voxel_size` (`0.16`), `intensity_mode` (`clip`/`log`/`const`), `intensity_scale` (`1.0`), `ground_mode` (`fixed`/`auto`), `delta_z` (`0.0`), `passes` (`4`) | Quy đổi điểm VF về miền KITTI; đổi `voxel_size` khỏi `0.16` làm sai lệch lưới mà weights đã học, giảm `passes` làm mất vùng nhìn |
| B. Đầu ra model | `score_thr` (`0.1`), `thr_car`/`thr_cyc`/`thr_ped` (`None`), `nms_thr` (`0.01`), `nms_pre` (`100`), `max_num` (`50`) | Lọc box theo độ tin cậy chung hoặc riêng từng lớp và đặt ngưỡng NMS bên trong mạng |
| C. Hậu xử lý VF | `range_m` (`50.0`), `min_points` (`0`), `merge_nms_thr` (`0.1`), `size_pad_m` (`0.1`), `truck_l_min` (`5.6`), `truck_h_min` (`2.4`), `bus_l_min` (`9.0`), `bus_h_min` (`2.8`) | Cắt bán kính giữ box, lọc box quá ít điểm LiDAR (`min_points`), nới kích thước (`size_pad_m`) và đổi nhãn `Car` dài/cao thành `Truck` hoặc `Bus` theo luật kích thước |

### Các bước thực hiện Pha 1

Bạn chạy thử model trên 6 frame chung của buổi giảng trong `demo/frames_lec.txt` (`s02f07`, `s03f15`, `s05f40`, `s01f39`, `s03f18`, `s05f09`) theo cấu hình gốc, rồi thử ghi đè hai tham số `--set score_thr=0.4 --set min_points=5` để quan sát trực tiếp hiệu ứng đánh đổi giữa xóa box thừa và làm rơi box thật. Trên 6 frame này, với các box khớp nhãn, trung vị sai lệch là `13 cm` ở tâm, `2.3°` ở góc yaw và dưới `6 cm` ở kích thước.

:::code-tabs{title="Chạy pre-label 6 frame buổi giảng với cấu hình gốc và cấu hình thử ngưỡng"}
```powershell {title="Windows (PowerShell)"}
python scripts/pp_prelabel.py --data "<Data-VF>" --frames demo/frames_lec.txt --out out/lec.json --plot out/lec_png
python scripts/pp_prelabel.py --data "<Data-VF>" --frames demo/frames_lec.txt --set score_thr=0.4 --set min_points=5
```
```bash {title="macOS / Linux (Bash)"}
python scripts/pp_prelabel.py --data "<Data-VF>" --frames demo/frames_lec.txt --out out/lec.json --plot out/lec_png
python scripts/pp_prelabel.py --data "<Data-VF>" --frames demo/frames_lec.txt --set score_thr=0.4 --set min_points=5
```
:::

1. Chạy lệnh đầu tiên để xuất danh sách box `out/lec.json` và vẽ 6 ảnh tổng hợp `out/lec_png/*.jpg` (mỗi ảnh gồm 7 góc camera và bản đồ BEV bán kính `50 m`).
2. Mở ảnh `out/lec_png/s05f40.jpg` cùng các frame còn lại để đối chiếu box trên camera với vị trí vòng tròn BEV: ghi nhận model nhận xe con gần khá ổn, nhưng dễ gán nhầm xe buýt lớn thành nhiều hộp nhỏ, bắn nhầm hộp giả vào hàng cây/bụi rậm, hoặc bỏ sót `Pedestrian` và `Cyclist` ở xa.

:::image{src="walkthrough_images/cli_03_lec_prelabel_s05f40.jpg" alt="Ảnh 7 camera và BEV 50 m của frame s05f40 với 35 cuboid pre-label" caption="Ảnh out/lec_png/s05f40.jpg hiển thị 7 camera và BEV vòng đỏ 50 m kèm mã pp_id."}
:::

3. Chạy lệnh thứ hai với `--set score_thr=0.4 --set min_points=5` và so sánh số lượng box in trên terminal (chẳng hạn ở `s05f40` giảm từ `35` box xuống còn `20` box): quan sát xem những box biến mất là nhiễu trong lùm cây/hàng rào hay là phương tiện thật ở vành ngoài.

### Checkpoint Pha 1 và chuyển sang buổi chiều

Pha 1 hoàn thành khi bạn đọc được cả hai đầu ra terminal của 6 frame buổi sáng, mở đủ 6 ảnh `out/lec_png/*.jpg` và giải thích được vì sao hệ số phạt `1.0` cho mỗi box bị sót lớn hơn gấp ba lần hệ số `0.3` cho mỗi box thừa. Kết thúc **45 phút** thực hành trong buổi lý thuyết sáng (`09:00–13:00`, gồm Pha 0 `10′` + Pha 1 `35′`), bạn sẵn sàng bước vào buổi thực hành chiều (`16:00–19:30`).

## Pha 2 — Lab Buổi chiều Tập A (Bước 1–5): Chọn frame theo SEED, pre-label và sửa tay trên CVAT (0:45 → 1:45 · 60 phút)

### Bản chất vấn đề: tách tập tune (A) và tập nộp (B) bằng mã học viên + SEED

Ngay khi phần thực hành buổi chiều (`16:00–19:30`) bắt đầu, giảng viên sẽ công bố một chuỗi `<SEED>` chung cho buổi học. Lệnh `scripts/vf_cvat.py day14` kết hợp `<SEED>` với mã học viên `<ID>` qua hàm băm SHA-256 (`d14:<SEED>:<ID>`) để sinh hai danh sách frame hoàn toàn riêng cho từng người: **tập A (`frames_A.txt`) gồm 3 frame** thuộc 3 sequence khác nhau để tinh chỉnh tham số, và **tập B (`frames_B.txt`) gồm 12 frame** trải trên 4 sequence (3 frame mỗi sequence, cách nhau tối thiểu 15 frame) dùng cho bài nộp cuối. Các frame được tự động lọc tránh toàn bộ 6 frame buổi giảng, tránh 10 frame Day 13 của chính bạn, đồng thời mỗi frame tập A luôn cách mọi frame tập B cùng sequence ít nhất 6 chỉ số để tránh rò rỉ dữ liệu khi tinh chỉnh.

### Giải nghĩa quy tắc mã học viên có số 0 đầu, BOM và cờ `--org`

Giữ đúng ba quy ước kỹ thuật sau trước khi chạy lệnh ở Pha 2:
- **Giữ số `0` đầu mã `<ID>`:** `vf_common.normalize_student` chỉ viết hoa chữ, nên `"001"` và `"01"` tạo hai bộ frame và hai thư mục (`d14_001/` vs `d14_01/`) khác nhau; trên PowerShell luôn bọc `--student "001"`. Không sửa tay `frames_A.txt` hay `frames_B.txt`.
- **Giữ nguyên file do script tạo.** Script đọc được BOM UTF-8 trong `frames_*.txt`, cấu hình, `pred_*.json` và `ref_*.json`. Nếu file JSON bị hỏng, tạo lại dự đoán bằng `scripts/pp_prelabel.py` hoặc xuất lại nhãn từ CVAT bằng `scripts/vf_cvat.py reference`.
- **Cờ `--org <SLUG>` là tùy chọn:** bỏ hẳn `--org` khi chạy CVAT Docker cá nhân (`http://localhost:8080`, tránh lỗi 404); chỉ truyền thống nhất ở Bước 3, 5, 7 và 9 khi server lớp có organization.

### Lựa chọn thiết kế: chỉnh sửa trên pre-label thay vì vẽ lại từ đầu và bẫy `Shape` vs `Track`

Ở bước 4, hãy dành khoảng 45 phút để kiểm các hộp dự đoán trên CVAT và chỉnh theo hướng dẫn Day 13. Giữ hộp đúng, sửa hộp sai lớp hoặc lệch vị trí, xóa hộp thừa và thêm đối tượng bị sót. Rubric mục 1 chấm chất lượng nhãn tham chiếu tập A, gồm cả các đối tượng mô hình bỏ sót. Hộp có sẵn các thuộc tính `pp_id`, `score`, `warn` và `note` để đối chiếu với ảnh BEV. Khi thêm hộp mới, chọn **Shape**. Lệnh `scripts/vf_cvat.py reference` dừng nếu còn hộp dạng **Track**. Nếu lệnh báo trùng tên task, kiểm các ID trên CVAT và đổi tên task bạn sẽ không xuất.

:::callout{tone="warning" title="Cảnh báo bắt buộc: luôn chọn Shape (KHÔNG chọn Track) và đối chiếu Items"}
Nếu lệnh xuất báo có **Track**, vẽ lại các hộp đó bằng **Shape** rồi xóa Track. Bấm **Save** và chạy lại `reference`. Đối chiếu số hộp in ra cho từng frame với **Items** trên CVAT.
:::

### Các bước thực hiện Bước 1–5 (Tập A)

Chạy lần lượt Bước 1 (sinh frame), Bước 2 (pre-label + ảnh BEV), Bước 3 (đẩy `D14-<ID>-A`) và Bước 5 (kéo `ref_A.json`) sau khi hoàn tất Bước 4. Mật khẩu CVAT nhập qua `CVAT_PASSWORD` hoặc gõ tại dấu nhắc, không ghi ra file.

:::code-tabs{title="Thực hiện Bước 1, 2, 3 và 5 cho Tập A (thay mã học viên, SEED và đường dẫn dữ liệu)"}
```powershell {title="Windows (PowerShell)"}
python scripts/vf_cvat.py day14 --student "<ID>" --seed "<SEED>" --data "<Data-VF>"
python scripts/pp_prelabel.py --data "<Data-VF>" --frames "d14_<ID>/frames_A.txt" --out "d14_<ID>/pred_A.json" --plot "d14_<ID>/png_A"
python scripts/vf_cvat.py prelabel --student "<ID>" --task A --data "<Data-VF>" --boxes "d14_<ID>/pred_A.json" --host http://localhost:8080 --user "<ten>"
# ... Sửa tay tập A trên CVAT 3D (Bước 4) và bấm Save ...
python scripts/vf_cvat.py reference --student "<ID>" --task A --host http://localhost:8080 --user "<ten>"
python scripts/pp_prelabel.py --data "<Data-VF>" --frames "d14_<ID>/frames_A.txt" --gt "d14_<ID>/ref_A.json"
```
```bash {title="macOS / Linux (Bash)"}
python scripts/vf_cvat.py day14 --student "<ID>" --seed "<SEED>" --data "<Data-VF>"
python scripts/pp_prelabel.py --data "<Data-VF>" --frames "d14_<ID>/frames_A.txt" --out "d14_<ID>/pred_A.json" --plot "d14_<ID>/png_A"
python scripts/vf_cvat.py prelabel --student "<ID>" --task A --data "<Data-VF>" --boxes "d14_<ID>/pred_A.json" --host http://localhost:8080 --user "<ten>"
# ... Sửa tay tập A trên CVAT 3D (Bước 4) và bấm Save ...
python scripts/vf_cvat.py reference --student "<ID>" --task A --host http://localhost:8080 --user "<ten>"
python scripts/pp_prelabel.py --data "<Data-VF>" --frames "d14_<ID>/frames_A.txt" --gt "d14_<ID>/ref_A.json"
```
:::

1. **Bước 1 — Sinh danh sách frame:** Chạy `scripts/vf_cvat.py day14` tạo `d14_<ID>/frames_A.txt` và `frames_B.txt`; kiểm tra dòng `seed_tag=` khớp hạt giống buổi học.
2. **Bước 2 — Pre-label Tập A:** Chạy `scripts/pp_prelabel.py` tạo `pred_A.json` và 3 ảnh `d14_<ID>/png_A/*.jpg` để xem trước vùng thừa/sót.

:::image{src="walkthrough_images/cli_01_prelabel_7cam_bev_s02f45.jpg" alt="Ảnh pre-label tập A trên s02f45 gồm 7 camera và BEV 50 m" caption="Ảnh d14_001/png_A/s02f45.jpg ở Bước 2 giúp rà nhanh 7 camera và BEV trước khi lên CVAT."}
:::

3. **Bước 3 — Đẩy task lên CVAT:** Chạy `scripts/vf_cvat.py prelabel --task A` (bỏ `--org` trên Docker local). Mở trang **Tasks**, vào `D14-<ID>-A` kiểm tra đủ 5 nhãn (`Car`, `Truck`, `Bus`, `Cyclist`, `Pedestrian`) và đúng 3 frame.

:::image{src="walkthrough_images/cvat_01_tasks_list_A.png" alt="Trang Tasks trên CVAT hiển thị task D14-001-A đã được tạo" caption="Trang Tasks sau Bước 3: task D14-001-A xuất hiện kèm trạng thái 1 annotating."}
:::

:::image{src="walkthrough_images/cvat_02_task_details_A.png" alt="Chi tiết Task D14-001-A với 5 nhãn cuboid và 1 Job 3 frame" caption="Trang chi tiết Task: 5 nhãn cuboid và Job gồm đúng 3 frame tập A."}
:::

4. **Bước 4 — Sửa tay 3 frame trên CVAT 3D (~45 phút):** Mở Job ở `Standard 3D` và thực hiện đủ bốn thao tác chuẩn hóa:
   - **Kiểm tra thuộc tính:** Bảng phải hiện camera, cây `Objects` (`Items`) và `DETAILS` (`L/W/H`, `pp_id`, `score`, `warn`, `note`).

:::image{src="walkthrough_images/cvat_03_3d_editor_initial.png" alt="Không gian CVAT 3D trên s02f45.pcd với 46 cuboid và 3 mặt cắt" caption="Giao diện CVAT 3D ở Bước 4: khung 3D trên cùng và 3 mặt cắt Top/Side/Front bên dưới."}
:::

:::image{src="walkthrough_images/cvat_04_object_details_attrs.png" alt="Thẻ đối tượng Car mở DETAILS hiển thị L W H và 4 thuộc tính" caption="Mục DETAILS hiển thị L/W/H cùng bốn thuộc tính pp_id, score, warn, note do tool tạo sẵn."}
:::

   - **Sửa box lệch/sai lớp:** Kéo các điểm đỏ ở `Top / Side / Front` hoặc gõ `L/W/H`, rồi đổi nhãn đúng (`Car` → `Truck`/`Bus`).

:::image{src="walkthrough_images/cvat_05_edit_length_4_23.png" alt="Chỉnh chiều dài L của cuboid Car từ 3.78 thành 4.23 trong DETAILS" caption="Sửa kích thước hộp bằng cách gõ số vào L/W/H hoặc kéo các điểm điều chỉnh trên khung cắt."}
:::

   - **Xóa box thừa:** Bấm `⋮ → Remove` trên thẻ đối tượng (hoặc `Delete`), kiểm tra `Items` giảm đi 1.

:::image{src="walkthrough_images/cvat_06_delete_menu_obj2.png" alt="Mở menu ba chấm trên đối tượng số 2 chọn Remove xóa hộp thừa" caption="Bấm nút ba chấm ở góc thẻ đối tượng và chọn Remove để xóa hộp dự đoán sai."}
:::

:::image{src="walkthrough_images/cvat_07_after_delete_45_items.png" alt="Bảng Objects sau khi xóa hiển thị chỉ số Items giảm từ 46 xuống 45" caption="Sau khi xóa một box thừa trên s02f45, chỉ số Items giảm ngay từ 46 xuống 45."}
:::

   - **Thêm box sót bằng `Shape` và Save:** Chọn icon lập phương, bấm **`Shape`** (KHÔNG `Track`), **double-click** khung 3D, căn 3 mặt cắt `Top / Side / Front`, rồi bấm **`Save`** tới khi nút mờ đi.

:::image{src="walkthrough_images/cvat_08_draw_cuboid_shape_vs_track.png" alt="Popup Draw new cuboid hiển thị hai nút Shape và Track cạnh nhau" caption="Khi vẽ hộp mới, luôn chọn Shape (trái) và tránh Track (phải) để reference không bỏ sót."}
:::

:::image{src="walkthrough_images/cvat_09_add_bus_shape_46_items.png" alt="Thêm thành công đối tượng #101 Bus dạng CUBOID SHAPE đạt 46 Items" caption="Đối tượng #101 nhãn Bus tạo bằng CUBOID SHAPE làm tổng Items tăng trở lại 46."}
:::

:::image{src="walkthrough_images/cvat_10_saved_clean_46_items.png" alt="Màn hình CVAT sau khi bấm Save với đủ 46 cuboid dạng Shape" caption="Bấm Save tới khi nút Save mờ đi trước khi chạy lệnh kéo nhãn tham chiếu về máy."}
:::

5. **Bước 5.** Chạy `scripts/vf_cvat.py reference --task A` để xuất `d14_<ID>/ref_A.json`. Đối chiếu số hộp với **Items** trên CVAT. Nếu server đảo chiều dài và chiều rộng, thêm `--scale-order wlh`. Sau đó dùng `scripts/pp_prelabel.py` với `--gt d14_<ID>/ref_A.json` để đo chi phí của cấu hình gốc.

### Checkpoint Pha 2 và giờ nghỉ giải lao (1:45 → 1:55 · 10 phút)

Trước khi nghỉ, kiểm tra `d14_<ID>/` có `frames_A.txt`, `frames_B.txt`, `pred_A.json` và `ref_A.json`. Kết quả xuất phải khớp số **Items** trên từng frame CVAT. Đọc số hộp cần xóa, thêm và sửa do `scripts/pp_prelabel.py` in ra để biết lỗi nào tốn nhiều công nhất với cấu hình gốc. Bạn sẽ dùng các số này và ảnh BEV để viết dự đoán. Trong **10 phút nghỉ (1:45 → 1:55)**, sao chép `report/individual/MSSV_HoTen.md` thành `report/individual/<MSSV>_<HoTen>.md` (không dấu). Bạn sẽ điền và push mục 1 ở bước 5b.

## Pha 3 — Bước 5b, Sweep và So sánh trước/sau trên Tập A (1:55 → 2:25 · 30 phút)

### Bản chất vấn đề: commit dự đoán trước khi sweep và rủi ro overfit trên 3 frame

Ở bước 5b, viết hai dòng vào mục 1 báo cáo cá nhân. Dòng đầu nêu tham số bạn dự đoán sẽ giảm công sửa nhiều nhất. Dòng thứ hai giải thích dựa trên ảnh BEV và số hộp cần xóa, thêm hoặc sửa ở tập A. Dùng `git add` để đưa báo cáo vào commit `5b: dự đoán`. Push commit lên repo cá nhân `K5-L23-DAY14-HoVaTen-MSSV-PointPillarsPrelabel` và mở GitHub kiểm tra trước khi chạy `scripts/pp_sweep.py`. Sau sweep, giữ nguyên dự đoán ở mục 1 và ghi phần đối chiếu vào mục 3. Theo [RUBRIC.md](RUBRIC.md), dự đoán sai vẫn có thể được đủ 20 điểm nếu bạn giải thích đúng kết quả.

### Giải nghĩa cơ chế cache suy luận và kiểm tra chéo leave-one-out

`scripts/pp_sweep.py` lưu dự đoán thô trong bộ nhớ theo từng tổ hợp tham số trong `MODEL_KNOBS`. Với mỗi tổ hợp, mô hình chạy một lần trên từng frame với ngưỡng `score_thr = 0.05`. Khi chỉ đổi ngưỡng điểm hoặc các tham số hậu xử lý như `min_points` và `range_m`, script dùng lại dự đoán thô để khỏi chạy mô hình lần nữa.

Cuối lượt quét, đọc mục **leave-one-out**. Script lần lượt giữ một frame để kiểm và dùng hai frame còn lại để chọn cấu hình. Nó so chi phí của cấu hình đó với chi phí thấp nhất có thể trên frame kiểm, gọi là `oracle`. Nếu chênh lệch lớn, cấu hình có thể chỉ phù hợp với các frame dùng để chọn. Ghi kết quả từng lượt và nhận xét vào mục 2 báo cáo.

### Lựa chọn thiết kế: giữ mốc gốc `0.1` trong lưới và đọc ảnh BEV `52 m`

Khi truyền `--grid` cho `scripts/pp_sweep.py`, giữ hai quy tắc vận hành quan trọng: thứ nhất, luôn viết đủ dấu `=` dạng `key=vals` (bỏ thiếu `=` làm parser dừng và báo lỗi `--grid ...: không có tham số này`); thứ hai, luôn giữ mức mặc định `score_thr=0.1` cùng `min_points=0` trong lưới làm đối chứng để thuật toán không ép chọn `0.2` khi chi phí tăng. Ở Bước 8, `scripts/pp_compare.py` vẽ hai khung BEV cạnh nhau (`gốc` bên trái, `của bạn` bên phải) trong phạm vi **`±52 m`** kèm vòng đỏ **`50 m`** đúng phạm vi chấm `correction_cost`: **xanh lá** (giữ nguyên), **vàng** (cần sửa), **đỏ** (thừa, cần xóa) và **xanh dương** (sót, cần vẽ thêm).

### Các bước thực hiện Bước 5b, 6 và 8

Đảm bảo `origin` đã trỏ về repo nộp cá nhân của bạn (`K5-L23-DAY14-HoVaTen-MSSV-PointPillarsPrelabel`) trước khi push commit Bước 5b, rồi mới chạy `scripts/pp_sweep.py` và `scripts/pp_compare.py`.

:::code-tabs{title="Bước 5b (commit trước sweep), Bước 6 (sweep) và Bước 8 (compare)"}
```powershell {title="Windows (PowerShell)"}
# Bước 5b: Điền Mục 1 trong report/individual/<MSSV>_<HoTen>.md rồi commit & push TRƯỚC sweep
git remote set-url origin "https://github.com/<tai-khoan>/K5-L23-DAY14-HoVaTen-MSSV-PointPillarsPrelabel.git"
git add "report/individual/<MSSV>_<HoTen>.md"
git commit -m "5b: dự đoán"
git branch -M main
git push -u origin main
python scripts/pp_sweep.py --data "<Data-VF>" --frames "d14_<ID>/frames_A.txt" --gt "d14_<ID>/ref_A.json" --grid score_thr=0.1,0.2,0.3,0.4,0.5 --grid min_points=0,5,10 --out "d14_<ID>/sweep.csv"
python scripts/pp_compare.py --data "<Data-VF>" --frames "d14_<ID>/frames_A.txt" --gt "d14_<ID>/ref_A.json" --tuned "d14_<ID>/best_config.json" --out "d14_<ID>/compare"
```
```bash {title="macOS / Linux (Bash)"}
# Bước 5b: Điền Mục 1 trong report/individual/<MSSV>_<HoTen>.md rồi commit & push TRƯỚC sweep
git remote set-url origin "https://github.com/<tài-khoản>/K5-L23-DAY14-HoVaTen-MSSV-PointPillarsPrelabel.git"
git add "report/individual/<MSSV>_<HoTen>.md"
git commit -m "5b: dự đoán"
git branch -M main
git push -u origin main
python scripts/pp_sweep.py --data "<Data-VF>" --frames "d14_<ID>/frames_A.txt" --gt "d14_<ID>/ref_A.json" --grid score_thr=0.1,0.2,0.3,0.4,0.5 --grid min_points=0,5,10 --out "d14_<ID>/sweep.csv"
python scripts/pp_compare.py --data "<Data-VF>" --frames "d14_<ID>/frames_A.txt" --gt "d14_<ID>/ref_A.json" --tuned "d14_<ID>/best_config.json" --out "d14_<ID>/compare"
```
:::

1. **Bước 5b:** Điền mục 1 (mã `<ID>`, `seed_tag` từ `frames_A.txt` và 2 dòng dự đoán dựa trên ảnh BEV `png_A/` cùng baseline Bước 5) vào `report/individual/<MSSV>_<HoTen>.md`, rồi chạy `git commit -m "5b: dự đoán"` và `git push` trước khi chạy sweep; tuyệt đối không sửa lại mục 1 sau khi có kết quả.
2. **Bước 6:** Chạy `scripts/pp_sweep.py` trên 15 cấu hình (`5 × 3` trên tập A) để xuất `d14_<ID>/sweep.csv` và `d14_<ID>/best_config.json`. Ghi bảng gốc vs `best_config.json`, kết quả **leave-one-out** và lý do chọn cấu hình vào mục 2 báo cáo.
3. **Bước 8:** Chạy `scripts/pp_compare.py` để tính phần trăm thay đổi công sửa và tạo ảnh BEV `d14_<ID>/compare/*.jpg`. Chèn 1 ảnh vào mục 3 báo cáo (`![so sánh](../../d14_<ID>/compare/<frame>.jpg)`), ghi nhận xét thẳng thắn dự đoán Bước 5b đúng hay sai so với số xóa/thêm/sửa thực tế.

:::image{src="walkthrough_images/cli_02_compare_52m_bev_s02f45.jpg" alt="Ảnh scripts/pp_compare.py frame s02f45 phạm vi 52 m với vòng đỏ 50 m" caption="Ảnh d14_001/compare/s02f45.jpg: bản đồ BEV 52 m kèm vòng đỏ 50 m đối chiếu gốc và tuned."}
:::

### Checkpoint Pha 3 và chuyển sang Tập B

Bạn kết thúc Pha 3 khi lịch sử GitHub đã có commit `"5b: dự đoán"` nằm trước bước chạy sweep, trong `d14_<ID>/` đã có đủ `sweep.csv`, `best_config.json`, bộ ảnh `compare/*.jpg`, và `report/individual/<MSSV>_<HoTen>.md` đã điền xong mục 1, 2 và 3 (kèm nhận định `leave-one-out` và 1 ảnh đối chiếu). Hãy đọc lại mức chênh lệch giữa cấu hình được chọn trên 2 frame còn lại với điểm `oracle` của từng frame trước khi áp dụng `best_config.json` lên 12 frame độc lập của Tập B ở Pha 4.

## Pha 4 — Chạy Tập B (Bước 7, Bước 9), hoàn thiện báo cáo và nộp commit SHA (2:25 → 4:05 · 100 phút)

### Bản chất vấn đề: kiểm chứng 12 frame độc lập và chống nộp thẳng box của model

Trong thực tế triển khai dữ liệu, một bộ siêu tham số chỉ có giá trị khi vượt qua bài kiểm tra trên dữ liệu mới chưa từng tham gia quá trình tinh chỉnh. **Tập B (`frames_B.txt`, 12 frame trải đều trên 4 sequence)** chính là sản phẩm bàn giao trọng tâm của Day 14 (`30/100 điểm` theo [RUBRIC.md](RUBRIC.md) mục 4). Ở Pha 4, bạn dùng `d14_<ID>/best_config.json` để sinh `pred_B.json`, đẩy lên CVAT thành task `D14-<ID>-B`, tự rà và sửa toàn bộ 12 frame bằng **`Shape`**, xuất `ref_B.json` ở Bước 9 rồi đo lại chi phí sửa bằng `scripts/pp_prelabel.py --gt d14_<ID>/ref_B.json` cho mục 4 báo cáo. **Mọi box nộp phải được người kiểm; tuyệt đối không nộp thẳng box của model**: theo Rubric mục 4, nếu mọi box trong mọi frame của `ref_B.json` trùng khớp `pred_B.json` (tọa độ, kích thước, yaw, lớp), bài bị mất toàn bộ 30 điểm mục này. Hoàn thành thêm mục 5 báo cáo nêu ít nhất 2 giới hạn.

### Giải nghĩa quy trình xuất `ref_B.json`, cấu hình `.gitignore` và nộp commit SHA trên LMS

Vì dùng CVAT Docker cá nhân (`http://localhost:8080`), coach **không mở CVAT cá nhân của bạn** mà chấm hoàn toàn từ các file trong repo tại **commit SHA** bạn nộp trên LMS ([SUBMISSION.md](SUBMISSION.md)). Đầu tiên, coach `git checkout <SHA>` và chạy `check_student.py` với `SEED` buổi lab trên `d14_<ID>/frames_A.txt` và `frames_B.txt`; nếu báo **CÓ VẤN ĐỀ** (sai SEED, sai ID hoặc sửa tay frame), các mục 1–5 không được chấm.

Trong cấu hình repo, `.gitignore` chỉ bỏ qua **`d14_*/png_*/`** (cùng `third_party/`, `out/`, `nop/`, `xem/`, `.venv/`, `*.zip`, `.env`), nên mọi file bắt buộc trong `d14_<ID>/` và `report/individual/<MSSV>_<HoTen>.md` **đều được phép commit và push** lên repo cá nhân **`K5-L23-DAY14-HoVaTen-MSSV-PointPillarsPrelabel`** (không dấu, không khoảng trắng; mời GitHub coach nếu Private). Không commit `Data-VF`, `third_party/`, mật khẩu CVAT hay `.env`. Hạn nộp link `https://github.com/<tài-khoản>/<repo>/commit/<SHA>` lên LMS trước **12:00 trưa hôm sau (GMT+7)**; không force-push hoặc xóa commit đã nộp ([RULES.md](RULES.md)).

### Các bước thực hiện Bước 7, Bước 9 và nộp bài lên GitHub/LMS

:::code-tabs{title="Bước 7, Bước 9: Pre-label B, xuất ref_B.json, đo mục 4 báo cáo và push commit SHA"}
```powershell {title="Windows (PowerShell)"}
python scripts/pp_prelabel.py --data "<Data-VF>" --frames "d14_<ID>/frames_B.txt" --config "d14_<ID>/best_config.json" --out "d14_<ID>/pred_B.json"
python scripts/vf_cvat.py prelabel --student "<ID>" --task B --data "<Data-VF>" --boxes "d14_<ID>/pred_B.json" --host http://localhost:8080 --user "<ten>"
# ... Mở task D14-<ID>-B trên CVAT 3D, sửa đủ 12 frame bằng Shape và bấm Save ...
python scripts/vf_cvat.py reference --student "<ID>" --task B --host http://localhost:8080 --user "<ten>"
python scripts/pp_prelabel.py --data "<Data-VF>" --frames "d14_<ID>/frames_B.txt" --config "d14_<ID>/best_config.json" --gt "d14_<ID>/ref_B.json"
# ... Điền mục 4 & mục 5 vào report/individual/<MSSV>_<HoTen>.md rồi commit & push ...
git add "d14_<ID>" report
git commit -m "Day 14: bài nộp"
git push -u origin main
git rev-parse HEAD
```
```bash {title="macOS / Linux (Bash)"}
python scripts/pp_prelabel.py --data "<Data-VF>" --frames "d14_<ID>/frames_B.txt" --config "d14_<ID>/best_config.json" --out "d14_<ID>/pred_B.json"
python scripts/vf_cvat.py prelabel --student "<ID>" --task B --data "<Data-VF>" --boxes "d14_<ID>/pred_B.json" --host http://localhost:8080 --user "<ten>"
# ... Mở task D14-<ID>-B trên CVAT 3D, sửa đủ 12 frame bằng Shape và bấm Save ...
python scripts/vf_cvat.py reference --student "<ID>" --task B --host http://localhost:8080 --user "<ten>"
python scripts/pp_prelabel.py --data "<Data-VF>" --frames "d14_<ID>/frames_B.txt" --config "d14_<ID>/best_config.json" --gt "d14_<ID>/ref_B.json"
# ... Điền mục 4 & mục 5 vào report/individual/<MSSV>_<HoTen>.md rồi commit & push ...
git add "d14_<ID>" report
git commit -m "Day 14: bài nộp"
git push -u origin main
git rev-parse HEAD
```
:::

1. **Bước 7a–7b.** Tạo `pred_B.json` từ `frames_B.txt` bằng cấu hình `best_config.json`. Nạp dự đoán lên CVAT để tạo task `D14-<ID>-B`. Kiểm tra trang **Tasks** có task A và task B. Khi tiếp tục trên task B đã tạo, dùng lại đúng file dự đoán đã nạp. Nếu script phát hiện file đã đổi, lệnh dừng và giữ nguyên nhãn cũ. Nếu script báo chưa thể đối chiếu, kiểm tra lại file đã nạp lúc tạo task. Trao đổi với coach trước khi tạo lại task vì `--replace` xóa toàn bộ nhãn đã sửa.

:::image{src="walkthrough_images/cvat_11_tasks_A_and_B.png" alt="Trang Tasks trên CVAT hiển thị đầy đủ cả task D14-001-A và D14-001-B" caption="Trang Tasks trên CVAT sau khi tạo thành công cả task D14-001-A (3 frame) và D14-001-B (12 frame)."}
:::

2. **Bước 7c:** Mở đủ 12 frame của `D14-<ID>-B` trên CVAT 3D, sửa hộp lệch/sai lớp (`Car` → `Truck`/`Bus`), xóa hộp thừa, vẽ thêm mọi đối tượng sót trong `50 m` bằng **`Shape`** (không `Track`), rồi bấm **Save**.
3. **Bước 9.** Xuất `d14_<ID>/ref_B.json` và kiểm tra đủ 12 frame. Đối chiếu số hộp xuất với **Items** trên CVAT. Chạy lệnh đo công sửa tập B ở [bước 9 trong README](README.md#xuất-nhãn-và-đo-công-sửa-tập-b-bước-9). Ghi số hộp cần xóa, thêm, sửa và chi phí vào mục 4. Nêu ít nhất hai giới hạn trong mục 5 báo cáo.
4. **Nộp bài.** Chạy lệnh Git theo [SUBMISSION.md](SUBMISSION.md#1-repo-nộp) từ gốc repo. Thay `<ID>` bằng mã trong tên thư mục đã tạo ở bước 1. Mở commit bạn vừa push lên repo cá nhân, rồi nộp link commit trên LMS trước **12:00 trưa hôm sau (GMT+7)**.

```text
K5-L23-DAY14-HoVaTen-MSSV-PointPillarsPrelabel/
├── d14_<ID>/                  # giữ số 0 đầu, ví dụ d14_001/ (png_*/ đã bị .gitignore bỏ qua)
│   ├── frames_A.txt           # 3 frame tập A kèm seed_tag= (không sửa tay)
│   ├── frames_B.txt           # 12 frame tập B kèm seed_tag= (không sửa tay)
│   ├── pred_A.json            # pre-label gốc trên tập A
│   ├── ref_A.json             # nhãn tham chiếu A đã sửa trên CVAT
│   ├── sweep.csv              # bảng quét lưới tham số Bước 6
│   ├── best_config.json       # cấu hình tốt nhất từ scripts/pp_sweep.py
│   ├── pred_B.json            # pre-label dùng best_config.json (B)
│   ├── compare/*.jpg          # ảnh BEV trước/sau 52 m (ít nhất ảnh dùng trong báo cáo)
│   └── ref_B.json             # nhãn Tập B 12 frame đã sửa xong (sản phẩm chính)
└── report/individual/<MSSV>_<HoTen>.md   # mục 1 đã commit trước sweep; đủ mục 1–5
```

### Checkpoint bàn giao cuối bài Lab

Trước khi dán link commit lên LMS, hãy mở repo `K5-L23-DAY14-HoVaTen-MSSV-PointPillarsPrelabel` trên GitHub ở đúng commit `git rev-parse HEAD` và đối chiếu đủ checklist trong [SUBMISSION.md](SUBMISSION.md) lẫn [CHECKPOINTS.md](CHECKPOINTS.md). Kiểm tra `seed_tag=` trong `frames_A.txt` và `frames_B.txt` còn nguyên vẹn, `ref_B.json` đủ 12 frame sau Save và khác `pred_B.json`, lịch sử commit chứng minh mục 1 (`5b: dự đoán`) có trước `sweep.csv`, `d14_*/png_*/` không bị push thừa, và không lộ `Data-VF`, `third_party/`, `<SEED>` thô hay mật khẩu CVAT.

:::checklist{title="Checklist bàn giao cuối bài Lab Day 14" tone="success"}
- [ ] `frames_A.txt` và `frames_B.txt` giữ nguyên `seed_tag=` và mã `<ID>` có số `0` đầu.
- [ ] Mục 1 báo cáo (`5b: dự đoán`) đã commit và push trước `sweep.csv`.
- [ ] Số hộp trong `ref_A.json` và `ref_B.json` khớp **Items** trên CVAT.
- [ ] Đã chạy `scripts/pp_prelabel.py --gt ref_B.json` và điền đủ 5 mục báo cáo.
- [ ] Repo `K5-L23-DAY14-HoVaTen-MSSV-PointPillarsPrelabel` đủ file theo `SUBMISSION.md`, đã bỏ qua `png_*/`.
- [ ] Đã nộp link commit `git rev-parse HEAD` lên LMS trước 12:00 trưa hôm sau.
:::
