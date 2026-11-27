# Báo cáo Hướng 2 (bonus, chưa chạy được) — Day 14

Thời gian thử: khoảng 45 phút, ngày 27/11/2026. Kết quả: **chưa chạy được** function. Ghi lại các bước đã làm và chỗ vướng.

## Đã làm

1. Xác định thư mục CVAT đang chạy: `D:\Ai20K\AutoQC-Pipeline\infra\cvat\cvat` (project Docker `cvat`, lấy từ label `com.docker.compose.project.working_dir`).
2. Bật serverless cho stack CVAT đang chạy:
   ```
   docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml up -d
   ```
   Kết quả: container `nuclio` (image `quay.io/nuclio/dashboard:1.16.3-amd64`) lên và `healthy`. CVAT vẫn chạy sau khi restart: `http://localhost:8080/api/server/about` trả `200`. Dữ liệu task A/B giữ nguyên.
3. Kiểm tra image: chưa có image `nuclio` nào trước đó; compose đã pull `dashboard:1.16.3-amd64`.

## Vướng ở đâu

- **Dashboard Nuclio không publish cổng ra host.** `docker port nuclio` rỗng; `docker inspect nuclio` chỉ thấy `80/tcp` trong network `cvat`. Vì vậy `nuctl` trên host không kết nối được `127.0.0.1:8070`.
- **`nuctl` chưa có trên máy.** `Get-Command nuctl` không thấy. Cần tải binary đúng phiên bản 1.16.3 từ GitHub release.
- Hệ quả: chưa deploy được function, nên chưa có cuboid từ CVAT để so với `pred_*.json`.

## Nếu làm tiếp

1. Thêm mapping cổng cho service `nuclio` (ví dụ `ports: ["127.0.0.1:8070:8070"]` bằng một file compose override), hoặc chạy `nuctl` trong network `cvat`.
2. Tải `nuctl` 1.16.3 cho Windows amd64, chạy `nuctl create project cvat --platform local`.
3. Viết function: `parse_pcd` (x y z rgb, binary) + đảo kênh intensity xấp xỉ từ `rgb` + `Detector(Config())` từ `vf_pp`, trả 16 số `points` mỗi cuboid. Build base `python:3.10-slim` + `pip install torch` (tải lớn, dễ quá thời gian 45 phút).
4. Deploy, gọi `POST /api/lambda/requests`, đối chiếu tâm box với `pred_*.json` (ngưỡng 0,5 m), quay video ngắn.

## Ghi chú an toàn

Dashboard Nuclio không có đăng nhập. Chưa mở cổng 8070 ra ngoài host.
