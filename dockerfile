# ใช้ base image ของ Python
FROM python:3.9-slim

# ติดตั้ง dependencies ที่จำเป็น เช่น cmake และ build-essential
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# กำหนด working directory
WORKDIR /app

# คัดลอก requirements.txt ไปยัง container
COPY requirements.txt .

# ติดตั้ง dependencies
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกโฟลเดอร์ app ทั้งหมดไปยัง /app ใน container
COPY app /app

# เปิดพอร์ต 8000 สำหรับ FastAPI
EXPOSE 8000

# รัน FastAPI เมื่อ container เริ่มทำงาน
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
