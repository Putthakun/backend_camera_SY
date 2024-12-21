from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from redis_client import redis_client
from datetime import datetime
from pydantic import BaseModel
from scipy.spatial.distance import cosine

import cv2
import insightface
import numpy as np
import json
import requests
from models import *
import time

app = FastAPI()

# โหลด InsightFace Model
model = insightface.app.FaceAnalysis()
model.prepare(ctx_id=0)  # ใช้ CPU (ctx_id=0) หรือ GPU (ctx_id=1)

# เปิดกล้องเว็บแคม
cap = cv2.VideoCapture(0)

# ตัวแปรสำหรับเช็คว่า vector พร้อมหรือไม่
data_ready = False

def detect_and_embed_faces(frame):
    faces = model.get(frame)
    embeddings = []

    for face in faces:
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        embedding = face.embedding
        embeddings.append(embedding)
        print(f"Face Embedding: {embedding[:5]}... (dim: {len(embedding)})")

    return frame, embeddings

# ฟังก์ชันสำหรับส่งภาพที่ได้จากกล้อง
def gen_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame, embeddings = detect_and_embed_faces(frame)

        for embedding in embeddings:
            response = requests.post(
                "http://localhost:8000/api/match-face-vector",
                json={"new_vector": embedding.tolist()}  
            )
            if response.status_code == 200:
                print(response.json())
            else:
                print(response.text)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.on_event("startup")
async def fetch_and_cache_vectors_on_startup():
    retry_attempts = 5  # จำนวนครั้งที่ backend camera จะพยายามดึงข้อมูล
    for attempt in range(retry_attempts):
        employee_vectors = redis_client.get("employee_vectors")
        
        if employee_vectors:
            vectors_data = json.loads(employee_vectors)
            print("Employee vectors retrieved successfully during startup.")
            return  # ถ้าสำเร็จ ก็จะทำการส่งข้อมูลหรือหยุดการทำงาน
        else:
            print(f"Attempt {attempt + 1}: No data found in Redis, retrying...")
        
        # ถ้ายังไม่ได้ข้อมูล จะรอ 2 วินาทีและลองใหม่
        time.sleep(2)
    
    # ถ้าหลังจากพยายามครบแล้ว ยังไม่ได้ข้อมูล
    print("Failed to retrieve employee vectors after retries.")
    # แจ้งเตือนเมื่อไม่สามารถดึงข้อมูลจาก Redis ได้
    raise HTTPException(status_code=404, detail="No employee vectors found in Redis after retrying")


def send_transaction_to_web_server(emp_id, camera_id):
    timestamp = datetime.now().isoformat()  # ใช้เวลาปัจจุบันในรูปแบบ ISO 8601
    payload = {
        "emp_id": emp_id,
        "camera_id": camera_id,
        "timestamp": timestamp
    }
    
    try:
        # ส่งข้อมูลไปยัง web server API
        response = requests.post("http://web_server:8001/api/record-transaction", json=payload)
        
        if response.status_code == 200:
            print("Transaction recorded successfully")
        else:
            print(f"Failed to record transaction: {response.text}")
    except Exception as e:
        print(f"Error sending data to web server: {e}")

@app.post("/api/match-face-vector")
async def match_face_vector(request: MatchFaceRequest):
    new_vector = request.new_vector  # ใช้ข้อมูลที่รับจาก body

    # แสดงค่าของ new_vector เพื่อ debug
    print(f"New face vector: {new_vector[:5]}... (dim: {len(new_vector)})")

    # ดึง employee_vectors จาก Redis
    employee_vectors = redis_client.get("employee_vectors")
    
    if not employee_vectors:
        raise HTTPException(status_code=404, detail="No employee vectors found in Redis")

    # แปลง JSON เป็น Python list
    vectors_data = json.loads(employee_vectors)

    # กำหนด threshold สำหรับการจับคู่ (เช่น 0.3)
    threshold = 0.7  # ลดค่า threshold
    matched_employee = None
    min_distance = float('inf')

    # เปรียบเทียบเวกเตอร์ใหม่กับเวกเตอร์ใน Redis
    for item in vectors_data:
        employee_id = item['employee_id']  # ใช้ 'employee_id' จาก dictionary
        stored_vector = item['vector']  # ใช้ 'vector' จาก dictionary

        # คำนวณ distance ระหว่างเวกเตอร์ใหม่กับเวกเตอร์ที่เก็บใน Redis
        distance = cosine(new_vector, stored_vector)

        print(f"Cosine distance: {distance}")

        if distance < min_distance:
            min_distance = distance
            matched_employee = employee_id

    if min_distance > threshold:
        raise HTTPException(status_code=404, detail="No matching face found")

    # ตรวจสอบว่าเราเคยจับคู่ในช่วงเวลา 10 วินาทีหรือไม่
    last_match_time = redis_client.get(f"last_match_{matched_employee}")

    if last_match_time:
        last_match_timestamp = datetime.fromisoformat(last_match_time)
        time_diff = datetime.now() - last_match_timestamp

        if time_diff.total_seconds() < 10:
            # ถ้าจับคู่ใบหน้าซ้ำในช่วงเวลา 10 วินาที, ไม่ต้องสร้าง transaction ใหม่
            print(f"Duplicate match within {time_diff.total_seconds()} seconds. Skipping transaction.")
            return {
                "message": "Duplicate match within 10 seconds. Skipping transaction.",
                "employee_id": matched_employee,
                "similarity_score": 1 - min_distance
            }

    # ส่งข้อมูลการทำธุรกรรมไปยัง web server
    send_transaction_to_web_server(matched_employee, 1)

    # อัพเดทเวลาล่าสุดที่จับคู่
    redis_client.set(f"last_match_{matched_employee}", datetime.now().isoformat())

    return {
        "message": "Matching employee found",
        "employee_id": matched_employee,
        "similarity_score": 1 - min_distance  # แปลงเป็นค่าความคล้ายคลึง
    }
