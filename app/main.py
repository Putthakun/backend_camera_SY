from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from redis_client import redis_client

import cv2
import insightface
import numpy as np
from scipy.spatial.distance import cosine
import json
import requests
from pydantic import BaseModel

app = FastAPI()

# โหลด InsightFace Model
model = insightface.app.FaceAnalysis()
model.prepare(ctx_id=0)  # ใช้ CPU (ctx_id=0) หรือ GPU (ctx_id=1)

# เปิดกล้องเว็บแคม
cap = cv2.VideoCapture(0)

# Pydantic model สำหรับรับข้อมูลจาก body
class MatchFaceRequest(BaseModel):
    new_vector: list  # รับข้อมูล new_vector เป็น list จาก body

def detect_and_embed_faces(frame):
    # ตรวจจับใบหน้าในเฟรม
    faces = model.get(frame)
    embeddings = []

    for face in faces:
        # ดึงพิกัดใบหน้า
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox

        # วาดกรอบรอบใบหน้า
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # สร้าง Face Embedding
        embedding = face.embedding
        embeddings.append(embedding)

        print(f"Face Embedding: {embedding[:5]}... (dim: {len(embedding)})")

    return frame, embeddings

def gen_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ทำการ mirror ภาพ
        frame = cv2.flip(frame, 1)

        # ตรวจจับใบหน้าและสร้าง Embedding
        frame, embeddings = detect_and_embed_faces(frame)

        print(f"type: {type(embeddings)}, length: {len(embeddings)}")
        
        # ส่ง embedding ไปยัง API สำหรับการจับคู่
        for embedding in embeddings:
            response = requests.post(
                "http://localhost:8000/api/match-face-vector",
                json={"new_vector": embedding.tolist()}  # ส่งข้อมูลใน body
            )

            if response.status_code == 200:
                print(response.json())
            else:
                print(response.text)

        # เข้ารหัสภาพเป็น JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/fetch-face-vectors-redis")
async def fetch_employee_vectors():
    # ดึงข้อมูล employee_vectors จาก Redis
    employee_vectors = redis_client.get("employee_vectors")
    
    if not employee_vectors:
        raise HTTPException(status_code=404, detail="No employee vectors found in Redis")

    # แปลงข้อมูลจาก JSON เป็น Python dictionary
    vectors_data = json.loads(employee_vectors)
    
    # พิมพ์ข้อมูลออกมาเพื่อทำการ debug
    print(f"vectors_data: {vectors_data}")  # แสดงข้อมูลที่ได้จาก Redis
    print(f"Data Type: {type(vectors_data)}")
    print(f"Length: {len(vectors_data)}")

    # ตรวจสอบชนิดของข้อมูลในแต่ละรายการ
    for i, item in enumerate(vectors_data):
        print(f"Item {i}: {item}")
        print(f"  Type: {type(item)}")
        print(f"  Keys: {item.keys() if isinstance(item, dict) else 'N/A'}")

    return {"message": "Employee vectors retrieved successfully", "data": vectors_data}


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
    threshold = 0.9  # ลดค่า threshold
    matched_employee = None
    min_distance = float('inf')

    # เปรียบเทียบเวกเตอร์ใหม่กับเวกเตอร์ใน Redis
    for item in vectors_data:
        employee_id = item['employee_id']  # ใช้ 'employee_id' จาก dictionary
        stored_vector = item['vector']  # ใช้ 'vector' จาก dictionary

        # แสดงค่าของ stored_vector เพื่อ debug
        print(f"Stored vector for employee {employee_id}: {stored_vector[:5]}... (dim: {len(stored_vector)})")

        print(f"Comparing new vector: {new_vector[:5]}... with stored vector: {stored_vector[:5]}...")

        # คำนวณ distance ระหว่างเวกเตอร์ใหม่กับเวกเตอร์ที่เก็บใน Redis
        distance = cosine(new_vector, stored_vector)

        print(f"Cosine distance: {distance}")

        if distance < min_distance:
            min_distance = distance
            matched_employee = employee_id

    if min_distance > threshold:
        raise HTTPException(status_code=404, detail="No matching face found")

    send_transaction_to_web_server(matched_employee, camera_id)

    return {
        "message": "Matching employee found",
        "employee_id": matched_employee,
        "similarity_score": 1 - min_distance  # แปลงเป็นค่าความคล้ายคลึง
    }

def send_transaction_to_web_server(emp_id, camera_id):
    timestamp = datetime.now().isoformat()  # ใช้เวลาปัจจุบันในรูปแบบ ISO 8601
    payload = {
        "emp_id": emp_id,
        "camera_id": camera_id,
        "timestamp": timestamp
    }
    
    try:
        # ส่งข้อมูลไปยัง web server API
        response = requests.post("http://your_web_server_address/api/record-transaction", json=payload)
        
        if response.status_code == 200:
            print("Transaction recorded successfully")
        else:
            print(f"Failed to record transaction: {response.text}")
    except Exception as e:
        print(f"Error sending data to web server: {e}")