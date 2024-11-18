import cv2
import insightface
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

# โหลด InsightFace Model
model = insightface.app.FaceAnalysis()  # โมเดล InsightFace สำหรับ Face Detection และ Embedding
model.prepare(ctx_id=0)  # ใช้ CPU (ctx_id=0) หรือ GPU (ctx_id=1)

# เปิดกล้องเว็บแคม
cap = cv2.VideoCapture(0)

def detect_and_embed_faces(frame):
    # ตรวจจับใบหน้าในเฟรม
    faces = model.get(frame)  # ใช้ InsightFace สำหรับการตรวจจับ

    for face in faces:
        # ดึงพิกัดใบหน้า
        bbox = face.bbox.astype(int)  # แปลงพิกัดเป็นจำนวนเต็ม
        x1, y1, x2, y2 = bbox

        # วาดกรอบรอบใบหน้า
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # สร้าง Face Embedding
        embedding = face.embedding
        print(f"Face Embedding: {embedding[:5]}... (dim: {len(embedding)})")  # แสดงตัวอย่าง Embedding

    return frame

def gen_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ทำการ mirror ภาพ
        frame = cv2.flip(frame, 1)

        # ตรวจจับใบหน้าและสร้าง Embedding
        frame = detect_and_embed_faces(frame)

        # เข้ารหัสภาพเป็น JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
