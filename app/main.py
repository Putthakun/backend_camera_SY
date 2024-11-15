import cv2
import dlib
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from io import BytesIO

# เริ่มต้น FastAPI app
app = FastAPI()

# โหลดโมเดลสำหรับตรวจจับใบหน้า
detector = dlib.get_frontal_face_detector()

# เปิดกล้องเว็บแคม
cap = cv2.VideoCapture(0)

def detect_faces(frame):
    # แปลงภาพเป็นสีเทา (grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # ตรวจจับใบหน้า
    faces = detector(gray)
    
    # วาดกรอบรอบใบหน้า
    for face in faces:
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return frame

def gen_frames():
    while True:
        # อ่านภาพจากกล้อง
        ret, frame = cap.read()
        if not ret:
            break
        
        # ตรวจจับใบหน้าในภาพ
        frame = detect_faces(frame)
        
        # แปลงภาพเป็น JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        # ส่งภาพในรูปแบบ bytes
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# สร้าง route สำหรับการแสดงผลภาพ real-time
@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
