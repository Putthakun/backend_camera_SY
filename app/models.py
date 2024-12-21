from pydantic import BaseModel
from typing import List

class VectorData(BaseModel):
    id: int
    vector: List[float]

class ReceiveVectorsRequest(BaseModel):
    vectors: List[VectorData]

class MatchFaceRequest(BaseModel):
    new_vector: list  # รับข้อมูล new_vector เป็น list จาก body