from pydantic import BaseModel
from typing import List

class VectorData(BaseModel):
    id: int
    vector: List[float]

class ReceiveVectorsRequest(BaseModel):
    vectors: List[VectorData]