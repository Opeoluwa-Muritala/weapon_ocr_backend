from pydantic import BaseModel

class ImagePayload(BaseModel):
    image_base64: str

class AnalysisResponse(BaseModel):
    weapon_detected: bool
    gun_detected: bool
    knife_detected: bool
    extracted_text: str
