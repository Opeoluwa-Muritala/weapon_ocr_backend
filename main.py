import os
import base64
import json
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from .schemas import ImagePayload, AnalysisResponse
from .alert import send_email_via_smtp

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY", ""))
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")
ALERT_EMAIL = os.getenv("ALERT_EMAIL")

def to_image_part(data_url: str):
    if "," in data_url:
        header, b64 = data_url.split(",", 1)
        mime = "image/jpeg" if "image/jpeg" in header or "image/jpg" in header else "image/png"
    else:
        b64 = data_url
        mime = "image/jpeg"
    data = base64.b64decode(b64)
    return {"mime_type": mime, "data": data}

def build_prompt() -> str:
    return (
        "You are a vision safety analyzer. Analyze the provided image and respond with a single JSON object. "
        "Determine if any weapon is present focusing on guns and knives. Extract any readable text using optical character recognition. "
        "Return strictly formatted JSON with fields: "
        "weapon_detected (boolean), gun_detected (boolean), knife_detected (boolean), extracted_text (string). "
        "Use true or false for booleans. Do not include markdown, code fences, or additional commentary. "
        "If no text is readable, extracted_text must be an empty string. "
    )

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(payload: ImagePayload, background_tasks: BackgroundTasks):
    image_part = to_image_part(payload.image_base64)
    model = genai.GenerativeModel(model_name=MODEL_NAME, generation_config={"response_mime_type": "application/json"})
    response = model.generate_content([build_prompt(), image_part])
    raw_text = response.text
    data = json.loads(raw_text)
    weapon_detected = bool(data.get("weapon_detected", False))
    gun_detected = bool(data.get("gun_detected", False))
    knife_detected = bool(data.get("knife_detected", False))
    extracted_text = str(data.get("extracted_text", ""))

    if weapon_detected and ALERT_EMAIL:
        subject = "Weapon detected"
        html_body = (
            f"<h3>Alert</h3>"
            f"<p>Weapon detected: {weapon_detected}</p>"
            f"<p>Gun detected: {gun_detected}</p>"
            f"<p>Knife detected: {knife_detected}</p>"
            f"<p>Extracted text:</p><pre>{extracted_text}</pre>"
        )
        background_tasks.add_task(send_email_via_smtp, ALERT_EMAIL, subject, html_body)

    return AnalysisResponse(
        weapon_detected=weapon_detected,
        gun_detected=gun_detected,
        knife_detected=knife_detected,
        extracted_text=extracted_text,
    )
