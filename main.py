import os
import base64
import json
import logging
import requests
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types  # Required for safety settings
from pydantic import BaseModel

logger = logging.getLogger("alert")
EMAIL_SERVICE_URL = os.getenv("EMAIL_SERVICE_URL")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Updated default model to avoid the 404 Not Found error
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
ALERT_EMAIL = os.getenv("ALERT_EMAIL")
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY", ""))

class ImagePayload(BaseModel):
    image_base64: str

class AnalysisResponse(BaseModel):
    weapon_detected: bool
    gun_detected: bool
    knife_detected: bool
    extracted_text: str


def send_email_via_smtp(recipient, subject, html_body):
    """
    Sends email by calling the FastAPI email microservice.
    Returns True if successful, False otherwise.
    """
    if not EMAIL_SERVICE_URL:
        logger.error("EMAIL_SERVICE_URL not set in environment variables")
        return False

    try:
        response = requests.post(
            EMAIL_SERVICE_URL,
            json={
                "to": recipient,
                "subject": subject,
                "html": html_body
            },
            timeout=10
        )
        res_json = response.json()

        if res_json.get("success"):
            logger.info(f"Email sent successfully to {recipient} via service")
            return True
        else:
            logger.error(f"Email service returned error: {res_json.get('error')}")
            return False

    except Exception as e:
        logger.error(f"Failed to contact email service: {e}")
        return False

def to_image_part(data_url: str):
    if "," in data_url:
        header, b64 = data_url.split(",", 1)
        mime = "image/jpeg" if "image/jpeg" in header or "image/jpg" in header else "image/png"
    else:
        b64 = data_url
        mime = "image/jpeg"
    data = base64.b64decode(b64)
    return {"inline_data": {"mime_type": mime, "data": data}}

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
    try:
        image_part = to_image_part(payload.image_base64)
        contents = [{"role": "user", "parts": [{"text": build_prompt()}, image_part]}]
        
        # Lower safety thresholds so the model is allowed to look for weapons
        safety_settings = [
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            )
        ]

        # Use GenerateContentConfig to pass the safety settings and require JSON
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                safety_settings=safety_settings
            )
        )
        
        raw_text = getattr(response, "text", None) or getattr(response, "output_text", "")
        
        if not raw_text:
            logger.error("Empty response from Gemini. Possible safety block.")
            raise ValueError("Empty response received from the vision model. Content may be blocked by safety filters.")

        # Clean up potential markdown formatting (```json ... ```) before parsing
        clean_text = raw_text.strip().removeprefix("```json").removesuffix("```").strip()
        data = json.loads(clean_text)
        
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

    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON: {raw_text}")
        raise HTTPException(status_code=500, detail="Invalid JSON format returned from the AI model.")
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
