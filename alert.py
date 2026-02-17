import os
import logging
import requests

logger = logging.getLogger("alert")
EMAIL_SERVICE_URL = os.getenv("EMAIL_SERVICE_URL")

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
