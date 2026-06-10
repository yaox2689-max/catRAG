from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv(_PROJECT_ROOT / ".env.local", override=False)

BAIDU_OCR_API_KEY = os.getenv("BAIDU_OCR_API_KEY") or ""
BAIDU_OCR_SECRET_KEY = os.getenv("BAIDU_OCR_SECRET_KEY") or ""
BAIDU_OCR_URL = os.getenv(
    "BAIDU_OCR_URL", "https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic"
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}


def is_supported_image(filename: str) -> bool:
    return Path(filename or "").suffix.lower() in IMAGE_EXTENSIONS


def _get_baidu_access_token() -> str | None:
    if not BAIDU_OCR_API_KEY or not BAIDU_OCR_SECRET_KEY:
        return None

    try:
        resp = requests.post(
            "https://aip.baidubce.com/oauth/2.0/token",
            data={
                "grant_type": "client_credentials",
                "client_id": BAIDU_OCR_API_KEY,
                "client_secret": BAIDU_OCR_SECRET_KEY,
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("access_token") or None
    except Exception:
        return None


def recognize_image_bytes(image_bytes: bytes, filename: str = "") -> dict[str, Any]:
    text, raw = "", None
    token = _get_baidu_access_token()
    if not token:
        return {
            "provider": "baidu",
            "filename": filename,
            "text": "",
            "raw": None,
            "errors": ["百度 OCR 未配置，请设置 BAIDU_OCR_API_KEY 和 BAIDU_OCR_SECRET_KEY"],
        }

    try:
        resp = requests.post(
            BAIDU_OCR_URL,
            params={"access_token": token},
            data={"image": base64.b64encode(image_bytes).decode("utf-8")},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        words = []
        for item in data.get("words_result", []) or []:
            w = (item.get("words") or "").strip()
            if w:
                words.append(w)

        text = "\n".join(words).strip()
        return {
            "provider": "baidu",
            "filename": filename,
            "text": text,
            "raw": data,
            "errors": None,
        }
    except Exception as e:
        return {
            "provider": "baidu",
            "filename": filename,
            "text": "",
            "raw": None,
            "errors": [str(e)],
        }
