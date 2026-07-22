import os
import tempfile
from pathlib import Path
import anyio
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai

app = FastAPI(title="Voice-Triage API (Free Version)")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# The new SDK automatically picks up the GEMINI_API_KEY environment variable!
client = genai.Client()

def _process_audio_sync(file_path: str) -> dict:
    """Uploads audio to Gemini and gets a structured JSON triage response."""
    try:
        # 1. Upload the temporary audio file to Gemini's servers
        audio_file = client.files.upload(file=file_path)
        
        prompt = (
            "You are an elite sports science AI. Listen to the provided audio file of a coach's report. "
            "1. Transcribe the audio exactly. "
            "2. Extract the player name and the physical symptom. "
            "3. Flag the injury risk as Low, Moderate, or High. "
            "4. Provide a brief, immediate triage protocol. "
            "Return your response STRICTLY as a JSON object with the exact keys: "
            "'transcript', 'player', 'symptom', 'risk', and 'protocol'."
        )
        
        # 2. Generate the response
        response = client.models.generate_content(
            model='gemini-3.5-flash',
            contents=[prompt, audio_file]
        )
        # 3. Clean up the file from Google's servers to be responsible
        client.files.delete(name=audio_file.name)
        
        return {"triage_assessment": response.text}
        
    except Exception as e:
        raise RuntimeError(f"Gemini API error: {str(e)}")

@app.post("/webhook/voice")
async def voice_webhook(file: UploadFile = File(...)):
    """Webhook endpoint to receive audio payloads."""
    
    suffix = Path(file.filename).suffix or ".mp3"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    
    try:
        audio_bytes = await file.read()
        Path(tmp_path).write_bytes(audio_bytes)
        
        # Process the audio in a separate thread
        result = await anyio.to_thread.run_sync(_process_audio_sync, tmp_path)
        return {"status": "success", "data": result}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)

@app.get("/")
async def health_check():
    return {"status": "active", "service": "Voice-Triage Backend (Free)"}