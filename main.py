import anyio
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

app = FastAPI(title="Sideline Triage Assistant API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# The new SDK automatically picks up the GEMINI_API_KEY environment variable!
client = genai.Client()

# Use "gemini-3.5-flash-lite" here instead if you want to trade a little
# accuracy for even faster/cheaper responses.
MODEL_NAME = "gemini-3.5-flash"

# A strict output schema. This lets Gemini skip "how should I format this"
# reasoning entirely and also removes the need to strip ```json fences on
# the frontend, which is itself a small speed + reliability win.
TRIAGE_SCHEMA = types.Schema(
    type="OBJECT",
    properties={
        "transcript": types.Schema(
            type="STRING",
            description="Best-effort transcript of whatever was said, even if it's just a few words.",
        ),
        "player": types.Schema(
            type="STRING",
            description="Player name/number if mentioned, otherwise 'Unknown'.",
        ),
        "symptom": types.Schema(
            type="STRING",
            description="The injury/symptom described, in plain language.",
        ),
        "risk": types.Schema(
            type="STRING",
            enum=["Low", "Moderate", "High"],
            description="Best-guess injury severity given the (possibly incomplete) information.",
        ),
        "protocol": types.Schema(
            type="STRING",
            description="Short, immediate, actionable triage steps a non-medical coach can follow right now.",
        ),
        "confidence_note": types.Schema(
            type="STRING",
            description="One short sentence flagging anything that was unclear or assumed, or 'None' if the report was clear.",
        ),
    },
    required=["transcript", "player", "symptom", "risk", "protocol", "confidence_note"],
)

PROMPT = (
    "You are a sports-science triage assistant used courtside by youth/amateur coaches "
    "who have NO medical training and only a few seconds to act. The audio you receive "
    "will often be short, urgent, and fragmented (e.g. 'Number 7, twisted her ankle bad' "
    "or just 'knee, he's screaming') rather than a full detailed report — that is expected "
    "and NOT a failure case. Do the best triage you can with whatever is said:\n"
    "1. Transcribe whatever was actually said, even if it's just a few words.\n"
    "2. Extract the player identifier (name/number) and the symptom/injury described. "
    "If something isn't mentioned, use 'Unknown' rather than asking for more detail.\n"
    "3. Give your best-judgment injury risk: Low, Moderate, or High. When information is "
    "sparse, err toward the safer (higher) risk category rather than assuming the best case.\n"
    "4. Give a short, immediate, step-by-step triage protocol a non-medical coach can follow "
    "right now on the sideline (e.g. do/don't move the player, ice, when to call emergency "
    "services). Keep it actionable, not a lecture.\n"
    "5. In 'confidence_note', briefly flag anything you had to assume because the report "
    "was incomplete, or 'None' if it was clear.\n"
    "Never refuse or ask for clarification — always return your best triage assessment from "
    "whatever audio you're given."
)


def _process_audio_sync(audio_bytes: bytes, mime_type: str) -> dict:
    """Sends audio directly as inline data, bypassing the File API processing delays."""
    try:
        audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[PROMPT, audio_part],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=TRIAGE_SCHEMA,
                # "minimal" thinking = fastest possible response for this kind of
                # short extraction/classification task (Gemini 3.x models).
                thinking_config=types.ThinkingConfig(thinking_level="minimal"),
            ),
        )

        return {"triage_assessment": response.text}

    except Exception as e:
        raise RuntimeError(f"Gemini API error: {str(e)}")


@app.post("/webhook/voice")
async def voice_webhook(file: UploadFile = File(...)):
    """Webhook endpoint to receive audio payloads."""
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio upload.")

        mime_type = file.content_type or "audio/webm"

        # Process the audio in a separate thread. We pass the bytes straight
        # through instead of writing/reading a temp file first — that disk
        # round-trip was pure overhead since the SDK just wants raw bytes.
        result = await anyio.to_thread.run_sync(_process_audio_sync, audio_bytes, mime_type)
        return {"status": "success", "data": result}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sideline Triage Assistant</title>
        <style>
            :root {
                --red: #e63946;
                --red-dark: #c1121f;
                --navy: #1d3557;
                --slate: #457b9d;
                --bg: #f1f5f9;
                --low: #2a9d8f;
                --moderate: #e9a13b;
                --high: #e63946;
            }
            * { box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 720px;
                margin: 0 auto;
                padding: 24px 16px 60px;
                background: var(--bg);
                color: #1e293b;
            }
            .container {
                background: white;
                padding: 28px;
                border-radius: 16px;
                box-shadow: 0 8px 30px rgba(0,0,0,0.08);
            }
            h1 { color: var(--navy); margin-bottom: 4px; font-size: 1.6rem; }
            .subtitle { color: #64748b; margin-top: 0; margin-bottom: 20px; font-size: 0.95rem; }

            .recorder {
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 14px;
                padding: 28px 16px;
                background: linear-gradient(180deg, #fafbfc, #f1f5f9);
                border: 1px solid #e2e8f0;
                border-radius: 14px;
            }
            .mic-btn {
                width: 92px;
                height: 92px;
                border-radius: 50%;
                border: none;
                background: var(--red);
                color: white;
                font-size: 34px;
                cursor: pointer;
                box-shadow: 0 4px 14px rgba(230,57,70,0.4);
                transition: transform 0.15s ease, background 0.2s ease;
            }
            .mic-btn:hover { background: var(--red-dark); transform: scale(1.04); }
            .mic-btn.recording {
                animation: pulse 1.4s infinite;
                background: var(--red-dark);
            }
            @keyframes pulse {
                0%   { box-shadow: 0 0 0 0 rgba(230,57,70,0.55); }
                70%  { box-shadow: 0 0 0 22px rgba(230,57,70,0); }
                100% { box-shadow: 0 0 0 0 rgba(230,57,70,0); }
            }
            .hint { font-size: 0.85rem; color: #64748b; text-align: center; max-width: 380px; }
            #timer {
                font-variant-numeric: tabular-nums;
                font-size: 1.3rem;
                font-weight: 700;
                color: var(--red-dark);
                min-height: 1.6rem;
            }

            #status {
                margin-top: 16px;
                font-weight: 600;
                color: var(--slate);
                text-align: center;
                min-height: 1.2rem;
            }

            .spinner {
                width: 18px; height: 18px;
                border: 3px solid #cbd5e1;
                border-top-color: var(--slate);
                border-radius: 50%;
                display: inline-block;
                vertical-align: middle;
                margin-right: 8px;
                animation: spin 0.8s linear infinite;
            }
            @keyframes spin { to { transform: rotate(360deg); } }

            #resultCard { margin-top: 24px; display: none; }
            .risk-badge {
                display: inline-block;
                padding: 4px 14px;
                border-radius: 999px;
                color: white;
                font-weight: 700;
                font-size: 0.85rem;
                letter-spacing: 0.03em;
            }
            .risk-Low { background: var(--low); }
            .risk-Moderate { background: var(--moderate); }
            .risk-High { background: var(--high); }

            .field {
                margin-top: 14px;
                padding: 12px 14px;
                background: #f8fafc;
                border-left: 4px solid var(--slate);
                border-radius: 6px;
            }
            .field.protocol { border-left-color: var(--navy); }
            .field.note { border-left-color: #cbd5e1; font-style: italic; color: #64748b; }
            .field-label {
                font-size: 0.72rem;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                color: #94a3b8;
                font-weight: 700;
                margin-bottom: 4px;
            }
            .field-value { font-size: 0.98rem; line-height: 1.45; white-space: pre-wrap; }

            #errorBox {
                display: none;
                margin-top: 20px;
                padding: 14px;
                background: #fff1f2;
                border: 1px solid #fecdd3;
                border-radius: 8px;
                color: #9f1239;
                font-size: 0.9rem;
                white-space: pre-wrap;
            }

            .retry-btn {
                margin-top: 16px;
                background: var(--navy);
                color: white;
                border: none;
                padding: 10px 18px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 0.9rem;
            }
            .retry-btn:hover { opacity: 0.9; }

            .disclaimer {
                display: flex;
                gap: 10px;
                align-items: flex-start;
                background: #fff8e6;
                border: 1px solid #fde68a;
                color: #7c5a00;
                border-radius: 10px;
                padding: 10px 14px;
                font-size: 0.82rem;
                line-height: 1.4;
                margin-bottom: 20px;
            }
            .disclaimer b { color: #5c4300; }

            .emergency-cta {
                display: none;
                align-items: center;
                justify-content: space-between;
                gap: 10px;
                margin-top: 16px;
                padding: 12px 16px;
                background: var(--high);
                color: white;
                border-radius: 10px;
                font-weight: 700;
                font-size: 0.9rem;
            }
            .emergency-cta a {
                color: white;
                background: rgba(0,0,0,0.2);
                padding: 6px 12px;
                border-radius: 6px;
                text-decoration: none;
                white-space: nowrap;
            }

            .footer-note {
                margin-top: 22px;
                text-align: center;
                font-size: 0.75rem;
                color: #94a3b8;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎙️ Sideline Triage Assistant</h1>
            <p class="subtitle">Instant AI guidance to help you decide what to do next — in seconds.</p>

            <div class="disclaimer">
                <span>⚠️</span>
                <div><b>This is decision support, not medical treatment.</b> It tells you what to
                do next — it doesn't replace a doctor, paramedic, or emergency services. For
                any life-threatening injury, call emergency services immediately.</div>
            </div>

            <div class="recorder">
                <button id="micBtn" class="mic-btn" onclick="toggleRecording()">🔴</button>
                <div id="timer"></div>
                <div class="hint">Tap, say what happened in a few words, tap again — e.g. "Number 7, twisted her ankle bad." No full report needed.</div>
            </div>

            <div id="status"></div>
            <div id="errorBox"></div>

            <div id="resultCard">
                <div style="display:flex; align-items:center; justify-content:space-between;">
                    <span class="field-label" style="margin:0;">AI GUIDANCE — NOT A DIAGNOSIS</span>
                    <span id="riskBadge" class="risk-badge"></span>
                </div>

                <div id="emergencyCta" class="emergency-cta">
                    <span>🚨 High risk — consider calling emergency services now</span>
                    <a href="tel:112">Call 112</a>
                </div>

                <div class="field">
                    <div class="field-label">Player</div>
                    <div class="field-value" id="fPlayer"></div>
                </div>
                <div class="field">
                    <div class="field-label">Symptom</div>
                    <div class="field-value" id="fSymptom"></div>
                </div>
                <div class="field protocol">
                    <div class="field-label">What To Do Right Now</div>
                    <div class="field-value" id="fProtocol"></div>
                </div>
                <div class="field">
                    <div class="field-label">What I Heard</div>
                    <div class="field-value" id="fTranscript"></div>
                </div>
                <div class="field note" id="fNoteWrap">
                    <div class="field-label">Heads-up</div>
                    <div class="field-value" id="fNote"></div>
                </div>

                <button class="retry-btn" onclick="resetUI()">Record another report</button>
            </div>

            <div class="footer-note">Sideline Triage Assistant gives first-response guidance only. Always follow up with a qualified medical professional.</div>
        </div>

        <script>
            let mediaRecorder;
            let audioChunks = [];
            let isRecording = false;
            let timerInterval;
            let secondsElapsed = 0;

            const micBtn = document.getElementById('micBtn');
            const timerEl = document.getElementById('timer');
            const statusDiv = document.getElementById('status');
            const errorBox = document.getElementById('errorBox');
            const resultCard = document.getElementById('resultCard');

            function toggleRecording() {
                if (isRecording) {
                    stopRecording();
                } else {
                    startRecording();
                }
            }

            function formatTime(s) {
                const m = Math.floor(s / 60).toString().padStart(2, '0');
                const sec = (s % 60).toString().padStart(2, '0');
                return m + ':' + sec;
            }

            async function startRecording() {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream);
                    audioChunks = [];

                    mediaRecorder.ondataavailable = event => audioChunks.push(event.data);
                    mediaRecorder.onstop = () => {
                        stream.getTracks().forEach(t => t.stop());
                        submitAudio();
                    };

                    mediaRecorder.start();
                    isRecording = true;
                    micBtn.classList.add('recording');
                    micBtn.textContent = '⏹️';
                    statusDiv.textContent = 'Listening...';
                    errorBox.style.display = 'none';
                    resultCard.style.display = 'none';

                    secondsElapsed = 0;
                    timerEl.textContent = formatTime(0);
                    timerInterval = setInterval(() => {
                        secondsElapsed++;
                        timerEl.textContent = formatTime(secondsElapsed);
                    }, 1000);

                } catch (err) {
                    showError('Microphone access denied or not available.\\n' + err);
                }
            }

            function stopRecording() {
                mediaRecorder.stop();
                isRecording = false;
                micBtn.classList.remove('recording');
                micBtn.textContent = '🔴';
                clearInterval(timerInterval);
            }

            async function submitAudio() {
                statusDiv.innerHTML = '<span class="spinner"></span>Analyzing...';
                timerEl.textContent = '';

                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                const formData = new FormData();
                formData.append('file', audioBlob, 'live_audio.webm');

                try {
                    const response = await fetch('/webhook/voice', { method: 'POST', body: formData });
                    const data = await response.json();

                    if (!response.ok || !data.data) {
                        showError('Backend error: ' + (data.detail || JSON.stringify(data)));
                        return;
                    }

                    const assessment = JSON.parse(data.data.triage_assessment);
                    renderResult(assessment);
                    statusDiv.textContent = '';
                } catch (error) {
                    showError('Network/frontend error: ' + error.message);
                }
            }

            function renderResult(a) {
                document.getElementById('fPlayer').textContent = a.player || 'Unknown';
                document.getElementById('fSymptom').textContent = a.symptom || 'Unknown';
                document.getElementById('fProtocol').textContent = a.protocol || '';
                document.getElementById('fTranscript').textContent = a.transcript || '';

                const badge = document.getElementById('riskBadge');
                const risk = a.risk || 'Moderate';
                badge.textContent = risk + ' Risk';
                badge.className = 'risk-badge risk-' + risk;

                const noteWrap = document.getElementById('fNoteWrap');
                if (a.confidence_note && a.confidence_note.toLowerCase() !== 'none') {
                    document.getElementById('fNote').textContent = a.confidence_note;
                    noteWrap.style.display = 'block';
                } else {
                    noteWrap.style.display = 'none';
                }

                document.getElementById('emergencyCta').style.display = (risk === 'High') ? 'flex' : 'none';

                resultCard.style.display = 'block';
            }

            function showError(msg) {
                statusDiv.textContent = '';
                errorBox.textContent = msg;
                errorBox.style.display = 'block';
            }

            function resetUI() {
                resultCard.style.display = 'none';
                errorBox.style.display = 'none';
                document.getElementById('emergencyCta').style.display = 'none';
                statusDiv.textContent = '';
                timerEl.textContent = '';
            }
        </script>
    </body>
    </html>
    """