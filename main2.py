import asyncio
import os
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
from openai import AzureOpenAI

# -------------------------------------------------------------------
# ENV LOADING (SAFE & EXPLICIT)
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

AOAI_KEY = os.getenv("AOAI_KEY")
AOAI_ENDPOINT = "https://eastus1ws.openai.azure.com/"
AZURE_SPEECH_KEY = os.getenv("AOAI_KEY")



# -------------------------------------------------------------------
# AZURE OPENAI (TRANSLATION)
# -------------------------------------------------------------------

CHAT_DEPLOYMENT = "gpt-4.1-nano"

aoai_client = AzureOpenAI(
    api_key=AOAI_KEY,
    azure_endpoint=AOAI_ENDPOINT,
    api_version="2024-10-01-preview",
)

# -------------------------------------------------------------------
# AZURE SPEECH CONFIG (ASR + TTS)
# -------------------------------------------------------------------

speech_config = speechsdk.SpeechConfig(
    subscription=AOAI_KEY,
    region="eastus2",
)

# 🔁 CHANGE THESE IF NEEDED
speech_config.speech_recognition_language = "hi-IN"
speech_config.speech_synthesis_voice_name = "te-IN-ShrutiNeural"

# Browser-playable MP3
speech_config.set_speech_synthesis_output_format(
    speechsdk.SpeechSynthesisOutputFormat.Audio48Khz192KBitRateMonoMp3
)

# -------------------------------------------------------------------
# FASTAPI
# -------------------------------------------------------------------

app = FastAPI()


@app.get("/")
def health():
    return {"status": "Live Translator Backend Running"}


# -------------------------------------------------------------------
# WEBSOCKET ENDPOINT
# -------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_event_loop()

    # ---- Push audio stream (PCM16 from browser) ----
    input_stream = speechsdk.audio.PushAudioInputStream(
        speechsdk.audio.AudioStreamFormat(
            samples_per_second=16000,
            bits_per_sample=16,
            channels=1,
        )
    )

    audio_config = speechsdk.audio.AudioConfig(stream=input_stream)
    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config,
    )

    # ---- FINAL SPEECH HANDLER ----
    async def handle_final_text(text: str):
        text = text.strip()
        if not text:
            return

        print("FINAL Transcript:", text)

        # -------- TRANSLATION --------
        translation = aoai_client.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            messages=[
                {
                    "role": "user",
                    "content": f"""Translate the following Hindi speech into natural, conversational Telugu.

Rules:
- Preserve intent and tone
- Do NOT translate word by word
- If unclear or incomplete, return EMPTY STRING
- Do NOT explain or apologize

Hindi:
{text}""",
                }
            ],
            temperature=0.3,
        ).choices[0].message.content.strip()

        if not translation:
            return

        print("Translation:", translation)

        # -------- TTS (AARTI DRAGON HD) --------
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config,
            audio_config=None,  # return audio bytes
        )

        result = synthesizer.speak_text_async(translation).get()

        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            return

        await ws.send_bytes(result.audio_data)

    # ---- ASR CALLBACK ----
    def recognized(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            asyncio.run_coroutine_threadsafe(
                handle_final_text(evt.result.text),
                loop,
            )

    recognizer.recognized.connect(recognized)
    recognizer.start_continuous_recognition()

    try:
        while True:
            audio_bytes = await ws.receive_bytes()
            input_stream.write(audio_bytes)

    except WebSocketDisconnect:
        print("WebSocket disconnected")

    finally:
        input_stream.close()
        recognizer.stop_continuous_recognition()