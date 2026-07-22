import requests

# The URL of your local FastAPI server
url = "http://127.0.0.1:8000/webhook/voice"

# Replace this with the name of a real audio file on your computer (e.g., a short voice memo)
audio_file_path = "test_audio.mp3"

try:
    with open(audio_file_path, "rb") as f:
        print(f"Sending {audio_file_path} to {url}...")
        files = {"file": (audio_file_path, f, "audio/mpeg")}
        response = requests.post(url, files=files)

    print("\n--- API Response ---")
    print(response.json())
except FileNotFoundError:
    print(f"Error: Could not find the file '{audio_file_path}'. Please make sure it exists in this folder.")