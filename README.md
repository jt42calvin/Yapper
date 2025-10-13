# Yapper — YouTube Comedy Highlight Generator

Extracts and trims the funniest moments from YouTube videos by combining local transcription (Whisper) with AI analysis (Ollama).

Why this exists
- Quickly surface the best part of long comedy videos so viewers can see a short highlight instead of watching an entire clip.

What it does
- Downloads a YouTube video (yt-dlp)
- Extracts audio (ffmpeg)
- Generates sentence-level transcripts using local Whisper
- Uses an LLM (Ollama) to locate the funniest moment and returns start/end timestamps
- Trims the video to create a highlight (.mp4)

Quick start (macOS / Linux)

1) Create and activate a Python virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

2) Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, install the main packages used in this repo:

```bash
pip install flask yt-dlp python-dotenv openai-whisper requests
```

3) Install ffmpeg (required for audio/video processing)

macOS (Homebrew):
```bash
brew install ffmpeg
```

Ubuntu/Debian:
```bash
sudo apt update && sudo apt install -y ffmpeg
```

4) (Optional) Since this uses Ollama for analysis, run it locally and ensure it's available at http://localhost:11434

5) Run the app

```bash
python3 laugh_line.py
```

Open http://127.0.0.1:5050 in your browser and paste a YouTube URL.

Notes & tips
- Some YouTube videos may require cookies or an authenticated session to download; export a cookies file and point yt-dlp to it if needed.
- The default code caps video duration to 10 minutes; change `MAX_VIDEO_DURATION` in `laugh_line.py` if you want longer downloads. I just didn't want to process large videos, personally.
- Whisper (local) and Ollama models can be memory/CPU intensive—use smaller models for development.

Known limitations
- Humor detection is imperfect and may select setup instead of punchline.
- Works best for spoken stand-up-style content, not visual-only jokes.

Contributing
- Read the source in `laugh_line.py` to understand the pipeline. Feel free to open an issue or PR for fixes and improvements.

Enjoy exploring highlights — and please run this tool only on content you are allowed to download. :)