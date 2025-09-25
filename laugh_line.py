"""
YouTube video downloader and transcription service using local Whisper + Ollama.

python3.13 -m venv venv
source venv/bin/activate
pip install flask yt-dlp python-dotenv openai-whisper requests

Try with this John Mulaney video: https://www.youtube.com/watch?v=DtBkEJU1zr0

Order of operations:
1. Download video using yt-dlp
2. Extract audio from video using ffmpeg
3. Use local Whisper model for transcription
4. Return transcript with timestamps to user
5. Use Ollama to summarize the funniest parts and generate clip highlights

TODO:
- Check if video is longer than 10 minutes BEFORE downloading
- Do basic error handling and edge cases
- Add Ollama integration for content analysis
"""

from flask import Flask, render_template, request, jsonify
import yt_dlp
import os
import re
import json
import subprocess
import whisper
import requests
import json
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
whisper_model = whisper.load_model("base") # Can use either "base", "small", "medium", or "large" depending on accuracy/speed tradeoffs

MAX_VIDEO_DURATION = 10 * 60  # 10 minutes in seconds
DOWNLOADS_DIR = "downloads" 
OLLAMA_BASE_URL = "http://localhost:11434"

if not os.path.exists(DOWNLOADS_DIR):
    os.makedirs(DOWNLOADS_DIR)

def get_video_id(url):
    """Extract YouTube video ID from URL
    The video ID is the end bit at the end of a YouTube URL
    In the example URL "https://www.youtube.com/watch?v=DtBkEJU1zr0", the video ID is "DtBkEJU1zr0"
    This is needed for file names and caching
    """
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1) # Returns the video ID
    return None # Invalid URL

def download_video(url):
    """Download YouTube video as .mp4 (≤1080p), then extract audio as .mp3 in the same directory."""
    video_id = get_video_id(url)
    if not video_id:
        raise Exception("Invalid YouTube URL format")

    output_video = os.path.join(DOWNLOADS_DIR, f"{video_id}_video.mp4")
    output_audio = os.path.join(DOWNLOADS_DIR, f"{video_id}_audio.mp3")

    print("Video ID:", video_id)

    # If video already cached, reuse it and extract audio if missing
    if os.path.exists(output_video):
        print("Using cached video:", output_video)
        if not os.path.exists(output_audio):
            extract_audio(output_video, output_audio)
        with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
            info = ydl.extract_info(url, download=False)
    else:
        ydl_opts = {
            "format": "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4][height<=1080]",
            "outtmpl": output_video,
            "quiet": True,
        } # Download best quality .mp4 capped at 1080p
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print("Downloading video...")
            info = ydl.extract_info(url, download=True)

        extract_audio(output_video, output_audio) # Extract .mp3 from .mp4

    if info.get("duration", 0) > MAX_VIDEO_DURATION:
        raise Exception("Video is too long (max 10 minutes)")

    return {
        "video_path": output_video,
        "audio_path": output_audio,
        "title": info.get("title", "Unknown"),
        "duration": info.get("duration", 0),
        "uploader": info.get("uploader", "Unknown"),
    }

def extract_audio(video_path, audio_path):
    """Extract audio from video using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",  # overwrite if exists
        "-i", video_path,
        "-vn",  # no video
        "-acodec", "mp3",
        audio_path,
    ]
    subprocess.run(cmd, check=True)

def generate_transcript_with_whisper(audio_path):
    """Generate transcript from audio file using local Whisper model"""
    if not os.path.exists(audio_path):
        raise Exception("Audio file not found: " + audio_path)

    try:
        print("Transcribing audio with local Whisper...")
        result = whisper_model.transcribe(audio_path)
        
        formatted_transcript = []
        for segment in result["segments"]:
            formatted_transcript.append({
                'start_time': int(segment['start']),  # Convert to integer seconds
                'text': segment['text'].strip()
            })

        print("Generated transcript with " + str(len(formatted_transcript)) + " segments")
        return formatted_transcript
        
    except Exception as e:
        print("Error generating transcript: " + str(e))
        raise Exception("Whisper transcription failed: " + str(e))

def call_ollama(prompt, model="llama3.2"):
    """Call Ollama API to analyze newly generated transcript"""
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            }
        )

        if response.status_code == 200:
            return response.json()["response"]
        else:
            print(f"Ollama API error: {response.status_code}")
            return None

    except requests.exceptions.ConnectionError:
        print("Could not connect to Ollama. Make sure it's running on localhost:11434")
        return None
    except Exception as e:
        print("Error calling Ollama: " +str(e))
        return None

def analyze_transcript_with_ollama(transcript_text):
    """Use Ollteama to find funny moments and generate highlights"""
    prompt = f"""
    Analyze this comedy transcript and identify the 3-5 funniest moments. For each moment, provide:
    1. A brief description of why it's funny
    2. The approximate timestamp where it occurs
    3. A short quote from that moment

    Transcript:
    {transcript_text}

    Please format your response as a JSON array with objects containing 'description', 'timestamp', and 'quote' fields.
    """
    
    analysis = call_ollama(prompt)
    if analysis:
        # AI models often wrap JSON in code blocks, this handles that case
        try:
            if "```json" in analysis:
                analysis = analysis.split("```json")[1].split("```")[0] # Get content inside ```json ... ```
            elif "```" in analysis:
                analysis = analysis.split("```")[1].split("```")[0] # Get anything else remaining in ```
            return json.loads(analysis.strip()) # Return cleaned JSON output
        except:
            # If JSON parsing fails, return a simple text analysis
            return [{"description": "Ollama analysis", "timestamp": 0, "quote": analysis[:200] + "..."}]
    
    return []

def create_transcript(audio_path):
    """Create transcript from audio file using local Whisper"""
    print("create_transcript() called with:", audio_path)

    try:
        transcript = generate_transcript_with_whisper(audio_path)
        return {
            "transcript": transcript,
        }

    except Exception as e:
        print("Transcript generation failed: " + str(e))
        placeholder_transcript = [
            {'start_time': 0, 'text': 'Audio download successful!'},
            {'start_time': 5, 'text': 'Local Whisper transcription failed.'},
            {'start_time': 10, 'text': 'Check the downloads directory in the Yapper parent directory.'},
            {'start_time': 15, 'text': 'Error: ' + str(e)}
        ] # Fallback to placeholder if transcription fails
        return {
            "transcript": placeholder_transcript,
        }

# TODO: Create a plaintext .txt file version of the transcript for easier reading
# TODO: Create a file that does another Ollama analysis on the transcript to find the funniest part(s)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        data = request.json
        url = data.get('url', '').strip()

        if not url:
            return jsonify({'error': 'No URL provided'}), 400

        # print("*** Step 1: DOWNLOAD VIDEO/AUDIO")
        try:
            video_result = download_video(url)
            video_info = {
                'title': video_result['title'],
                'duration': video_result['duration'],
                'uploader': video_result['uploader']
            }
        except Exception as e:
            return jsonify({'error': str(e)}), 400 # Download error

        # print("*** Step 2: EXTRACT AUDIO")
        try:
            audio_path = video_result['audio_path']
            result = create_transcript(audio_path)
            transcript = result["transcript"]
        except Exception as e:
            return jsonify({'error': 'Transcription failed: ' + str(e)}), 400
        
        return jsonify({
            'success': True,
            'video_info': video_info,
            'transcript': transcript,
        })
    except Exception as e:
        return jsonify({'error': 'Unexpected error: ' + str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5050)