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
4. Copy transcript generated from website to a new .txt file in a new directory, "transcripts"
    4a. Cache transcripts by video ID to avoid duplicates
5. Pass transcript to Ollama to find the funniest joke, collect timestamps of start and end of joke
6. Create a new trimmed .mp4 file with just the funniest joke using ffmpeg
    6a. Save trimmed video in a new directory, "highlights", named by video ID "video-id_highlight.mp4"

TODO:
- Optional: Check if video is longer than 10 minutes BEFORE downloading
- Do basic error handling and edge cases
- Improve analysis prompt for Ollama to get better timestamps/highlight
    - I believe this AI prompt is suggesting that we only care about the *first* funny moment.
    - We should look into having a better/more specific prompt so that we can get funny moments from any part of the video.
        - With further testing, I think it just favors the first funny moment of a video...
- Adjust frontend to show trimmed highlight video after it's been generated
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import yt_dlp
import os
import re
import json
import subprocess
import whisper
import requests
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
whisper_model = whisper.load_model("base") # Can use either "base", "small", "medium", or "large" depending on accuracy/speed tradeoffs

MAX_VIDEO_DURATION = 10 * 60  # 10 minutes in seconds
DOWNLOADS_DIR = "downloads" 
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma3:1b-it-qat"

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
        raise Exception("Video is too long (max 10 minutes)") # I'm debating keeping this altogether. If someone wants to download a longer video, they can on their own device

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

def call_ollama(prompt, model=OLLAMA_MODEL):
    """Call Ollama API to analyze newly generated transcript"""
    try:
        print(f"Calling Ollama with model: {model}")
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 500  # Limit response length
                }
            },
            timeout=60
        )

        if response.status_code == 200:
            return response.json()["response"]
        else:
            print(f"Ollama API error: {response.status_code} - {response.text}")
            return None

    except requests.exceptions.ConnectionError:
        print("Could not connect to Ollama. Make sure it's running on localhost:11434")
        print("Try running: ollama serve")
        return None
    except requests.exceptions.Timeout:
        print("Ollama request timed out. The model might be loading or busy.")
        return None
    except Exception as e:
        print("Error calling Ollama: " + str(e))
        return None

def analyze_transcript_with_ollama(transcript_text):
    """Use Ollama to read the transcript from Whisper and identify the funniest moment
    Returns the start time, end time, and quote of the funniest moment
    As of now, this returns a single funny moment, future development will return multiple
    """
    
    prompt = f"""Analyze this comedy video transcript and identify the single funniest moment. 

    Return ONLY a JSON array with these exact fields:
    - description: Brief description of the funniest moment
    - start_time: Start time in seconds (integer)
    - end_time: End time in seconds (integer)
    - quote: Exact quote from transcript

    Example format:
    [{{"description": "A hilarious joke about cats", "start_time": 120, "end_time": 130, "quote": "Why did the cat sit on the computer? To keep an eye on the mouse!"}}]

    Transcript:
    {transcript_text}

    Response (JSON only):"""
    
    analysis = call_ollama(prompt)
    if analysis:
        # AI models often wrap JSON in code blocks, this handles that case
        try:
            analysis = analysis.strip() # Clean up the response
            
            if "```json" in analysis:
                analysis = analysis.split("```json")[1].split("```")[0]
            elif "```" in analysis:
                analysis = analysis.split("```")[1].split("```")[0]
            
            # Remove any leading/trailing text that isn't JSON
            start_bracket = analysis.find('[')
            end_bracket = analysis.rfind(']') + 1
            
            if start_bracket != -1 and end_bracket != 0:
                analysis = analysis[start_bracket:end_bracket]
            
            return json.loads(analysis.strip())
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response from Ollama: {e}")
            print(f"Raw response: {analysis}")
            # If JSON parsing fails, return a simple fallback
            return [{
                "description": "Ollama analysis (parsing failed)", 
                "start_time": 0, 
                "end_time": 30, 
                "quote": analysis[:200] + "..." if analysis else "No analysis available"
            }]
        except Exception as e:
            print(f"Unexpected error parsing Ollama response: {e}")
            return []
    
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

def create_transcript_txt(transcript):
    """Create a plaintext .txt file version of the transcript for easier reading"""
    if not os.path.exists("transcripts"):
        os.makedirs("transcripts")
    
    video_id = get_video_id(transcript['url'])
    if not video_id:
        video_id = "unknown_video"

    txt_path = os.path.join("transcripts", f"{video_id}_transcript.txt")
    
    with open(txt_path, "w") as f:
        for segment in transcript['transcript']:
            minutes = segment['start_time'] // 60
            seconds = segment['start_time'] % 60
            f.write(f"[{minutes:02}:{seconds:02}] {segment['text']}\n")
    
    print(f"Transcript saved to {txt_path}")
    return txt_path

def create_highlight_video(video_path, start_time, end_time):
    """Create a new trimmed .mp4 file with just the funniest joke using ffmpeg"""
    if not os.path.exists("highlights"):
        os.makedirs("highlights")

    video_id = os.path.basename(video_path).split('_')[0]

    highlight_path = os.path.join("highlights", f"{video_id}_highlight.mp4")

    cmd = [
        "ffmpeg",
        "-y",  # overwrite video if exists with same video ID
        "-i", video_path,
        "-ss", str(start_time),
        "-to", str(end_time),
        "-c", "copy",
        highlight_path,
    ]
    subprocess.run(cmd, check=True)

    print(f"Highlight video saved to {highlight_path}")
    return highlight_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/highlights/<path:filename>')
def serve_highlight(filename):
    return send_from_directory('highlights', filename)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        data = request.json
        url = data.get('url', '').strip()

        if not url:
            return jsonify({'error': 'No URL provided'}), 400

        print("*** Step 1: DOWNLOAD VIDEO/AUDIO")
        try:
            video_result = download_video(url)
            video_info = {
                'title': video_result['title'],
                'duration': video_result['duration'],
                'uploader': video_result['uploader'],
                'path': video_result['video_path']  # Added this for highlight creation
            }
        except Exception as e:
            return jsonify({'error': str(e)}), 400 # Download error

        print("*** Step 2: EXTRACT AUDIO")
        try:
            audio_path = video_result['audio_path']
            result = create_transcript(audio_path)
            transcript = result["transcript"]
        except Exception as e:
            return jsonify({'error': 'Transcription failed: ' + str(e)}), 400
        
        print("*** Step 3: SAVE TRANSCRIPT AS .TXT")
        try:
            transcript_txt_path = create_transcript_txt({
                'url': url,
                'transcript': transcript
            })
            print("Transcript saved at:", transcript_txt_path)
        except Exception as e:
            return jsonify({'error': 'Unexpected error: ' + str(e)}), 500

        print("*** Step 4: ANALYZE TRANSCRIPT WITH OLLAMA")
        try:
            transcript_text = " ".join([seg['text'] for seg in transcript])
            analysis = analyze_transcript_with_ollama(transcript_text)
            print("Ollama analysis:", analysis)
            
            if not analysis or len(analysis) == 0:
                return jsonify({'error': 'No funny moments found in transcript'}), 400
                
        except Exception as e:
            return jsonify({'error': 'Ollama analysis failed: ' + str(e)}), 500

        print("*** Step 5: CREATE TRIMMED HIGHLIGHT VIDEO")
        try:
            funny_moment = analysis[0]
            highlight_path = create_highlight_video(
                video_info['path'], 
                funny_moment['start_time'], 
                funny_moment['end_time']
            )
            print("Highlight video created at:", highlight_path)
            video_info['highlight_path'] = highlight_path
            video_info['funny_moment'] = funny_moment
            
        except Exception as e:
            print(f"Highlight video creation failed: {e}")
            video_info['highlight_error'] = str(e) # Don't fail the whole request if highlight creation fails

        return jsonify({
            'success': True,
            'video_info': video_info,
            'transcript': transcript,
            'analysis': analysis
        })
    except Exception as e:
        return jsonify({'error': 'Unexpected error: ' + str(e)}), 500 # Catch-all


if __name__ == '__main__':
    app.run(debug=True, port=5050)