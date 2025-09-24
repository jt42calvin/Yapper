"""
YouTube video downloader and transcription service using Whisper API.

python3.13 -m venv venv
source venv/bin/activate
python3 laugh_line.py

Try with this John Mulaney video: https://www.youtube.com/watch?v=DtBkEJU1zr0

Order of operations:
1. Download video using yt-dlp
2. Extract audio from video using ffmpeg
3. Send audio to Whisper API for transcription
4. Return transcript with timestamps to user
5. Use an LLM to summarize the funniest parts and generate clip highlights

TODO:
- Check if video is longer than 10 minutes BEFORE downloading
- Implement actual Whisper API transcription instead of placeholder
- Do basic error handling and edge cases
"""

from flask import Flask, render_template, request, jsonify
import yt_dlp
import os
import re
import time
import subprocess

app = Flask(__name__)

MAX_VIDEO_DURATION = 10 * 60  # 10 minutes in seconds
DOWNLOADS_DIR = "downloads" 

if not os.path.exists(DOWNLOADS_DIR):
    os.makedirs(DOWNLOADS_DIR)

def get_video_id(url):
    """Extract YouTube video ID from URL
    The video ID is the end bit at the end of a YouTube URL.
    In the example URL "https://www.youtube.com/watch?v=DtBkEJU1zr0", the video ID is "DtBkEJU1zr0".
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

def create_transcript(audio_path):
    """Create transcript from audio file using Whisper API"""
    print("create_transcript() called with:", audio_path)

    # TODO: Implement actual transcription with Whisper API
    # For now, return placeholder transcript

    placeholder_transcript = [
        {'start_time': 0, 'text': 'Audio download successful!'},
        {'start_time': 5, 'text': 'If you made it here, the video has been downloaded.'},
        {'start_time': 10, 'text': 'Check the downloads directory in the Yapper parent directory.'},
        {'start_time': 15, 'text': 'Next step: implement Whisper transcription.'}
    ]

    return placeholder_transcript

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
            audio_path = video_result['audio_path']  # Already extracted in download_video()
        except Exception as e:
            return jsonify({'error': 'Audio extraction failed: ' + str(e)}), 400

        # print("*** Step 3: CREATE TRANSCRIPT")
        try:
            transcript = create_transcript(audio_path)
        except Exception as e:
            return jsonify({'error': 'Transcription failed: ' + str(e)}), 400
        
        return jsonify({
            'success': True,
            'video_info': video_info,
            'transcript': transcript
        })

    except Exception as e:
        return jsonify({'error': 'Unexpected error: ' + str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5050)