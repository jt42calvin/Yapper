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
7. User feedback system to refine highlights

TODO:
- Ensure basic error handling and edge cases for all cases
- Adjust the transcription generation so that each sentence is its own segment with a start time
    - This will help with the "more context next" feature, which will jump to the next sentence
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import yt_dlp
import os
import re
import json
import subprocess
import whisper
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
whisper_model = whisper.load_model("base")

MAX_VIDEO_DURATION = 10 * 60  # 10 minutes in seconds
DOWNLOADS_DIR = "downloads" 
FEEDBACK_DIR = "feedback"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma3:1b-it-qat"

# Create necessary directories
for directory in [DOWNLOADS_DIR, FEEDBACK_DIR, "transcripts", "highlights"]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Store current session data (video_id -> session info)
current_sessions = {}

def get_video_id(url):
    """Extract YouTube video ID from URL"""
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def load_feedback(video_id):
    """Load feedback JSON for a video"""
    feedback_path = os.path.join(FEEDBACK_DIR, f"{video_id}_feedback.json")
    if os.path.exists(feedback_path):
        with open(feedback_path, 'r') as f:
            return json.load(f)
    return {
        "video_id": video_id,
        "feedback_history": [],
        "flagged_sections": []
    }

def save_feedback(video_id, feedback_data):
    """Save feedback JSON for a video"""
    feedback_path = os.path.join(FEEDBACK_DIR, f"{video_id}_feedback.json")
    with open(feedback_path, 'w') as f:
        json.dump(feedback_data, indent=2, fp=f)

def download_video(url):
    """Download YouTube video as .mp4 (≤1080p), then extract audio as .mp3 in the same directory."""
    video_id = get_video_id(url)
    if not video_id:
        raise Exception("Invalid YouTube URL format")

    output_video = os.path.join(DOWNLOADS_DIR, f"{video_id}_video.mp4")
    output_audio = os.path.join(DOWNLOADS_DIR, f"{video_id}_audio.mp3")

    # print("Video ID:", video_id)

    if os.path.exists(output_video):
        # print("Using cached video:", output_video)
        if not os.path.exists(output_audio):
            extract_audio(output_video, output_audio)
        with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
            info = ydl.extract_info(url, download=False)
    else:
        ydl_opts = {
            "format": "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4][height<=1080]",
            "outtmpl": output_video,
            "quiet": True,
            "cookiefile": None,
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "extractor_args": {
                "youtube": {
                    "player_client": ["android", "web"],
                    "skip": ["hls", "dash"]
                }
            },
            "http_headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-us,en;q=0.5",
                "Sec-Fetch-Mode": "navigate"
            }
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # print("Downloading video...")
            info = ydl.extract_info(url, download=True)

        extract_audio(output_video, output_audio)

    if info.get("duration", 0) > MAX_VIDEO_DURATION:
        raise Exception("Video is too long (max 10 minutes)")

    return {
        "video_path": output_video,
        "audio_path": output_audio,
        "video_id": video_id,
        "title": info.get("title", "Unknown"),
        "duration": info.get("duration", 0),
        "uploader": info.get("uploader", "Unknown"),
    }

def extract_audio(video_path, audio_path):
    """Extract audio from video using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "mp3",
        audio_path,
    ]
    subprocess.run(cmd, check=True)

def generate_transcript_with_whisper(audio_path):
    """Generate transcript from audio file using local Whisper model"""
    if not os.path.exists(audio_path):
        raise Exception("Audio file not found: " + audio_path)

    try:
        # print("Transcribing audio with local Whisper...")
        result = whisper_model.transcribe(audio_path)
        
        formatted_transcript = []
        for segment in result["segments"]:
            formatted_transcript.append({
                'start_time': int(segment['start']),
                'text': segment['text'].strip()
            })

        # print("Generated transcript with " + str(len(formatted_transcript)) + " segments")
        return formatted_transcript
        
    except Exception as e:
        print("Error generating transcript: " + str(e))
        raise Exception("Whisper transcription failed: " + str(e))

def call_ollama(prompt, model=OLLAMA_MODEL):
    """Call Ollama API to analyze newly generated transcript"""
    try:
        # print(f"Calling Ollama with model: {model}")
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 500
                }
            },
            timeout=60
        )

        if response.status_code == 200:
            return response.json()["response"]
        else:
            # print(f"Ollama API error: {response.status_code} - {response.text}")
            return None

    except requests.exceptions.ConnectionError:
        print("Could not connect to Ollama. Make sure it's running on localhost:11434")
        return None
    except requests.exceptions.Timeout:
        print("Ollama request timed out.")
        return None
    except Exception as e:
        print("Error calling Ollama: " + str(e))
        return None

def analyze_transcript_with_ollama(transcript_text, exclude_sections=None):
    """Use Ollama to identify the funniest moment, excluding flagged sections"""
    
    exclude_info = ""
    if exclude_sections and len(exclude_sections) > 0:
        exclude_info = "\n\nDo NOT select moments within these time ranges (already flagged):\n"
        for section in exclude_sections:
            exclude_info += f"- {section['start']}s to {section['end']}s\n"
    
    prompt = f"""Analyze this comedy video transcript and identify the single funniest moment. 

    Return ONLY a JSON array with these exact fields:
    - description: Brief description of the funniest moment
    - start_time: Start time in seconds (integer)
    - end_time: End time in seconds (integer)
    - quote: Exact quote from transcript

    Funny moment should be greater than 25 seconds long and include context before the joke.
    {exclude_info}

    Example format:
    [{{"description": "A hilarious joke about cats", "start_time": 100, "end_time": 130, "quote": "Why did the cat sit on the computer?"}}]

    Transcript:
    {transcript_text}

    Response (JSON only):"""
    
    analysis = call_ollama(prompt)
    if analysis:
        try:
            analysis = analysis.strip()
            
            if "```json" in analysis:
                analysis = analysis.split("```json")[1].split("```")[0]
            elif "```" in analysis:
                analysis = analysis.split("```")[1].split("```")[0]
            
            start_bracket = analysis.find('[')
            end_bracket = analysis.rfind(']') + 1
            
            if start_bracket != -1 and end_bracket != 0:
                analysis = analysis[start_bracket:end_bracket]
            
            return json.loads(analysis.strip())
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response from Ollama: {e}")
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
    # print("create_transcript() called with:", audio_path)

    try:
        transcript = generate_transcript_with_whisper(audio_path)
        return {"transcript": transcript}

    except Exception as e:
        print("Transcript generation failed: " + str(e))
        placeholder_transcript = [
            {'start_time': 0, 'text': 'Audio download successful!'},
            {'start_time': 5, 'text': 'Local Whisper transcription failed.'},
            {'start_time': 10, 'text': 'Check the downloads directory.'},
            {'start_time': 15, 'text': 'Error: ' + str(e)}
        ]
        return {"transcript": placeholder_transcript}

def create_transcript_txt(transcript):
    """Create a plaintext .txt file version of the transcript"""
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
    
    # print(f"Transcript saved to {txt_path}")
    return txt_path

def create_highlight_video(video_path, start_time, end_time, video_id, suffix="highlight"):
    """Create a trimmed .mp4 file using ffmpeg"""
    if not os.path.exists("highlights"):
        os.makedirs("highlights")

    highlight_path = os.path.join("highlights", f"{video_id}_{suffix}.mp4")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-ss", str(start_time),
        "-to", str(end_time + 1),
        "-c", "copy",
        highlight_path,
    ]
    subprocess.run(cmd, check=True)

    # print(f"Highlight video saved to {highlight_path}")
    return highlight_path

def find_next_transcript_segment(transcript, current_end_time):
    """Find the next sentence/segment in the transcript after current_end_time"""
    for segment in transcript:
        if segment['start_time'] > current_end_time:
            return segment['start_time']
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/highlights/<path:filename>')
def serve_highlight(filename):
    """Serve highlight videos from the highlights directory"""
    return send_from_directory('highlights', filename)

@app.route('/downloads/<path:filename>')
def serve_download(filename):
    """Serve files from the downloads directory"""
    return send_from_directory('downloads', filename)

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
                'path': video_result['video_path'],
                'video_id': video_result['video_id']
            }
        except Exception as e:
            return jsonify({'error': str(e)}), 400

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
            # print("Transcript saved at:", transcript_txt_path)
        except Exception as e:
            return jsonify({'error': 'Unexpected error: ' + str(e)}), 500

        print("*** Step 4: ANALYZE TRANSCRIPT WITH OLLAMA")
        try:
            feedback_data = load_feedback(video_info['video_id'])
            transcript_text = " ".join([seg['text'] for seg in transcript])
            analysis = analyze_transcript_with_ollama(
                transcript_text, 
                exclude_sections=feedback_data.get('flagged_sections', [])
            )
            # print("Ollama analysis:", analysis)
            
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
                funny_moment['end_time'],
                video_info['video_id']
            )
            # print("Highlight video created at:", highlight_path)
            video_info['highlight_filename'] = f"{video_info['video_id']}_highlight.mp4"
            video_info['funny_moment'] = funny_moment
            
            # Store session data for feedback
            current_sessions[video_info['video_id']] = {
                'video_path': video_info['path'],
                'transcript': transcript,
                'current_clip': {
                    'start': funny_moment['start_time'],
                    'end': funny_moment['end_time']
                }
            }
            
        except Exception as e:
            print(f"Highlight video creation failed: {e}")
            video_info['highlight_error'] = str(e)

        return jsonify({
            'success': True,
            'video_info': video_info,
            'transcript': transcript,
            'analysis': analysis
        })
    except Exception as e:
        return jsonify({'error': 'Unexpected error: ' + str(e)}), 500

@app.route('/feedback/more-context-beginning', methods=['POST'])
def feedback_more_context_beginning():
    """Extend the clip by adding 10 seconds at the beginning"""
    try:
        data = request.json
        video_id = data.get('video_id')
        
        if not video_id or video_id not in current_sessions:
            return jsonify({'error': 'No active session for this video'}), 400
        
        session = current_sessions[video_id]
        current_clip = session['current_clip']
        
        # Extend beginning by 10 seconds (but not below 0)
        new_start = max(0, current_clip['start'] - 10)
        new_end = current_clip['end']
        
        # Create new highlight
        highlight_path = create_highlight_video(
            session['video_path'],
            new_start,
            new_end,
            video_id,
            suffix="highlight"
        )
        
        # Update session
        session['current_clip'] = {'start': new_start, 'end': new_end}
        
        # Save feedback
        feedback_data = load_feedback(video_id)
        feedback_data['feedback_history'].append({
            'timestamp': datetime.now().isoformat(),
            'type': 'more_context_beginning',
            'original_clip': current_clip,
            'new_clip': {'start': new_start, 'end': new_end}
        })
        save_feedback(video_id, feedback_data)
        
        return jsonify({
            'success': True,
            'new_clip': {'start': new_start, 'end': new_end},
            'highlight_filename': f"{video_id}_highlight.mp4"
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/feedback/more-context-next', methods=['POST'])
def feedback_more_context_next():
    """Jump to the next sentence in the transcript"""
    try:
        data = request.json
        video_id = data.get('video_id')
        
        if not video_id or video_id not in current_sessions:
            return jsonify({'error': 'No active session for this video'}), 400
        
        session = current_sessions[video_id]
        current_clip = session['current_clip']
        transcript = session['transcript']
        
        # Find next segment
        next_start = find_next_transcript_segment(transcript, current_clip['end'])
        
        if next_start is None:
            return jsonify({'error': 'No more content after current clip'}), 400
        
        # Extend to include next segment (add ~15 seconds for context)
        new_start = current_clip['start']
        new_end = next_start + 15
        
        # Create new highlight
        highlight_path = create_highlight_video(
            session['video_path'],
            new_start,
            new_end,
            video_id,
            suffix="highlight"
        )
        
        # Update session
        session['current_clip'] = {'start': new_start, 'end': new_end}
        
        # Save feedback
        feedback_data = load_feedback(video_id)
        feedback_data['feedback_history'].append({
            'timestamp': datetime.now().isoformat(),
            'type': 'more_context_next',
            'original_clip': current_clip,
            'new_clip': {'start': new_start, 'end': new_end}
        })
        save_feedback(video_id, feedback_data)
        
        return jsonify({
            'success': True,
            'new_clip': {'start': new_start, 'end': new_end},
            'highlight_filename': f"{video_id}_highlight.mp4"
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/feedback/not-funny', methods=['POST'])
def feedback_not_funny():
    """Find the next funniest moment in the video"""
    try:
        data = request.json
        video_id = data.get('video_id')
        
        if not video_id or video_id not in current_sessions:
            return jsonify({'error': 'No active session for this video'}), 400
        
        session = current_sessions[video_id]
        current_clip = session['current_clip']
        transcript = session['transcript']
        
        # Load feedback and add current clip to flagged sections
        feedback_data = load_feedback(video_id)
        feedback_data['flagged_sections'].append({
            'start': current_clip['start'],
            'end': current_clip['end'],
            'reason': 'not_funny'
        })
        
        # Re-analyze with exclusions
        transcript_text = " ".join([seg['text'] for seg in transcript])
        analysis = analyze_transcript_with_ollama(
            transcript_text,
            exclude_sections=feedback_data['flagged_sections']
        )
        
        if not analysis or len(analysis) == 0:
            return jsonify({'error': 'No more funny moments found'}), 400
        
        funny_moment = analysis[0]
        
        # Create new highlight
        highlight_path = create_highlight_video(
            session['video_path'],
            funny_moment['start_time'],
            funny_moment['end_time'],
            video_id,
            suffix="highlight"
        )
        
        # Update session
        session['current_clip'] = {
            'start': funny_moment['start_time'],
            'end': funny_moment['end_time']
        }
        
        # Save feedback
        feedback_data['feedback_history'].append({
            'timestamp': datetime.now().isoformat(),
            'type': 'not_funny',
            'original_clip': current_clip,
            'new_clip': session['current_clip']
        })
        save_feedback(video_id, feedback_data)
        
        return jsonify({
            'success': True,
            'new_clip': session['current_clip'],
            'funny_moment': funny_moment,
            'highlight_filename': f"{video_id}_highlight.mp4"
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/feedback/not-appropriate', methods=['POST'])
def feedback_not_appropriate():
    """Flag current clip as inappropriate"""
    try:
        data = request.json
        video_id = data.get('video_id')
        
        if not video_id or video_id not in current_sessions:
            return jsonify({'error': 'No active session for this video'}), 400
        
        session = current_sessions[video_id]
        current_clip = session['current_clip']
        
        # Load feedback and flag section
        feedback_data = load_feedback(video_id)
        feedback_data['flagged_sections'].append({
            'start': current_clip['start'],
            'end': current_clip['end'],
            'reason': 'not_appropriate'
        })
        
        feedback_data['feedback_history'].append({
            'timestamp': datetime.now().isoformat(),
            'type': 'not_appropriate',
            'flagged_clip': current_clip
        })
        
        save_feedback(video_id, feedback_data)
        
        return jsonify({
            'success': True,
            'message': 'Section flagged as inappropriate'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5050)