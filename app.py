from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os
import uuid
from werkzeug.utils import secure_filename
import cv2
import anthropic
from datetime import datetime
import json
import threading
from queue import Queue
import base64
import subprocess
import shutil
import soundfile as sf
import requests

#Import the processing modules
import audio_processor
import image_processor

from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
FRAMES_FOLDER = 'extracted/frames'
RESULTS_FOLDER = 'results'
SAMPLE_RATE_SECONDS = 5
AUDIO_CLIP_DURATION_SECONDS = 3
AUDIO_SAMPLE_RATE = 44100 # The sample rate to work with
POKE_API_KEY = os.getenv('POKE_API_KEY', 'your-poke-api-key-here')
POKE_API_URL = 'https://poke.com/api/v1/inbound-sms/webhook'

# Create directories if they don't exist
for folder in [UPLOAD_FOLDER, FRAMES_FOLDER, RESULTS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Allowed video extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}

# Initialize Anthropic client
anthropic_client = anthropic.Anthropic(
    api_key=os.getenv('ANTHROPIC_API_KEY', 'your-api-key-here')
)

# Global processing queue
processing_queue = Queue()
processing_results = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_frames_simple(video_path, output_folder, frame_rate=1):
    """Extract frames from video using OpenCV only"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / frame_rate)
    
    frame_count = 0
    extracted_frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps
            frame_filename = f"frame_{int(timestamp):04d}.jpg"
            frame_path = os.path.join(output_folder, frame_filename)
            cv2.imwrite(frame_path, frame)
            extracted_frames.append({
                'timestamp': timestamp,
                'frame_path': frame_path,
                'frame_filename': frame_filename
            })
        
        frame_count += 1
    
    cap.release()
    return extracted_frames

def format_timestamp(seconds):
    """Converts seconds into HH-MM-SS format for filenames."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}-{m:02d}-{s:02d}"

def extract_full_audio_track(video_path, temp_audio_path):
    """
    Uses ffmpeg to extract the full audio track to a temporary WAV file.
    This is much more robust than library-based in-memory extraction.
    Returns True on success, False on failure.
    """
    print("Extracting full audio track with ffmpeg...")
    command = [
        'ffmpeg',
        '-i', video_path,
        '-vn',            # No video
        '-acodec', 'pcm_s16le', # Use standard WAV codec
        '-ar', str(AUDIO_SAMPLE_RATE), # Set audio sample rate
        '-ac', '1',       # Set to mono
        '-y',             # Overwrite output file if it exists
        temp_audio_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("Audio extraction successful.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: ffmpeg command failed. Is ffmpeg installed and in your system's PATH?")
        return False

def analyze_frame_emotion(frame_path):
    """Analyze emotions in a frame using Anthropic API"""
    try:
        with open(frame_path, 'rb') as image_file:
            image_data = image_file.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        message = anthropic_client.messages.create(
            model="claude-3-5-haiku-20241022",  # Fast model with high rate limits
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": "Analyze the emotions in this image. Return JSON with emotion scores 0-10 for: joy, sadness, anger, fear, surprise, disgust, neutral. Format: {\"emotions\": {\"joy\": 5, \"sadness\": 2}, \"description\": \"what you see\"}"
                        }
                    ]
                }
            ]
        )
        
        response_text = message.content[0].text
        return {"visual_analysis": response_text}
        
    except Exception as e:
        print(f"Error analyzing frame emotion: {e}")
        return {"error": str(e)}
    
def send_to_poke(message):
    """Send a message to Poke API"""
    try:
        response = requests.post(
            POKE_API_URL,
            headers={
                'Authorization': f'Bearer {POKE_API_KEY}',
                'Content-Type': 'application/json'
            },
            json={'message': message}
        )
        return response.json()
    except Exception as e:
        print(f"Error sending to Poke: {e}")
        return {"error": str(e)}
    
def analyze_video_feedback(video_data):
    """Analyze all visual analysis data and provide constructive feedback"""
    try:
        # Extract all visual analyses
        visual_analyses = []
        emotion_scores = []
        
        for result in video_data.get('results', []):
            if 'frame_emotion' in result and 'visual_analysis' in result['frame_emotion']:
                visual_analyses.append(result['frame_emotion']['visual_analysis'])
                # Try to extract emotion scores if in JSON format
                try:
                    emotion_data = json.loads(result['frame_emotion']['visual_analysis'])
                    if 'emotions' in emotion_data:
                        emotion_scores.append(emotion_data['emotions'])
                except:
                    pass
        
        # Create comprehensive analysis prompt
        prompt = f"""
        A Flask-based web application that analyzes emotions from videos collected by the Mentra Live glass, using computer vision, audio processing, and AI-powered feedback generation. Built for HackMIT 2025 with integration to Anthropic's LLM Model - Claude and Poke's automation platform. We want Poke to interpret the visual notes collected by the Claude model so that it can provide constructive criticism for behavior, and facial reactions so that people can feel more comfortable in social interactions and can grow individually.
        Visual Analysis Data: {visual_analyses}

        Please provide:
        1. POSITIVE ASPECTS: What the person does well emotionally (genuine smiles, engaging expressions, confident body language, etc.)
        2. CONSTRUCTIVE FEEDBACK: Areas for improvement in emotional expression and presentation (maintaining consistent energy, reducing nervous expressions, etc.)
        3. OVERALL IMPRESSION: Summary of their emotional presence and charisma
        4. ACTIONABLE TIPS: Specific suggestions for improving their on-camera presence

        Keep feedback supportive and encouraging while being honest about areas for growth. Focus on presentation skills and emotional communication.
        """
        
        message = anthropic_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=800,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        feedback = message.content[0].text
        
        # Send feedback to Poke
        poke_message = f"Video Analysis Feedback:\n\n{feedback}"
        poke_response = send_to_poke(poke_message)
        
        return {
            "feedback": feedback,
            "poke_response": poke_response,
            "total_frames_analyzed": len(visual_analyses)
        }
        
    except Exception as e:
        print(f"Error analyzing video feedback: {e}")
        return {"error": str(e)}

def process_video_enhanced(video_id, video_path):
    """Adapted from your teammate's process_local_video function"""
    try:
        # Create unique folder for this video (similar to your teammate's setup)
        video_output_dir = os.path.join(RESULTS_FOLDER, video_id)
        image_output_dir = os.path.join(video_output_dir, "processed_images")
        audio_output_dir = os.path.join(video_output_dir, "processed_audio")
        temp_dir = os.path.join(video_output_dir, "temp")
        
        # Clean up if exists (your teammate's approach)
        if os.path.exists(video_output_dir):
            print(f"Output directory {video_output_dir} already exists. Removing it.")
            shutil.rmtree(video_output_dir)
        
        os.makedirs(image_output_dir, exist_ok=True)
        os.makedirs(audio_output_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)

        # Extract full audio track (your teammate's method)
        temp_audio_path = os.path.join(temp_dir, "full_audio.wav")
        audio_success = extract_full_audio_track(video_path, temp_audio_path)

        # Set up for sampling (your teammate's approach)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"Video FPS: {fps:.2f}, Duration: {duration:.2f}s")
        
        results = []
        last_image_hash = None
        
        # Main sampling loop (adapted from your teammate's code)
        for t in range(0, int(duration), SAMPLE_RATE_SECONDS):
            timestamp_str = format_timestamp(t)
            print(f"\n--- Processing sample at {t}s ({timestamp_str}) ---")
            
            segment_result = {
                'timestamp': t,
                'timestamp_str': timestamp_str
            }
            
            # A. Process a Single Video Frame (your teammate's approach)
            frame_id = int(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            
            if ret:
                image_data = image_processor.process_image_frame_from_memory(frame)
                if image_data:
                    current_hash = image_data['features']['perceptual_hash']
                    if current_hash == last_image_hash:
                        print("Skipping redundant image frame.")
                        segment_result['frame_emotion'] = {"skipped": "duplicate_frame"}
                    else:
                        last_image_hash = current_hash
                        image_savename = f"frame_{timestamp_str}.jpg"
                        image_savepath = os.path.join(image_output_dir, image_savename)
                        cv2.imwrite(image_savepath, image_data['processed_image'])
                        
                        # Use your existing emotion analysis
                        frame_emotion = analyze_frame_emotion(image_savepath)
                        segment_result['frame_emotion'] = frame_emotion
                        segment_result['image_features'] = image_data['features']
                        
                        print(f"Saved processed image: {image_savename}")

            # B. Process Audio Clip (your teammate's approach)
            if audio_success:
                try:
                    start_sample = int(t * AUDIO_SAMPLE_RATE)
                    end_sample = int((t + AUDIO_CLIP_DURATION_SECONDS) * AUDIO_SAMPLE_RATE)
                    
                    # Read the sample directly from the temporary WAV file
                    audio_array, _ = sf.read(temp_audio_path, start=start_sample, stop=end_sample, dtype='float32')

                    audio_data = audio_processor.process_audio_clip_from_numpy(audio_array, AUDIO_SAMPLE_RATE)
                    
                    if audio_data:
                        audio_savename = f"clip_{timestamp_str}.wav"
                        audio_savepath = os.path.join(audio_output_dir, audio_savename)
                        
                        sf.write(audio_savepath, audio_data['processed_audio_array'], audio_data['sample_rate'])
                        
                        segment_result['audio_features'] = audio_data['features']
                        print(f"Saved processed audio clip: {audio_savename}")

                except Exception as e:
                    print(f"Could not process audio clip at {t}s: {e}")
            
            results.append(segment_result)

        # Clean up and finalize (your teammate's approach)
        cap.release()
        shutil.rmtree(temp_dir) # Remove temporary audio file and folder
        
        # Save results (your existing format)
        results_file = os.path.join(RESULTS_FOLDER, f"{video_id}_analysis.json")
        with open(results_file, 'w') as f:
            json.dump({
                'video_id': video_id,
                'processed_at': datetime.now().isoformat(),
                'total_segments': len(results),
                'results': results
            }, f, indent=2)
        
        processing_results[video_id] = {
            'status': 'completed',
            'results_file': results_file,
            'total_segments': len(results)
        }
        
        print(f"Video {video_id} processing completed!")
        
    except Exception as e:
        print(f"Error processing video {video_id}: {e}")
        processing_results[video_id] = {
            'status': 'error',
            'error': str(e)
        }


def worker():
    """Background worker to process videos"""
    while True:
        video_id, video_path = processing_queue.get()
        processing_results[video_id] = {'status': 'processing'}
        process_video_enhanced(video_id, video_path)
        processing_queue.task_done()

# Start background worker
worker_thread = threading.Thread(target=worker, daemon=True)
worker_thread.start()

# Simple HTML template
MAIN_PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>ReLive! - AI-Powered Video Emotion Analysis</title>
    <style>
        :root {
            --bg-primary: #FFFAF2;
            --border-color: #b1b0fe;
            --window-bg: #c19acb;
            --cream: #FFFDD0;
            --light-grey: #e8e8e8;
            --gradient-pink-green: linear-gradient(to right, #f56ebd, #97f0b6);
            --gradient-red-lime: linear-gradient(to right, #fd5a47, #c0e264);
            --gradient-purple-blue: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        * { box-sizing: border-box; margin: 0; padding: 0; }
        html, body { height: 100%; font-family: 'Nunito', sans-serif; font-weight: 550; }
        
        body {
            background: linear-gradient(135deg, var(--bg-primary) 0%, #f8f4e9 100%);
            color: #333;
            overflow-x: hidden;
        }
        
        header.appbar {
            height: 80px;
            display: flex; align-items: center; justify-content: space-between;
            padding: 0 2rem;
            border-bottom: 2px solid var(--border-color);
            background: var(--light-grey);
            position: fixed; left: 0; right: 0; top: 0; z-index: 1000;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        header .app-title {
            font-weight: 700;
            color: #333;
            font-size: 2.5rem;
        }
        
        .nav-links {
            display: flex;
            gap: 2rem;
            align-items: center;
        }
        
        .nav-links a {
            text-decoration: none;
            color: #333;
            font-weight: 600;
            transition: color 0.3s ease;
        }
        
        .nav-links a:hover {
            color: #f56ebd;
        }
        
        .btn {
            padding: 0.8rem 1.5rem;
            border-radius: 0.75rem;
            background: var(--gradient-red-lime);
            color: white; border: none; font-weight: 700;
            font-size: 1rem; cursor: pointer;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            transition: all 0.2s ease;
            font-family: 'Nunito', sans-serif;
            text-decoration: none; display: inline-block;
            letter-spacing: 0.5px;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.2);
        }
        
        .btn-secondary {
            background: var(--gradient-pink-green);
        }
        
        main {
            margin-top: 80px;
            min-height: calc(100vh - 80px);
        }
        
        /* Hero Section */
        .hero-section {
            min-height: 60vh;
            display: flex;
            align-items: center;
            padding: 4rem 2rem;
            background: var(--gradient-purple-blue);
            color: white;
            text-align: center;
        }
        
        .hero-content {
            flex: 1;
            max-width: 800px;
            margin: 0 auto;
        }
        
        .hero-content h1 {
            font-size: 4rem;
            margin-bottom: 1.5rem;
            font-weight: 700;
        }
        
        .mission-statement {
            font-size: 1.3rem;
            line-height: 1.6;
            margin-bottom: 2rem;
            opacity: 0.95;
        }
        
        /* Video Section */
        .video-section {
            padding: 4rem 2rem;
            display: flex;
            gap: 3rem;
            align-items: strech;
            max-width: 1600px;
            margin: 0 auto;
        }
        
        .demo-video {
            flex: 3;
        }
        
        .other-videos {
            flex: 2;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }
        
        .video-container {
            position: relative;
            background: var(--window-bg);
            border-radius: 1rem;
            overflow: hidden;
            box-shadow: 0 10px 25px rgba(0,0,0,0.15);
            border: 2px solid var(--border-color);
        }
        
        .video-header {
            background: var(--gradient-pink-green);
            color: white;
            padding: 1rem;
            font-weight: 700;
            text-align: center;
        }
        
        .video-wrapper {
            padding: 1rem;
            aspect-ratio: 16/9;
            background: #000;
        }
        
        .video-wrapper iframe,
        .video-wrapper video {
            width: 100%;
            height: 100%;
            border: none;
            border-radius: 0.5rem;
        }
        
        /* Before/After Comparison Sections */
        .comparison-section {
            padding: 4rem 2rem;
            background: linear-gradient(135deg, #f8f4e9 0%, white 100%);
        }
        
        .section-title {
            text-align: center;
            font-size: 3rem;
            color: #333;
            margin-bottom: 1rem;
            font-weight: 700;
        }
        
        .section-subtitle {
            text-align: center;
            font-size: 1.2rem;
            color: #666;
            margin-bottom: 3rem;
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
        }
        
        .comparison-container {
            max-width: 1400px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 3rem;
            margin-bottom: 4rem;
        }
        
        .comparison-item {
            background: white;
            border-radius: 1rem;
            overflow: hidden;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
            border: 2px solid var(--border-color);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .comparison-item:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(0,0,0,0.15);
        }
        
        .comparison-header {
            padding: 1.5rem;
            font-weight: 700;
            font-size: 1.3rem;
            text-align: center;
            color: white;
        }
        
        .before-header {
            background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        }
        
        .after-header {
            background: linear-gradient(135deg, #00b894, #00cec9);
        }
        
        .comparison-content {
            padding: 2rem;
        }
        
        .video-comparison {
            aspect-ratio: 16/9;
            width: 100%;
            border-radius: 0.5rem;
            overflow: hidden;
            background: #000;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .video-comparison video {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        
        .video-comparison img {
            width: 100%;
            height: 100%;
            object-fit: contain;
            background: #f5f5f5;
        }
        
        .audio-comparison {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }
        
        .waveform-container {
            background: #f8f9fa;
            border-radius: 0.5rem;
            padding: 1rem;
            border: 1px solid #e9ecef;
        }
        
        .waveform-visual {
            width: 100%;
            height: 80px;
            background: linear-gradient(90deg, #ddd 0%, #999 50%, #ddd 100%);
            border-radius: 0.25rem;
            margin-bottom: 1rem;
            position: relative;
            overflow: hidden;
        }
        
        .waveform-visual.before::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: repeating-linear-gradient(
                90deg,
                #ff6b6b 0px,
                #ff6b6b 2px,
                transparent 2px,
                transparent 8px
            );
            opacity: 0.7;
        }
        
        .waveform-visual.after::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: repeating-linear-gradient(
                90deg,
                #00b894 0px,
                #00b894 1px,
                transparent 1px,
                transparent 4px
            );
            opacity: 0.8;
        }
        
        .audio-info {
            text-align: center;
        }
        
        .audio-info h4 {
            margin: 0 0 0.5rem 0;
            color: #333;
        }
        
        .audio-info p {
            margin: 0;
            color: #666;
            font-size: 0.9rem;
        }
        
        /* Call-to-Action Section */
        .cta-section {
            padding: 4rem 2rem;
            background: var(--gradient-red-lime);
            color: white;
            text-align: center;
        }
        
        .cta-section h2 {
            font-size: 2.5rem;
            margin-bottom: 1rem;
        }
        
        .cta-section p {
            font-size: 1.2rem;
            margin-bottom: 2rem;
            opacity: 0.95;
        }
        
        .btn-white {
            background: white;
            color: #333;
            font-weight: 700;
        }
        
        .btn-white:hover {
            background: #f0f0f0;
        }
        
        /* Responsive Design */
        @media (max-width: 1200px) {
            .video-section {
                flex-direction: column;
            }
            
            .comparison-container {
                grid-template-columns: 1fr;
            }
        }
        
        @media (max-width: 768px) {
            .hero-content h1 {
                font-size: 2.5rem;
            }
            
            .other-videos {
                gap: 1rem;
            }
            
            .section-title {
                font-size: 2.2rem;
            }
        }
    </style>
</head>
<body>
    <header class="appbar">
        <div style="display: flex; align-items: center; gap: 1rem;">
            <img src="/static/logo/darklines.jpg" alt="ReLive Logo" style="height: 50px; width: auto;">
            <div class="app-title">ReLive!</div>
        </div>
        <nav class="nav-links">
            <a href="/demo" class="btn">Try It Now!</a>
        </nav>
    </header>

    <main>
        <!-- Hero Section with Problem Statement -->
        <section class="hero-section">
            <div class="hero-content">
                <h1>RePlay Moments</h1>
                <p class="mission-statement">
                    Every young adult has moments of self-doubt and social anxiety, especially individuals with ADHD and Autism. We are trying to alleviate the social anxiety of interactions by recording and classifying social events throughout your day, so you don't have to memorize or stress about every moment.
                </p>
                <a href="/demo" class="btn">Start Your Analysis</a>
            </div>
        </section>

        <!-- Video Demo Section -->
        <section class="video-section">
            <div class="demo-video">
                <div class="video-container">
                    <div class="video-header">🎬 Main Demo - See ReLive! in Action</div>
                    <div class="video-wrapper">
                        <iframe src="https://www.youtube.com/embed/dQw4w9WgXcQ" allowfullscreen></iframe>
                    </div>
                </div>
            </div>
            
            <div class="other-videos">
                <div class="video-container">
                    <div class="video-header">📊 Video 1 by Mentra Camere</div>
                    <div class="video-wrapper">
                        <video controls>
                            <source src="/static/modules/Video_1.mp4" type="video/mp4">
                            <source src="/static/modules/Video_2.mp4" type="video/mp4">
                            Your browser does not support the video tag.
                        </video>
                    </div>
                </div>
                
                <div class="video-container">
                    <div class="video-header">💡 Video 2 by Mentra Camere</div>
                    <div class="video-wrapper">
                        <video controls>
                            <source src="/static/modules/Video_2.mp4" type="video/mp4">
                            <source src="/static/modules/Video_2.webm" type="video/webm">
                            Your browser does not support the video tag.
                        </video>
                    </div>
                </div>
            </div>
        </section>

        <!-- Before/After Comparison Section -->
        <section class="comparison-section">
            <h2 class="section-title">🎵 Experience Our Advanced Filtration</h2>
            <p class="section-subtitle">See and hear the dramatic difference our AI-powered audio and video enhancement makes</p>
            
            <!-- Audio Comparison -->
            <div class="comparison-container">
                <div class="comparison-item">
                    <div class="comparison-header before-header">
                        🔊 Original Audio - Before Processing
                    </div>
                    <div class="comparison-content">
                        <div class="audio-comparison">
                            <div class="waveform-container">
                                <div class="waveform-visual before"></div>
                                <div class="audio-info">
                                    <h4>Raw Audio Sample</h4>
                                    <p>Background noise, static, distortion present</p>
                                </div>
                            </div>
                            <audio controls style="width: 100%; margin-top: 1rem;">
                                <source src="/static/audio/audio_raw_00-00-20.wav" type="audio/wav">
                                <source src="/static/audio/sample_before.mp3" type="audio/mpeg">
                                Your browser does not support the audio element.
                            </audio>
                        </div>
                    </div>
                </div>
                
                <div class="comparison-item">
                    <div class="comparison-header after-header">
                        ✨ Enhanced Audio - After Processing
                    </div>
                    <div class="comparison-content">
                        <div class="audio-comparison">
                            <div class="waveform-container">
                                <div class="waveform-visual after"></div>
                                <div class="audio-info">
                                    <h4>AI-Enhanced Audio</h4>
                                    <p>Crystal clear, noise-free, optimized</p>
                                </div>
                            </div>
                            <audio controls style="width: 100%; margin-top: 1rem;">
                                <source src="/static/audio/audio_processed_00-00-20.wav" type="audio/wav">
                                <source src="/static/audio/sample_after.mp3" type="audio/mpeg">
                                Your browser does not support the audio element.
                            </audio>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Video/Image Comparison -->
            <div class="comparison-container">
                <div class="comparison-item">
                    <div class="comparison-header before-header">
                        📹 Original Frame - Before Processing
                    </div>
                    <div class="comparison-content">
                        <div class="video-comparison">
                            <img src="/static/video/preprocesspic.png" alt="Before processing frame">
                        </div>
                    </div>
                </div>
                
                <div class="comparison-item">
                    <div class="comparison-header after-header">
                        ✨ Enhanced Frame - After Processing
                    </div>
                    <div class="comparison-content">
                        <div class="video-comparison">
                            <img src="/static/video/frame_00-00-05.jpg" alt="After processing frame">
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Call to Action Section -->
        <section class="cta-section">
            <h2>Ready to Transform Your Communication?</h2>
            <p>Be one of the first to utilize ReLive! </p>
            <a href="/demo" class="btn btn-white">Start Your Free Analysis</a>
        </section>
    </main>

    <script>
        // Smooth scrolling for navigation links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });
    </script>
</body>
</html>
"""

UPLOAD_HTML = """
<html>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Video Emotion Analysis — ReLive!</title>
    <style>
        :root {
            --bg-primary: #FFFAF2;
            --border-color: #b1b0fe;
            --window-bg: #c19acb;
            --cream: #FFFDD0;
            --light-grey: #e8e8e8;
            --gradient-pink-green: linear-gradient(to right, #f56ebd, #97f0b6);
            --gradient-red-lime: linear-gradient(to right, #fd5a47, #c0e264);
        }
        
        * { box-sizing: border-box; margin: 0; padding: 0; }
        html, body { height: 100%; font-family: 'Nunito', sans-serif; font-weight: 550; }
        
        body {
            background: linear-gradient(135deg, var(--bg-primary) 0%, #f8f4e9 100%);
            color: #f5f5f5;
            overflow-x: hidden;
        }
        
        header.appbar {
            height: 80px;
            display: flex; align-items: center; justify-content: center;
            padding: 0 1rem;
            border-bottom: 2px solid var(--border-color);
            background: var(--light-grey);
            position: fixed; left: 0; right: 0; top: 0; z-index: 1000;
        }
        
        header .app-title {
            font-weight: 700;
            color: #333;
            font-size: 3rem;
            text-align: center;
        }
        
        .main-container {
            margin-top: 56px;
            padding: 2rem;
            min-height: calc(100vh - 56px);
            display: flex;
            justify-content: center;
            align-items: stretch;
        }
        
        .window {
            width: 100%;
            height: calc(100vh - 56px - 4rem);
            box-shadow: 0 10px 25px -5px rgba(0,0,0,0.15);
            border-radius: 1rem;
            border: 1px solid var(--border-color);
            background-color: var(--window-bg);
            backdrop-filter: blur(8px);
            transition: all 0.2s ease;
            display: flex;
            flex-direction: column;
        }
        
        .title-bar {
            display: flex; align-items: center; gap: .5rem;
            text-align: center;
            padding: .75rem 1rem;
            border-top-left-radius: 1rem;
            border-top-right-radius: 1rem;
            background: var(--gradient-pink-green);
            color: white;
            flex-shrink: 0;
        }
        
        .win-dot {
            width: 0.6rem; height: 0.6rem; border-radius: 50%;
            background-color: rgba(255,255,255,0.9);
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        
        .title-bar-text { 
            font-size: 1.2rem; font-weight: 700; flex-grow: 1; 
        }
        
        .window-body {
            padding: 2rem;
            color: #f5f5f5;
            flex: 1;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
        }
        
        .upload-section {
            text-align: center;
            margin-bottom: 2rem;
            flex-shrink: 0;
        }
        
        .upload-section h2 {
            font-size: 2.2rem;
            margin-bottom: 0.5rem;
            color: #f5f5f5;
        }
        
        .upload-section p {
            opacity: 0.9;
            margin-bottom: 1.5rem;
            font-size: 1.1rem;
        }
        
        .drop-zone {
            border: 3px dashed rgba(255,255,255,0.4);
            padding: 4rem 2rem;
            border-radius: 1rem;
            margin: 1.5rem 0;
            background: rgba(255,255,255,0.1);
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .drop-zone:hover {
            background: rgba(255,255,255,0.15);
            border-color: rgba(255,255,255,0.6);
            transform: translateY(-2px);
        }
        
        .btn {
            padding: 1rem 2rem;
            border-radius: 0.75rem;
            background: var(--gradient-red-lime);
            color: white; border: none; font-weight: 700;
            font-size: 1rem; cursor: pointer;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            transition: all 0.2s ease;
            font-family: 'Nunito', sans-serif;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.2);
        }
        
        .file-item, .status {
            padding: 1.2rem;
            background: rgba(255,255,255,0.1);
            margin: 0.5rem 0;
            border-radius: 0.75rem;
            border: 1px solid rgba(255,255,255,0.2);
        }
    </style>
</head>
<body>
    <header class="appbar">
        <div class="app-title">ReLive!</div>
    </header>

    <div class="main-container">
        <div class="window">
            <div class="title-bar">
                <div class="win-dot"></div>
                <div class="title-bar-text">Video Emotion Analysis</div>
            </div>
            
            <div class="window-body">
                <div class="upload-section">
                    <h2>🎭 Analyze Video Emotions</h2>
                    <p>Upload videos to analyze emotions from visual cues and get AI-powered feedback</p>
                    
                    <div class="drop-zone" onclick="document.getElementById('fileInput').click()">
                        <h3>📁 Click to select videos</h3>
                        <p>Supports: MP4, AVI, MOV, MKV, WMV, FLV, WEBM</p>
                    </div>
                    
                    <input type="file" id="fileInput" multiple accept="video/*" style="display: none;">
                    <button id="uploadBtn" class="btn" style="display: none;">🚀 Upload & Analyze Videos</button>
                </div>
                
                <div id="fileList"></div>
                <div id="status" style="display: none;"></div>
            </div>
        </div>
    </div>

    <script>
        let selectedFiles = [];
        let uploadedVideos = [];
        
        document.getElementById('fileInput').addEventListener('change', (e) => {
            selectedFiles = Array.from(e.target.files);
            displayFiles();
            document.getElementById('uploadBtn').style.display = selectedFiles.length > 0 ? 'block' : 'none';
        });

        function displayFiles() {
            const fileList = document.getElementById('fileList');
            fileList.innerHTML = '';
            selectedFiles.forEach(file => {
                const div = document.createElement('div');
                div.className = 'file-item';
                div.innerHTML = `📹 ${file.name} <span style="opacity: 0.7;">(${(file.size/1024/1024).toFixed(1)} MB)</span>`;
                fileList.appendChild(div);
            });
        }

        document.getElementById('uploadBtn').addEventListener('click', async () => {
            const formData = new FormData();
            selectedFiles.forEach(file => formData.append('videos', file));
            
            try {
                const response = await fetch('/upload', { method: 'POST', body: formData });
                const result = await response.json();
                uploadedVideos = result.videos;
                
                selectedFiles = [];
                document.getElementById('fileList').innerHTML = '';
                document.getElementById('uploadBtn').style.display = 'none';
                
                startPolling();
            } catch (error) {
                alert('Upload failed: ' + error);
            }
        });

        function startPolling() {
            const statusDiv = document.getElementById('status');
            statusDiv.style.display = 'block';
            statusDiv.innerHTML = '<h3>🔄 Processing videos...</h3><p>Please wait while we analyze your videos for emotional content.</p>';
            
            const interval = setInterval(async () => {
                let allCompleted = true;
                let statusHTML = '<h3>📊 Processing Status:</h3>';
                
                for (const video of uploadedVideos) {
                    const response = await fetch(`/status/${video.video_id}`);
                    const status = await response.json();
                    
                    const statusText = status.status === 'completed' ? '✅ Completed' : 
                                     status.status === 'processing' ? '🔄 Processing...' : 
                                     status.status === 'error' ? '❌ Error' : '⏳ Queued';
                    
                    statusHTML += `<div class="file-item">${video.filename}: ${statusText}</div>`;
                    
                    if (status.status !== 'completed') {
                        allCompleted = false;
                    }
                }
                
                statusDiv.innerHTML = statusHTML;
                
                if (allCompleted) {
                    clearInterval(interval);
                    statusDiv.innerHTML += '<div style="margin-top: 1rem;"><button class="btn" onclick="getFeedback()">💬 Get AI Feedback</button> <button class="btn" onclick="viewResults()">📈 View Results</button></div>';
                }
            }, 2000);
        }

        function viewResults() {
            window.open('/all_results', '_blank');
        }

        async function getFeedback() {
            try {
                // Show loading message
                const statusDiv = document.getElementById('status');
                statusDiv.innerHTML = '<h3>🔄 Generating AI feedback...</h3><p>Please wait while we analyze your video data.</p>';
                
                const response = await fetch('/analyze_feedback');
                const data = await response.json();
                
                if (data.error) {
                    statusDiv.innerHTML = `<h3>❌ Error</h3><p>${data.error}</p>`;
                    return;
                }
                
                // Create formatted feedback display
                let feedbackHtml = '<h3>🎭 AI Feedback Analysis Complete!</h3>';
                feedbackHtml += `<p><strong>Total Videos Analyzed:</strong> ${data.total_videos_analyzed}</p>`;
                feedbackHtml += `<p><strong>Analysis Generated:</strong> ${new Date(data.analysis_generated_at).toLocaleString()}</p>`;
                
                // Display each video's feedback
                data.feedback_results.forEach((result, index) => {
                    feedbackHtml += `
                        <div class="file-item" style="margin-top: 1.5rem; text-align: left;">
                            <h4>📹 Video ${index + 1} Analysis</h4>
                            <p><strong>Frames Analyzed:</strong> ${result.total_frames_analyzed}</p>
                            
                            <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; white-space: pre-wrap; font-family: inherit; line-height: 1.4;">
        ${result.feedback}
                            </div>
                            
                            ${result.poke_response && !result.poke_response.error ? 
                                '<p style="color: #4ade80;"><strong>✅ Feedback sent to Poke successfully!</strong></p>' : 
                                '<p style="color: #f87171;"><strong>⚠️ Note: Feedback not sent to Poke (check API configuration)</strong></p>'
                            }
                        </div>
                    `;
                });
                
                // Add option to view raw results
                feedbackHtml += `
                    <div style="margin-top: 2rem; text-align: center;">
                        <button class="btn" onclick="viewResults()" style="margin-right: 1rem;">📈 View Raw Results</button>
                        <button class="btn" onclick="window.print()">🖨️ Print Feedback</button>
                    </div>
                `;
                
                statusDiv.innerHTML = feedbackHtml;
                
            } catch (error) {
                console.error('Feedback error:', error);
                document.getElementById('status').innerHTML = `
                    <h3>❌ Error</h3>
                    <p>Failed to get feedback analysis: ${error.message}</p>
                    <button class="btn" onclick="getFeedback()">🔄 Try Again</button>
                `;
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return MAIN_PAGE_HTML

@app.route('/demo')
def demo():
    return UPLOAD_HTML

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'videos' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400
    
    files = request.files.getlist('videos')
    uploaded_videos = []
    
    for file in files:
        if file and file.filename != '' and allowed_file(file.filename):
            video_id = str(uuid.uuid4())
            filename = secure_filename(file.filename)
            
            video_path = os.path.join(UPLOAD_FOLDER, f"{video_id}_{filename}")
            file.save(video_path)
            
            processing_queue.put((video_id, video_path))
            processing_results[video_id] = {'status': 'queued'}
            
            uploaded_videos.append({
                'video_id': video_id,
                'filename': filename,
                'status': 'queued'
            })
    
    return jsonify({
        'message': f'Successfully uploaded {len(uploaded_videos)} videos',
        'videos': uploaded_videos
    })

@app.route('/static/<path:filename>')
def static_files(filename):
    from flask import send_from_directory
    return send_from_directory('static', filename)

@app.route('/status/<video_id>')
def get_status(video_id):
    if video_id not in processing_results:
        return jsonify({'error': 'Video not found'}), 404
    return jsonify(processing_results[video_id])

@app.route('/results/<video_id>')
def get_results(video_id):
    if video_id not in processing_results:
        return jsonify({'error': 'Video not found'}), 404
    
    result = processing_results[video_id]
    if result['status'] != 'completed':
        return jsonify({'error': 'Video processing not completed'}), 400
    
    try:
        with open(result['results_file'], 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': f'Error loading results: {e}'}), 500

@app.route('/all_results')
def get_all_results():
    all_results = []
    for video_id, status_info in processing_results.items():
        if status_info['status'] == 'completed':
            try:
                with open(status_info['results_file'], 'r') as f:
                    data = json.load(f)
                all_results.append(data)
            except:
                continue
    
    return jsonify(all_results)

@app.route('/analyze_feedback')
def get_analysis_feedback():
    """Analyze all videos and provide constructive feedback via Poke"""
    all_results = []
    for video_id, status_info in processing_results.items():
        if status_info['status'] == 'completed':
            try:
                with open(status_info['results_file'], 'r') as f:
                    data = json.load(f)
                all_results.append(data)
            except:
                continue
    
    if not all_results:
        return jsonify({'error': 'No completed video analyses found'}), 404
    
    # Analyze each video and combine feedback
    all_feedback = []
    for video_data in all_results:
        feedback = analyze_video_feedback(video_data)
        feedback['video_id'] = video_data['video_id']
        all_feedback.append(feedback)
    
    return jsonify({
        'total_videos_analyzed': len(all_results),
        'feedback_results': all_feedback,
        'analysis_generated_at': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🚀 Starting Video Emotion Analysis Server...")
    print("📁 Make sure to set ANTHROPIC_API_KEY in your environment")
    print("🌐 Server will run at: http://localhost:5000")
    app.run(debug=True, port=5000)