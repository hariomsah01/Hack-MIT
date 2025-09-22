# 🎬 Video Emotion Analysis with AI Feedback

A Flask-based web application that analyzes emotions from videos collected by the Mentra Live glass, using computer vision, audio processing, and AI-powered feedback generation. Built for HackMIT 2025 with integration to Anthropic's LLM Model - Claude and Poke's automation platform.

## ✨ Features

- Multi-video upload with drag-and-drop interface
- Advanced frame extraction and image enhancement (CLAHE Grayscale)
- Audio processing with MFCC feature extraction
- Emotion analysis using Anthropic's Claude API
- AI-powered feedback generation for presentation skills
- Poke integration for automated feedback delivery
- Real-time processing status updates
- RESTful API for frontend integration

## 💻 Technology Stack

- **Backend**: Flask, Python 3.8+
- **Computer Vision**: OpenCV, custom image processing
- **Audio Processing**: SoundFile, FFmpeg, MFCC analysis
- **AI Analysis**: Anthropic Claude API
- **Automation**: Poke API integration
- **Frontend**: HTML5, JavaScript, CSS3

## 📦 Installation

### 🛠️ Prerequisites

- Python 3.8 or higher
- FFmpeg installed and in PATH
- Anthropic API key
- Poke API key

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/video-emotion-analysis.git
cd video-emotion-analysis
```

2. **Install Python dependencies**
```bash
pip install flask flask-cors opencv-python anthropic soundfile python-dotenv requests
```

3. **Install system dependencies**
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

4. **Set up environment variables**

Create a `.env` file:
```bash
ANTHROPIC_API_KEY=your-anthropic-api-key-here
POKE_API_KEY=your-poke-api-key-here
```

5. **Create required modules**

Ensure you have the following files:
- `audio_processor.py` - Audio processing utilities
- `image_processor.py` - Image enhancement and analysis

## Usage

### Running the Application

```bash
python app.py
```

The server will start at `http://localhost:5000`

### API Endpoints

- `POST /upload` - Upload multiple video files
- `GET /status/<video_id>` - Check processing status
- `GET /results/<video_id>` - Get analysis results for specific video
- `GET /all_results` - Get all analysis results (JSON format)
- `GET /analyze_feedback` - Generate AI feedback and send to Poke

### Processing Pipeline

1. **Video Upload**: Drag and drop video files
2. **Frame Extraction**: Extract frames at 2-second intervals
3. **Image Processing**: Apply CLAHE enhancement and feature extraction
4. **Audio Processing**: Extract audio clips and compute MFCC features
5. **Emotion Analysis**: Analyze facial expressions using Claude API
6. **Feedback Generation**: Create constructive presentation feedback
7. **Poke Integration**: Automatically send feedback via Poke API

## ⚙️ Configuration

### Sampling Settings
```python
SAMPLE_RATE_SECONDS = 2  # Extract frame every 2 seconds
AUDIO_CLIP_DURATION_SECONDS = 3  # 3-second audio clips
AUDIO_SAMPLE_RATE = 44100  # Audio sample rate
```

### Supported Video Formats
- MP4, AVI, MOV, MKV, WMV, FLV, WebM

## 📊 Example Output

### Emotion Analysis Result
```json
{
  "timestamp": 8,
  "frame_emotion": {
    "visual_analysis": {
      "emotions": {
        "joy": 8,
        "sadness": 1,
        "anger": 0,
        "fear": 0,
        "surprise": 3,
        "disgust": 0,
        "neutral": 2
      },
      "description": "A young person wearing glasses is smiling broadly..."
    }
  },
  "image_features": {
    "dimensions": {"width": 320, "height": 568},
    "perceptual_hash": "afb5a3cc00113daf",
    "enhancement_method": "CLAHE_Grayscale"
  },
  "audio_features": {
    "mfcc_mean": [-284.71, 128.55, -30.30, ...],
    "sample_rate": 16000
  }
}
```

### AI Feedback Example
```
POSITIVE ASPECTS:
Your authentic expressions show excellent emotional range, with genuine moments of joy (scores 8-9) creating strong viewer engagement...

CONSTRUCTIVE FEEDBACK:
Consider maintaining more consistent energy levels during neutral segments...

ACTIONABLE TIPS:
1. Practice maintaining eye contact with the camera
2. Use purposeful hand gestures to emphasize points
3. Vary vocal tone to match emotional content
```

## 📁 Project Structure

```
video-emotion-analysis/
├── app.py                 # Main Flask application
├── audio_processor.py     # Audio processing utilities
├── image_processor.py     # Image enhancement utilities
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables
├── uploads/              # Uploaded video files
├── extracted/
│   ├── frames/           # Extracted video frames
│   └── audio/            # Extracted audio segments
└── results/              # Analysis results (JSON)
```

## 🔑 API Keys Setup

### Anthropic API Key
1. Go to https://console.anthropic.com/
2. Create an account and generate an API key
3. Add to `.env` file: `ANTHROPIC_API_KEY=your-key`

### Poke API Key
1. Go to https://poke.com/settings/advanced
2. Create an API key
3. Add to `.env` file: `POKE_API_KEY=your-key`

## 🔧 Development

### Adding New Emotion Models
Modify the `analyze_frame_emotion()` function to use different AI models or add custom emotion detection algorithms.

### Customizing Audio Features
Update `audio_processor.py` to extract additional audio features like spectral features, rhythm analysis, or voice sentiment.

### Extending Feedback Generation
Enhance the feedback prompts in `analyze_video_feedback()` to provide more specific coaching for different use cases (presentations, interviews, etc.).

## Troubleshooting

### Common Issues

**FFmpeg not found**
```bash
# Add FFmpeg to PATH or install via package manager
export PATH=$PATH:/path/to/ffmpeg/bin
```

**Module import errors**
```bash
# Install missing dependencies
pip install -r requirements.txt
```

**API rate limits**
- Claude Haiku: 25,000 input tokens/minute
- Reduce frame extraction rate if hitting limits

**Memory issues**
- Process videos sequentially
- Reduce `SAMPLE_RATE_SECONDS` for large videos

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built for HackMIT 2025
- Anthropic Claude API for emotion analysis
- Poke platform for automation integration
- OpenCV community for computer vision tools

## 👥 Team

- **Backend Development**: Video processing and AI integration
- **Frontend Development**: User interface and experience
- **AI Integration**: Emotion analysis and feedback generation

---

**Built with ❤️ at HackMIT 2025** 🚀
