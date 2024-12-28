# Object Detection and Tracking System

This project combines YOLO (You Only Look Once) object detection with DeepSORT tracking to create a comprehensive system for real-time object detection and tracking across multiple input sources. The system supports video files, webcam feeds, RTSP streams, and static images.

## 🌟 Features

- **Multiple Input Sources Support**:
  - Video file processing
  - Webcam real-time detection
  - IP Camera (RTSP) streaming
  - Static image detection

- **Advanced Tracking**:
  - Object tracking with unique IDs
  - Track duration monitoring
  - ID switch detection
  - Consistent color coding per track

- **Visualization**:
  - Semi-transparent bounding boxes
  - Real-time FPS counter
  - Object class labels
  - Track duration display
  - Track ID visualization

- **Comprehensive Metrics**:
  - Processing performance stats
  - Detection confidence analysis
  - Tracking performance metrics
  - Bounding box statistics

- **User-Friendly Interface**:
  - Interactive menu system
  - Custom weight file selection
  - Operation cancellation (ESC)
  - Return to menu option
  - Progress bar for video processing

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/object-detection-tracking.git
cd object-detection-tracking
```

2. Create and activate virtual environment (optional but recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## 📦 Requirements

- Python 3.8+
- OpenCV
- Ultralytics YOLO
- Deep SORT Real-Time
- NumPy
- Pandas
- tqdm

## 🚀 Usage

1. Run the main script:
```bash
python main.py
```

2. Select from available options:
```
=== Object Tracking System ===
1. Process Video File
2. Use Webcam
3. Use IP Camera (RTSP)
4. Process Image
5. Exit
```

3. Enter the path to your YOLO weights file (press Enter for default 'best.pt')

4. Provide required input paths based on your selection

### Controls During Operation:
- Press 'q' to stop processing
- Press 'ESC' to cancel operation
- After processing:
  - Press 'M' to return to main menu
  - Press 'Q' to quit program

## 📊 Output Metrics

The system provides comprehensive metrics including:

- Processing Summary:
  - Processed Frames
  - Time Taken
  - Average FPS
  - Video Duration

- Detection and Tracking Metrics:
  - Total Detections
  - Unique Tracks
  - Objects per Frame
  - ID Switches

- Confidence Metrics:
  - Average Confidence
  - Min/Max Confidence

- Tracking Performance:
  - Track Duration
  - Track Length
  - Bounding Box Areas

## 🖼️ Sample Output

```
📊 Processing Summary:
  • Processed Frames: 300
  • Time Taken: 10.5 seconds
  • Average FPS: 28.57
  • Video Duration: 10.00 seconds

📊 Detection and Tracking Metrics:
  • Total Detections: 450
  • Unique Tracks: 5
  • Average Objects per Frame: 1.50
  • ID Switches: 2
```

## 🎯 Use Cases

- Security and Surveillance
- Traffic Monitoring
- People Counting
- Object Movement Analysis
- Retail Analytics
- Sports Analysis

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👏 Acknowledgments

- YOLOv8 by Ultralytics
- Deep SORT algorithm
- OpenCV team
- Contributor community

## 👏 Supervisor

- Prof. Chao-Tung Yang, Tunghai University
- Assistant Prof. Ding-Hsiang Huang., Ph.D, Tunghai University
- OpenCV team
- Contributor community

## ✉️ Contact

Anggi Andriyadi - anggi.andriyadi@gmail.com
Tunghai University

Project Link: [https://github.com/yourusername/object-detection-tracking](https://github.com/yourusername/object-detection-tracking)

# Result
[![Object Detection and Tracking Demo](https://img.youtube.com/vi/5ccwZCiLfRk/0.jpg)](https://youtu.be/5ccwZCiLfRk)
