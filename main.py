import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import pandas as pd
from datetime import datetime
import time
from tqdm import tqdm
import os

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def draw_box_with_opacity(img, bbox, color, opacity=0.5, text="", text2=""):
    """Draw box with opacity and text"""
    overlay = img.copy()
    x1, y1, x2, y2 = map(int, bbox)
    
    # Draw filled rectangle on overlay
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    
    # Blend overlay with original image
    cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)
    
    # Draw border
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    # Draw text with background
    if text:
        # Text background
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1-text_height-8), (x1+text_width, y1), color, -1)
        # Text
        cv2.putText(img, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    
    if text2:
        # Second text background
        (text_width, text_height), _ = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1-text_height-28), (x1+text_width, y1-20), color, -1)
        # Second text
        cv2.putText(img, text2, (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

def process_image(model_path, image_path, output_path=None):
    print("\nInitializing...")
    model = YOLO(model_path)
    
    # Read image
    frame = cv2.imread(image_path)
    if frame is None:
        print("Error: Could not read image!")
        return True
        
    height, width = frame.shape[:2]
    
    # Initialize metrics
    confidence_scores = []
    bbox_areas = []
    detections = 0
    
    # Detect objects
    results = model.predict(frame, conf=0.2, show=False)
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model.names[cls]
            
            confidence_scores.append(conf)
            area = (x2 - x1) * (y2 - y1) / (width * height)
            bbox_areas.append(area)
            
            # Generate random color for this detection
            color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
            
            # Draw box and label
            label = f"{class_name} {conf:.2f}"
            draw_box_with_opacity(frame, [x1, y1, x2, y2], color, 0.3, label)
            detections += 1
    
    # Show image with detections
    cv2.imshow('Detection Result', frame)
    
    # Save output if specified
    if output_path:
        cv2.imwrite(output_path, frame)
        print(f"\nOutput saved to: {output_path}")
    
    # Calculate and display metrics
    if confidence_scores and bbox_areas:
        metrics = {
            "Detection Summary": {
                "Total Detections": detections,
                "Image Size": f"{width}x{height}"
            },
            "Confidence Metrics": {
                "Average Confidence": f"{np.mean(confidence_scores):.3f}",
                "Min Confidence": f"{min(confidence_scores):.3f}",
                "Max Confidence": f"{max(confidence_scores):.3f}"
            },
            "Bounding Box Metrics": {
                "Average BBox Area": f"{np.mean(bbox_areas):.4f}",
                "BBox Area Range": f"{min(bbox_areas):.4f} - {max(bbox_areas):.4f}"
            }
        }
        
        print("\nProcessing Complete!")
        for category, category_metrics in metrics.items():
            print(f"\n📊 {category}:")
            for metric, value in category_metrics.items():
                print(f"  • {metric}: {value}")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    while True:
        print("\nPress 'M' to return to main menu or 'Q' to quit")
        choice = input().strip().upper()
        if choice == 'M':
            return True
        elif choice == 'Q':
            return False
        else:
            print("Invalid choice. Please press 'M' or 'Q'")
def process_video(model_path, source, output_path=None, use_webcam=False, use_rtsp=False):
    print("\nInitializing... Press ESC to cancel")
    model = YOLO(model_path)
    tracker = DeepSort(
        max_age=60,               # Increased from 30
        n_init=5,                 # Increased from 3
        nn_budget=100,            # Keep more samples
        embedder='mobilenet',
        max_iou_distance=0.7,    # Increased from default
    )
    
    # Check for cancel during initialization
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        print("\nOperation cancelled")
        cv2.destroyAllWindows()
        return True
    
    # Initialize video capture based on source type
    if use_webcam:
        cap = cv2.VideoCapture(0)
        total_frames = float('inf')
        pbar = None
    elif use_rtsp:
        print(f"Connecting to RTSP stream: {source}")
        cap = cv2.VideoCapture(source)
        total_frames = float('inf')
        pbar = None
        if not cap.isOpened():
            raise Exception("Failed to connect to RTSP stream")
    else:
        cap = cv2.VideoCapture(source)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total_frames, desc='Processing video')
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if use_webcam or use_rtsp:
        fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize video writer if output path is provided
    out = None
    if output_path:
        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )
    
    # Initialize metrics
    start_time = time.time()
    tracks_data = {}
    confidence_scores = []
    bbox_areas = []
    objects_per_frame = []
    id_switches = 0
    last_ids = set()
    frame_count = 0
    
    # Store colors for track IDs
    track_colors = {}
    
    # FPS calculation variables
    fps_start_time = time.time()
    fps_frame_count = 0
    fps_display = 0
    
    print("\nProcessing started. Press 'q' to stop or 'ESC' to cancel...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        current_ids = set()
        
        # Update FPS calculation
        fps_frame_count += 1
        if (time.time() - fps_start_time) > 1.0:
            fps_display = fps_frame_count
            fps_frame_count = 0
            fps_start_time = time.time()
        
        # Detect objects
        results = model.predict(frame, conf=0.2, show=False)
        
        # Process detections
        detections = []
        frame_confidences = []
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = model.names[cls]
                
                frame_confidences.append(conf)
                area = (x2 - x1) * (y2 - y1) / (width * height)
                bbox_areas.append(area)
                
                w = x2 - x1
                h = y2 - y1
                detections.append(([x1,y1,w,h], conf, class_name))
        
        # Update tracks
        tracks = tracker.update_tracks(detections, frame=frame)
        objects_per_frame.append(len(tracks))
        
        # Process tracks
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            current_ids.add(track_id)
            ltrb = track.to_ltrb()
            class_name = track.det_class
            
            # Generate and store consistent color for track_id
            if track_id not in track_colors:
                track_colors[track_id] = (
                    np.random.randint(0,255),
                    np.random.randint(0,255),
                    np.random.randint(0,255)
                )
            color = track_colors[track_id]
            
            if track_id not in tracks_data:
                tracks_data[track_id] = {
                    'class_name': class_name,
                    'start_frame': frame_count,
                    'positions': []
                }
            
            tracks_data[track_id]['positions'].append({
                'frame': frame_count,
                'bbox': ltrb
            })
            
            # Calculate duration
            duration = (frame_count - tracks_data[track_id]['start_frame']) / fps
            
            # Draw box with opacity
            label = f"#{track_id} {class_name}"
            duration_text = f"Time: {duration:.1f}s"
            draw_box_with_opacity(frame, ltrb, color, 0.3, label, duration_text)
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {fps_display}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Check for ID switches
        if last_ids:
            disappeared = last_ids - current_ids
            appeared = current_ids - last_ids
            if disappeared and appeared:
                id_switches += 1
        
        last_ids = current_ids
        confidence_scores.extend(frame_confidences)
        
        if out:
            out.write(frame)
            
        cv2.imshow('Tracking', frame)
        
        if pbar:
            pbar.update(1)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 27:  # ESC key
            print("\nOperation cancelled")
            break
    
    # Calculate and display metrics
    end_time = time.time()
    processing_time = end_time - start_time
    
    if tracks_data and confidence_scores and bbox_areas:
        total_detections = sum(objects_per_frame)
        unique_tracks = len(tracks_data)
        
        metrics = {
            "Processing Summary": {
                "Processed Frames": frame_count,
                "Time Taken": f"{processing_time:.2f} seconds",
                "Average FPS": f"{frame_count/processing_time:.2f}",
                "Video Duration": f"{frame_count/fps:.2f} seconds"
            },
            "Detection and Tracking Metrics": {
                "Total Detections": total_detections,
                "Unique Tracks": unique_tracks,
                "Average Objects per Frame": f"{np.mean(objects_per_frame):.2f}",
                "Max Objects in Frame": max(objects_per_frame),
                "Min Objects in Frame": min(objects_per_frame),
                "ID Switches": id_switches
            },
            "Confidence Metrics": {
                "Average Confidence": f"{np.mean(confidence_scores):.3f}",
                "Min Confidence": f"{min(confidence_scores):.3f}",
                "Max Confidence": f"{max(confidence_scores):.3f}"
            },
            "Tracking Performance": {
                "Average Track Duration": f"{np.mean([len(t['positions'])/fps for t in tracks_data.values()]):.2f}s",
                "Average Track Length": f"{np.mean([len(t['positions']) for t in tracks_data.values()]):.2f} frames",
                "Max Track Length": max([len(t['positions']) for t in tracks_data.values()]),
                "Average BBox Area": f"{np.mean(bbox_areas):.4f}",
                "BBox Area Range": f"{min(bbox_areas):.4f} - {max(bbox_areas):.4f}",
                "Total Bounding Boxes": total_detections
            }
        }
        
        print("\nProcessing Complete!")
        for category, category_metrics in metrics.items():
            print(f"\n📊 {category}:")
            for metric, value in category_metrics.items():
                print(f"  • {metric}: {value}")
    
    # Clean up
    if pbar:
        pbar.close()
    if out:
        out.release()
    cap.release()
    cv2.destroyAllWindows()
    
    while True:
        print("\nPress 'M' to return to main menu or 'Q' to quit")
        choice = input().strip().upper()
        if choice == 'M':
            return True
        elif choice == 'Q':
            return False
        else:
            print("Invalid choice. Please press 'M' or 'Q'")

def main():
    while True:
        clear_screen()
        print("\n=== Object Tracking System ===")
        print("1. Process Video File")
        print("2. Use Webcam")
        print("3. Use IP Camera (RTSP)")
        print("4. Process Image")
        print("5. Exit")
        print("\nSelect an option (1-5): ")
        
        try:
            choice = input().strip()
            
            # First ask for weight file for all options except exit
            if choice in ['1', '2', '3', '4']:
                print("\nEnter path to weight file (press Enter for default 'best.pt'): ")
                weight_path = input().strip()
                if not weight_path:
                    weight_path = 'best.pt'
                if not os.path.exists(weight_path):
                    print(f"\nError: Weight file {weight_path} not found!")
                    input("\nPress Enter to continue...")
                    continue
            
            if choice == "1":
                video_path = input("\nEnter video file path: ")
                output_path = input("Enter output video path (or press Enter to skip saving): ")
                if not output_path.strip():
                    output_path = None
                    
                if os.path.exists(video_path):
                    return_to_menu = process_video(
                        model_path=weight_path,
                        source=video_path,
                        output_path=output_path,
                        use_webcam=False
                    )
                    if not return_to_menu:
                        break
                else:
                    print(f"\nError: File {video_path} not found!")
                    input("\nPress Enter to continue...")
                    
            elif choice == "2":
                output_path = input("\nEnter output video path (or press Enter to skip saving): ")
                if not output_path.strip():
                    output_path = None
                    
                return_to_menu = process_video(
                    model_path=weight_path,
                    source=None,
                    output_path=output_path,
                    use_webcam=True
                )
                if not return_to_menu:
                    break
                
            elif choice == "3":
                rtsp_url = input("\nEnter RTSP URL (e.g., rtsp://username:password@ip:port/stream): ")
                output_path = input("Enter output video path (or press Enter to skip saving): ")
                if not output_path.strip():
                    output_path = None
                
                return_to_menu = process_video(
                    model_path=weight_path,
                    source=rtsp_url,
                    output_path=output_path,
                    use_rtsp=True
                )
                if not return_to_menu:
                    break
                
            elif choice == "4":
                image_path = input("\nEnter image file path: ")
                output_path = input("Enter output image path (or press Enter to skip saving): ")
                if not output_path.strip():
                    output_path = None
                    
                if os.path.exists(image_path):
                    return_to_menu = process_image(
                        model_path=weight_path,
                        image_path=image_path,
                        output_path=output_path
                    )
                    if not return_to_menu:
                        break
                else:
                    print(f"\nError: File {image_path} not found!")
                    input("\nPress Enter to continue...")
                
            elif choice == "5":
                print("\nExiting program...")
                break
                
            else:
                print("\nInvalid choice! Please select 1-5.")
                input("\nPress Enter to continue...")
                
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()