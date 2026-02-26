"""
BINARY ANT DETECTOR
===================
Detects ants in images with YES/NO output and counts ants.

Usage:
  py detect_ant.py image.jpg              # Standard mode (~0.2s/image)
  py detect_ant.py image.jpg --sahi       # SAHI mode for small ants (~2.4s/image)
  py detect_ant.py folder/ --sahi         # Process folder with SAHI
"""

from ultralytics import YOLO
from pathlib import Path
import argparse

# Optimal confidence threshold
DEFAULT_CONFIDENCE = 0.40
MODEL_PATH = Path(__file__).parent / 'models/ant_detector.pt'


def detect_ant(image_path, confidence=DEFAULT_CONFIDENCE, use_sahi=False, save=True):
    """Detect ants in image and return count."""
    
    if use_sahi:
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
        
        model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=str(MODEL_PATH),
            confidence_threshold=confidence
        )
        
        result = get_sliced_prediction(
            image_path,
            model,
            slice_height=320,
            slice_width=320,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            verbose=0
        )
        
        if save:
            out_dir = Path('runs/detect/ant_results')
            out_dir.mkdir(parents=True, exist_ok=True)
            result.export_visuals(export_dir=str(out_dir))
        
        detections = []
        for pred in result.object_prediction_list:
            detections.append({
                'confidence': pred.score.value,
                'bbox': pred.bbox.to_xyxy()
            })
    else:
        model = YOLO(str(MODEL_PATH))
        
        results = model.predict(
            source=image_path,
            conf=confidence,
            save=save,
            project='runs/detect',
            name='ant_results',
            exist_ok=True,
            verbose=False
        )
        
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    'confidence': float(box.conf[0]),
                    'bbox': box.xyxy[0].tolist()
                })
    
    return {
        'is_ant': len(detections) > 0,
        'count': len(detections),
        'max_confidence': max([d['confidence'] for d in detections]) if detections else 0,
        'detections': detections
    }


def main():
    parser = argparse.ArgumentParser(description='Binary Ant Detector with Counting')
    parser.add_argument('image', help='Image file or folder')
    parser.add_argument('--conf', type=float, default=DEFAULT_CONFIDENCE, help='Confidence threshold')
    parser.add_argument('--sahi', action='store_true', help='Use SAHI for small ant detection (~2.4s/image)')
    parser.add_argument('--no-save', action='store_true', help='Do not save annotated images')
    args = parser.parse_args()
    
    image_path = Path(args.image)
    
    print()
    print('=' * 50)
    mode = 'SAHI MODE' if args.sahi else 'STANDARD'
    print(f'        ANT DETECTOR ({mode})')
    print('=' * 50)
    
    if image_path.is_file():
        images = [image_path]
    else:
        images = list(image_path.glob('*.jpg')) + list(image_path.glob('*.png'))
    
    images_with_ants = 0
    total_ants = 0
    
    for img in images:
        result = detect_ant(str(img), confidence=args.conf, use_sahi=args.sahi, save=not args.no_save)
        
        print()
        print(f'Image: {img.name}')
        
        if result['is_ant']:
            images_with_ants += 1
            total_ants += result['count']
            print(f'  >>> YES - ANT DETECTED <<<')
            print(f'  Count: {result["count"]} ant(s)')
            print(f'  Max confidence: {result["max_confidence"]*100:.0f}%')
        else:
            print(f'  >>> NO - No ant detected <<<')
    
    if len(images) > 1:
        print()
        print('=' * 50)
        print(f'SUMMARY')
        print(f'  Images with ants: {images_with_ants}/{len(images)}')
        print(f'  Total ants found: {total_ants}')
        print('=' * 50)
    
    print()
    print('Results saved to: runs/detect/ant_results/')


if __name__ == '__main__':
    main()
