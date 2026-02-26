# Ant Detector

Binary AI model for detecting ants in images. Uses YOLOv8 with optional SAHI tiling for small ant detection.

## Performance
- **Detection Rate:** 81% (standard) / 91% (SAHI mode)
- **False Positive Rate:** 6%
- **Speed:** 0.2s/image (CPU) | 50+ FPS (Jetson with TensorRT)

## Installation

```bash
pip install ultralytics sahi
```

## Usage

### Detect ants in images
```bash
# Single image
python detect_ant.py image.jpg

# Folder of images
python detect_ant.py images/

# SAHI mode for small ants (slower but more accurate)
python detect_ant.py image.jpg --sahi
```

### Output
```
ANT DETECTOR (STANDARD)
==================================================

Image: test.jpg
  >>> YES - ANT DETECTED <<<
  Count: 3 ant(s)
  Max confidence: 87%
```

## Model

The trained model is located at `models/ant_detector.pt`

### Use in your own code
```python
from ultralytics import YOLO

model = YOLO('models/ant_detector.pt')
results = model.predict('image.jpg', conf=0.40)

for r in results:
    if len(r.boxes) > 0:
        print(f"Ants detected: {len(r.boxes)}")
```

## Dataset

- **Training:** 823 images (ants + negative examples)
- **Validation:** 158 images
- Located in `data/images/` and `data/labels/`

## Jetson Deployment

For Jetson Orin Nano deployment:
```bash
# Convert to TensorRT on Jetson
yolo export model=models/ant_detector.pt format=engine device=0

# Run with camera
python detect_ant.py --source 0 --model ant_detector.engine
```

## License
MIT
