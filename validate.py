# test to see if this can upload to GitHub


from ultralytics import YOLO
from pathlib import Path

model = YOLO('models/ant_detector.pt')

labels_dir = Path('data/labels/val')
images_dir = Path('data/images/val')

ant_images = []
non_ant_images = []

for img_file in sorted(images_dir.glob('*.jpg')):
    lbl_file = labels_dir / f'{img_file.stem}.txt'
    if lbl_file.exists():
        content = lbl_file.read_text().strip()
        if content:
            ant_images.append(img_file)
        else:
            non_ant_images.append(img_file)

print('='*60)
print('  FULL VALIDATION - ANT DETECTOR v2')
print('='*60)
print(f'Total: {len(ant_images)} ant images, {len(non_ant_images)} non-ant images')

# Test all ants
print(f'\n[ANT IMAGES] - should detect')
print('-'*60)
results = model.predict(source=[str(img) for img in ant_images], conf=0.40, verbose=False)
correct = 0
for img, r in zip(ant_images, results):
    detected = len(r.boxes) > 0
    status = 'OK' if detected else 'MISS'
    conf = f'{float(r.boxes.conf[0])*100:.0f}%' if detected else '-'
    label = 'ANT' if detected else 'no'
    print(f'  {img.name[:35]:35} {label:4} [{status:4}] {conf}')
    if detected:
        correct += 1
print(f'\n  >>> ANT DETECTION: {correct}/{len(ant_images)} ({correct/len(ant_images)*100:.0f}%) <<<')

# Test all non-ants
print(f'\n[NON-ANT IMAGES] - should NOT detect')
print('-'*60)
results = model.predict(source=[str(img) for img in non_ant_images], conf=0.40, verbose=False)
false_pos = 0
fp_list = []
for img, r in zip(non_ant_images, results):
    detected = len(r.boxes) > 0
    if detected:
        false_pos += 1
        conf = f'{float(r.boxes.conf[0])*100:.0f}%'
        fp_list.append(f'  {img.name[:35]:35} ANT  [FP  ] {conf}')

if fp_list:
    for fp in fp_list:
        print(fp)
else:
    print('  (none detected - all correct!)')

print(f'\n  >>> FALSE POSITIVES: {false_pos}/{len(non_ant_images)} ({false_pos/len(non_ant_images)*100:.0f}%) <<<')

print()
print('='*60)
print(f'  FINAL: {correct/len(ant_images)*100:.0f}% detection, {false_pos/len(non_ant_images)*100:.0f}% false positive')
print('='*60)
