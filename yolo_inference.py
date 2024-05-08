# import model
from ultralytics import YOLO

# load model 
model = YOLO('models/best.pt')

# run model
results = model.predict('input_videos/aed0a26a-0c75-ee03-cc32-70b6e528d254_1280x720.mp4', save=True)
print(results[0]) # prints first frame
print('---')
for box in results[0].boxes:
    print(box)