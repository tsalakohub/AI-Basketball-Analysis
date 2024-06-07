# import model
from ultralytics import YOLO

# load model 
model = YOLO('models/best.pt')

# run model
results = model.predict('input_videos/input_video.mp4', save=True)
print(results[0]) # prints first frame
print('---')
for box in results[0].boxes:
    print(box)