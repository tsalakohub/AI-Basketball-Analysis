# import model
from ultralytics import YOLO

# load model 
model = YOLO('models/best.pt')

# run model
results = model.predict('input_videos/725edc7c-6e5e-fe5d-a1b4-219c3e07d32a_1280x720.mp4', save=True)
print(results[0]) # prints first frame
print('---')
for box in results[0].boxes:
    print(box)