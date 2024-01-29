from PIL import Image
import os
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import torch

classes_names = ['Cheating','Normal']

model = tf.keras.models.load_model('model.h5', compile=False)

model0 = YOLO('yolov8n-pose.pt')
device = torch.device('cpu')

i = 0
cap = cv2.VideoCapture(0)
print('Press "Esc", "q" or "Q" to exit.')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    persons = model0.predict(frame, conf=0.5, classes=[0])
    frame_color=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for p in persons:
        im_array = p.plot()
        im = Image.fromarray(im_array[..., ::-1])
        for b in p.boxes.xyxy:
          b=np.array(b)
          x, y, z, r = map(int, b)
          person_img = frame[y:r, x:z]
          person_img = cv2.resize(person_img, (200, 200))
          normalized_person_img =person_img / 255.0
          normalized_person_img = np.expand_dims(normalized_person_img, axis=0)
          result = model.predict(normalized_person_img)
          probabilities = tf.nn.softmax(result)
          result_class_index = tf.argmax(probabilities, axis=1).numpy()[0]
          result_class = classes_names[result_class_index]

          img_result = cv2.rectangle(frame_color, (x, y), (x + z, y + r), (36,255,12), 1)
          cv2.putText(img_result, result_class, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
          cv2.imshow('result', img_result)        
          if (result_class=='Cheating'):
            output_path = os.path.join('cheating', f'cheating{i}.jpg')
            Image.fromarray(person_img).save(output_path)
            print(i)
          i=i+1
    ch = cv2.waitKey(1)
    if ch == 27 or ch == ord('q') or ch == ord('Q'):
      break
cap.release()
cv2.destroyAllWindows()