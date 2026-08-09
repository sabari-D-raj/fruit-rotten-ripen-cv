import cv2
import numpy as np
from tensorflow.keras.models import load_model
from ultralytics import YOLO
cap=cv2.VideoCapture(0)
y_model=YOLO("yolo11n.pt")
model=load_model("fruits.keras")
classes=["fresh appple","fresh banana","fresh orange","rotten apple","rotten banana","rotten orange","unripe apple","unripe banana","unripe orange"]
while True:
    succes,frame=cap.read()
    if not succes:
        print("failed")
        break
    results=y_model(frame,verbose=False)
    for result in results:
        for box in result.boxes:
            class_id=int(box.cls[0])
            object_name=y_model.names[class_id]
            if object_name in ["apple","banana","orange"]:
                x1,y1,x2,y2=map(int,box.xyxy[0])
                fruit=frame[y1:y2,x1:x2]
                if fruit.size==0:
                    continue
                fruit_rgb = cv2.cvtColor(fruit, cv2.COLOR_BGR2RGB)
                img=cv2.resize(fruit_rgb,(244,244))
                img=img/255.0
                img=np.expand_dims(img,axis=0)
                prediction=model.predict(img,verbose=0)
                class_index=np.argmax(prediction)
                confidence=(np.max(prediction)*100)
                label=classes[class_index]
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                text=(f"{label}:{confidence}")
                cv2.putText(frame,text,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    if cv2.waitKey(1) & 0xFF==ord("q"):
        break
    cv2.imshow("fruit-detection",frame)
cap.release()
cv2.destroyAllWindows()