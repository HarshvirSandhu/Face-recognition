import os
import mediapipe as mp
import cv2
from skimage.metrics import structural_similarity as ssim
from datetime import datetime
mpface = mp.solutions.face_detection
face = mpface.FaceDetection(model_selection=1)
cap = cv2.VideoCapture(0)
h = cap.get(3)
w = cap.get(4)
print("Enter the file path: ")
data_path = str(input())
print(data_path)
video_num = "Facefound.mp4"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_num, fourcc, 40, (int(h), int(w)))
a = 0
while True:
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face.process(frame_rgb)
    h, w, channels = frame_rgb.shape
    if results.detections:
        # print(results.detections[0])
        for detect in results.detections:
            c1 = (int(detect.location_data.relative_bounding_box.xmin * w),
                  int(detect.location_data.relative_bounding_box.ymin * h))
            c2 = (c1[0]+int(detect.location_data.relative_bounding_box.width * w),
                  c1[1]+int(detect.location_data.relative_bounding_box.height * h))
            person = frame[c1[1]:c2[1], c1[0]:c2[0]]
            cv2.rectangle(frame, c1, c2, (255, 0, 0), 3)
            cv2.putText(frame, str(datetime.now().time()), (20, 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            cv2.putText(frame, str(datetime.now().date()), (20, 45), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            person = cv2.cvtColor(person, cv2.COLOR_BGR2GRAY)
            rx, ry = person.shape
            # print(person.shape)
            person = cv2.resize(person, (int(3*rx), int(3*ry)))
            for img_name in os.listdir(data_path):
                img = cv2.imread(data_path+'\\'+img_name, 0)
                img = cv2.resize(img, (int(3*rx), int(3*ry)))
                s = ssim(img, person)
                # print(s)
                if s > 0.7:
                    cv2.rectangle(frame, c1, c2, (0, 255, 0), 3)
                    a += 1
                    out.write(frame)
                    video_num = "Facefound" + str(a) + ".mp4"
            # cv2.imshow("FACE", person)
    # else:
        # print("No face")
        # video_num = "Facefound" + str(a) + ".mp4"
    cv2.imshow("LIVE", frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
out.release()