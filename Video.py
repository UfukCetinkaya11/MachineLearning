import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array


face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model('Saved_modelV3Augmented.h5')

class_labels= ['angry', 'disgust','fear','happy','sad','surprise','neutral']

cap=cv2.VideoCapture(0)

count = 0
temp = np.array([])
emotion_weight = {'angry':0.25,'disgust':0.2,'fear':0.3,'happy':0.6,'sad':0.3,'surprise':0.6,'neutral':0.9}

while True:
    ret,test_img=cap.read()
    labels = []
    gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.3, 5)


    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,0,255),thickness=2)
        roi_gray=gray_img[y:y+h,x:x+w]
        roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

        if(np.sum([roi_gray]) != 0):
            roi = roi_gray.astype('float')/255
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)


            predictions = model.predict(roi)[0]
            label=class_labels[predictions.argmax()]
            corresponding_weight = emotion_weight[label]
            concentration_index = corresponding_weight * predictions[predictions.argmax()]
            label_position = (x,y)
            cv2.putText(test_img, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            temp = np.append(temp,concentration_index)
            count = count + 1
            if(count % 15 ==0):
                mean_concentration = np.mean(temp)
                cv2.putText(test_img, str(mean_concentration), (x + w, y), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 0, 255), 2)
                temp = np.delete(temp,[0])
                count = 14



        else:
            cv2.putText(test_img,"No face found",(20,60), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255),2)
        print("\n\n")
    cv2.imshow("Emotion Recognition", test_img)
    if(cv2.waitKey(1) & 0xFF == ord("e")):
        break
cap.release()
cv2.destroyAllWindows()





