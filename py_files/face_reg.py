import cv2
import pickle
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot
from numpy import load, expand_dims, asarray
from keras.models import load_model

detector = MTCNN(min_face_size=80)
model_fn = load_model('/home/akash-peace/Downloads/facenet_keras.h5')
trained_model = pickle.load(open('/home/akash-peace/Downloads/finalized_model.sav', 'rb'))
names = ['Arya Stark', 'Jon Snow']


# For Images
own_img = cv2.imread('/home/akash-peace/Downloads/testing_imgs/test4.jpg')
own_img = cv2.cvtColor(own_img, cv2.COLOR_BGR2RGB)
faces = detector.detect_faces(own_img)
for i in range(len(faces)):
    x1, y1, width, height = faces[i]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    uni_size = cv2.resize(own_img[y1:y2, x1:x2], (160, 160))
    face_own_img = asarray(uni_size)
    face_own_img = face_own_img.astype('float32')
    mean, std = face_own_img.mean(), face_own_img.std()
    face_own_img = (face_own_img - mean) / std
    own_sample = expand_dims(face_own_img, axis=0)
    own_prediction = model_fn.predict(own_sample)
    # COMPARING
    comp_class = trained_model.predict(own_prediction)
    comp_prob = trained_model.predict_proba(own_prediction)
    class_prob = comp_prob[0, comp_class[0]]*100
    if class_prob > 99.75:
        pyplot.imshow(uni_size)
        title = '%s (%.3f)' % (names[comp_class[0]], class_prob)
        pyplot.title(title)
        pyplot.show()


# For Videos
video = cv2.VideoCapture('/home/akash-peace/Downloads/got_end.mp4')
cv2.namedWindow('GOT', cv2.WINDOW_AUTOSIZE)
frame_width = int(video.get(3))
frame_height = int(video.get(4))
size = (frame_width, frame_height)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('/home/akash-peace/Downloads/face_recognition_of_arya_and_jon_among_all.avi', fourcc, 25.0, size)
f = 0
try:
    while True:
        f += 1
        ret, frame = video.read()
        if f % 1 == 0:  # This is for skipping frames, if you want to speed up the recognizing process(but the quantity of frame will be affected).
            faces = detector.detect_faces(frame)
            for i in range(len(faces)):
                x1, y1, width, height = faces[i]['box']
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height
                uni_size = cv2.resize(frame[y1:y2, x1:x2], (160, 160))
                face_own_img = asarray(uni_size)
                face_own_img = face_own_img.astype('float32')
                mean, std = face_own_img.mean(), face_own_img.std()
                face_own_img = (face_own_img - mean) / std
                own_sample = expand_dims(face_own_img, axis=0)
                own_prediction = model_fn.predict(own_sample)
                # COMPARING
                comp_class = trained_model.predict(own_prediction)
                comp_prob = trained_model.predict_proba(own_prediction)
                class_prob = comp_prob[0, comp_class[0]] * 100
                if class_prob > 99.75:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (127, 127, 127), 1)
                    cv2.putText(frame, names[comp_class[0]], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 1)
            out.write(frame)
            cv2.imshow('GOT', frame)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
    out.release()
    cv2.destroyAllWindows()
except:
    print("Recognized Successfully")
    pass
