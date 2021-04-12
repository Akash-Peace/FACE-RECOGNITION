from mtcnn.mtcnn import MTCNN
import cv2
import os
index = 0
detector = MTCNN()
filenames = '/home/akash-peace/Downloads/img_data/'
for filename in os.listdir(filenames):
    filename = f'{filenames}{filename}'
    index += 1
    if filename[-3:] == 'jpg':
        data = cv2.imread(filename)
        faces = detector.detect_faces(data)
        for i in range(len(faces)):
            x1, y1, width, height = faces[i]['box']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            uni_size = cv2.resize(data[y1:y2, x1:x2], (160, 160))
            cv2.imwrite(f'/home/akash-peace/Downloads/face_extracted_all/img_extracted_{index}{i}.jpg', uni_size)
    else:
        pass