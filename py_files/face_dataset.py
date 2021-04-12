import os
from PIL import Image
from numpy import savez_compressed, asarray

def load_dataset(paths):
    x, y = [], []
    for path in paths:
        for filename in os.listdir(path):
            img = path + filename
            img = Image.open(img)
            img = img.convert('RGB')
            face = asarray(img)
            x.append(face)
            if path[44:50] == ('rain/a' or 'est/ar'):
                y.append('Arya Stark')
            else:
                y.append('Jon Snow')
    return asarray(x), asarray(y)
trainx, trainy = load_dataset(['/home/akash-peace/Downloads/face_extracted/train/arya_stark/', '/home/akash-peace/Downloads/face_extracted/train/jon_snow/'])
testx, testy = load_dataset(['/home/akash-peace/Downloads/face_extracted/test/arya_stark/', '/home/akash-peace/Downloads/face_extracted/test/jon_snow/'])
savez_compressed('/home/akash-peace/Downloads/arya_jon_face_dataset.npz', trainx, trainy, testx, testy)