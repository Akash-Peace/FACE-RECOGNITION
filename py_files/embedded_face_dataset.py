from numpy import load, expand_dims, asarray, savez_compressed
from keras.models import load_model
def get_embedding(model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = expand_dims(face_pixels, axis=0)
    prediction = model.predict(samples)
    return prediction
data = load('/home/akash-peace/Downloads/arya_jon_face_dataset.npz')
trainx, trainy, testx, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
model = load_model('/home/akash-peace/Downloads/facenet_keras.h5')
embed_trainx = []
for face_pixels in trainx:
    embedding = get_embedding(model, face_pixels)
    embed_trainx.append(embedding)
embed_trainx = asarray(embed_trainx)
embed_testx = []
for face_pixels in testx:
    embedding = get_embedding(model, face_pixels)
    embed_testx.append(embedding)
embed_testx = asarray(embed_testx)
savez_compressed('/home/akash-peace/Downloads/arya_jon_embedded_face_dataset.npz', embed_trainx, trainy, embed_testx, testy)