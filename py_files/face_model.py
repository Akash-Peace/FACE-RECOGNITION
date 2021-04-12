import pickle
from numpy import load
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC

data = load('/home/akash-peace/Downloads/arya_jon_embedded_face_dataset.npz')
trainx, trainy, testx, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
in_encoder = Normalizer(norm='l2')
testx = testx.reshape(11, 128)  # just for testing purpose
trainx = trainx.reshape(47, 128)
trainx = in_encoder.transform(trainx)
testx = in_encoder.transform(testx)  # just for testing purpose
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
out_encoder.fit(testy)  # just for testing purpose
testy = out_encoder.transform(testy)  # just for testing purpose
model = SVC(kernel='linear', probability=True)
model.fit(trainx, trainy)
pickle.dump(model, open('/home/akash-peace/Downloads/finalized_model.sav', 'wb'))