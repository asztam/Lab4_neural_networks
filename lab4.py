import numpy as np
import pandas as pd
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

df = pd.read_csv('textures_data.csv', sep=',')
data = df.to_numpy()

X = data[:, :-1].astype('float')
y = data[:, -1]

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y)

# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, onehot_encoded, test_size=0.3)

model = Sequential()
model.add(Dense(10, input_dim=72, activation='sigmoid'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.summary()

model.fit(X_train, y_train, epochs=100, batch_size=10, shuffle=True)

y_pred = model.predict(X_test)
y_pred_int = np.argmax(y_pred, axis=1)
y_test_int = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_test_int, y_pred_int)
print(cm)
