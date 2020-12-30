from tensorflow import keras
import numpy as np
from PIL import Image

dataset = np.load('imagesdataset.npz', allow_pickle=True)
myImages = dataset['DataX']
myLabels = dataset['DataY']

print(type(myImages))

classNames = ['triangle', 'square', 'circle']

myImages = myImages / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(50, 50)),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(myImages, myLabels, epochs=100)

testImages = []
for i in range(1, 7):
    file = Image.open('tests/' + str(i) + '.jpg').convert('L').resize((50, 50))
    testImages.append(np.array(file))

# print(np.array(testImages).shape)
testImages = np.array(testImages) / 255.0
# print(testImages.shape)

predictions = model.predict(testImages)
print(predictions)
for i in range(len(predictions)):
    print(i+1, ':', classNames[np.argmax(predictions[i])], predictions[i][np.argmax(predictions[i])]*100, '%')