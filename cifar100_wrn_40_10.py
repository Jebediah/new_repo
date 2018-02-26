import numpy as np
import sklearn.metrics as metrics
from sklearn.utils import shuffle

import wide_residual_network as wrn
from keras.datasets import cifar100
import keras.callbacks as callbacks
import keras.utils.np_utils as kutils
from keras.preprocessing.image import ImageDataGenerator
#from keras.utils import plot_model
from keras import optimizers

from keras import backend as K

batch_size = 100
nb_epoch = 100
img_rows, img_cols = 32, 32

(trainX, trainY), (testX, testY) = cifar100.load_data()

trainX = trainX.astype('float32')
trainX = (trainX - trainX.mean(axis=0)) / (trainX.std(axis=0))
testX = testX.astype('float32')
testX = (testX - testX.mean(axis=0)) / (testX.std(axis=0))

trainY = kutils.to_categorical(trainY)
testY = kutils.to_categorical(testY)

# split training data into training and validation sets
trainX, trainY = shuffle(trainX, trainY, random_state=34)

split = 40000
trainSetX = trainX[:split, :, :, :]
trainSetY = trainY[:split, :]
validX = trainX[split:, :, :, :]
validY = trainY[split:, :]

testgenerator = ImageDataGenerator(rotation_range=10,
                               width_shift_range=5./32,
                               height_shift_range=5./32,)

init_shape = (3, 32, 32) if K.image_dim_ordering() == 'th' else (32, 32, 3)

# For WRN-16-8 put N = 2, k = 8
# For WRN-28-10 put N = 4, k = 10
# For WRN-40-4 put N = 6, k = 4
model = wrn.create_wide_residual_network(init_shape, nb_classes=100, N=4, k=10, dropout=0.3)

model.summary()
#plot_model(model, "WRN_28_10.png", show_shapes=False)

adadelta = optimizers.adadelta()

model.compile(loss="categorical_crossentropy", optimizer=adadelta, metrics=["acc"])
print("Finished compiling")

#model.load_weights("weights/WRN-28-10 Weights.h5")
print("Model loaded.")

model.fit_generator(testgenerator.flow(trainSetX, trainSetY, batch_size=batch_size), steps_per_epoch=len(trainX) // batch_size, epochs=nb_epoch,
                   callbacks=[callbacks.ModelCheckpoint("weights/WRN-40-10 Weights.h5",
                                                        monitor="val_acc",
                                                        save_best_only=True,
                                                        verbose=1)],
                   validation_data=(validX, validY),
                   validation_steps=testX.shape[0] // batch_size)

model.save_weights("weights.h5")

predicted_x = model.predict(testX)
residuals = (np.argmax(predicted_x,1)!=np.argmax(testY,1))
labels = np.argmax(predicted_x,1)

id = [i for i in range(0, len(labels))]
labels = np.stack((id, labels))
np.savetxt("labels.csv", labels.transpose(), delimiter=",", fmt="%d,%d")

yPreds = model.predict(testX)
yPred = np.argmax(yPreds, axis=1)
yPred = kutils.to_categorical(yPred)
yTrue = testY

accuracy = metrics.accuracy_score(yTrue, yPred) * 100
error = 100 - accuracy
print("Accuracy : ", accuracy)
print("Error : ", error)
