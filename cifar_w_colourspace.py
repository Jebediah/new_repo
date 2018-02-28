import numpy as np
import sklearn.metrics as metrics
from sklearn.utils import shuffle

import wide_residual_network as wrn
from keras.datasets import cifar100
import keras.callbacks as callbacks
import keras.utils.np_utils as kutils
from keras.layers import Input
from keras.models import Model
from keras.layers.convolutional import Convolution2D
from keras.models import load_model

from keras.preprocessing.image import ImageDataGenerator
#from keras.utils import plot_model
from keras import optimizers

from keras import backend as K

batch_size = 128
nb_epoch = 32
img_rows, img_cols = 32, 32

(trainX, trainY), (testX, testY) = cifar100.load_data()

trainX = trainX.astype('float32')
trainX = (trainX - trainX.mean(axis=0)) / (trainX.std(axis=0))
testX = testX.astype('float32')
testX = (testX - testX.mean(axis=0)) / (testX.std(axis=0))

trainY = kutils.to_categorical(trainY)
testY = kutils.to_categorical(testY)

# split training data into training and validation sets
trainX, trainY = shuffle(trainX, trainY, random_state=4)

split = 45000
trainSetX = trainX[:split, :, :, :]
trainSetY = trainY[:split, :]
validX = trainX[split:, :, :, :]
validY = trainY[split:, :]

testgenerator = ImageDataGenerator(rotation_range=10,
                               width_shift_range=5./32,
                               height_shift_range=5./32,
                               horizontal_flip=True,
                               vertical_flip=False)

init_shape = (3, 32, 32) if K.image_dim_ordering() == 'th' else (32, 32, 3)

# For WRN-16-8 put N = 2, k = 8
# For WRN-28-10 put N = 4, k = 10
# For WRN-40-4 put N = 6, k = 4
#model = wrn.create_wide_residual_network(init_shape, nb_classes=100, N=4, k=10, dropout=0.3)

#model.summary()
#plot_model(model, "WRN_28_10.png", show_shapes=False)

opt = optimizers.sgd(lr=0.1, momentum=0.9, decay=0.0005, nesterov=True)

#model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["acc"])
#print("Finished compiling")

#model.load_weights("weights/WRN-28-10 Weights.h5")
#print("Model loaded.")

#ip_remap = Input(init_shape)
#y = Convolution2D(10, (1, 1), padding='same', kernel_initializer='he_normal', use_bias=False)(ip_remap)
#y = Convolution2D(3, (1, 1), padding='same', kernel_initializer='he_normal', use_bias=False)(y)

#colourspace = Model(ip_remap, y)

#merged_model = Model(inputs=colourspace.input, outputs=model(colourspace.output))
merged_model = load_model("weights/WRN-28-10 Colour.h5")
#merged_model.load_weights("weights/WRN-28-10_Colour_final.h5")
merged_model.summary()
#merged_model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["acc"])
#print("Finished compiling")

merged_model.fit_generator(testgenerator.flow(trainSetX, trainSetY, batch_size=batch_size), steps_per_epoch=len(trainX) // batch_size, epochs=nb_epoch,
                   callbacks=[callbacks.ModelCheckpoint("weights/WRN-28-10 Colour.h5",
                                                        monitor="val_acc",
                                                        save_best_only=True,
                                                        verbose=1)],
                   validation_data=(validX, validY),
                   validation_steps=validX.shape[0] // batch_size)

#merged_model.save("weights/WRN-28-10_Colour_final.h5")

yPreds = merged_model.predict(testX)
residuals = (np.argmax(yPreds,1)!=np.argmax(testY,1))
labels = np.argmax(yPreds,1)
error = sum(residuals)/len(residuals)
accuracy = 1 - error
print("Accuracy : ", accuracy)

id = [i for i in range(0, len(labels))]
labels = np.stack((id, labels))
np.savetxt("repeat.csv", labels.transpose(), delimiter=",", fmt="%d,%d")

