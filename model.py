import csv
import cv2
import sklearn
import numpy as np
import matplotlib.pyplot as plt

# load driving_log data
lines = []
with open('../driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# set image paths and the values of steering
images = []
measurements = []
for line in lines:
    for i in range(3):
        image = line[i]
        images.append(image)
        correction = 0.2
        if i == 0:
            measurement = float(line[3])  # center camera
        elif i == 1:
            measurement = float(line[3]) + correction  # left camera
        elif i == 2:
            measurement = float(line[3]) - correction  # right camera
        measurements.append(measurement)


# generator for getting training samples
def generator(samples, measurements, batch_size=32):
    num_samples = len(samples)
    while True:
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            batch_measurements = measurements[offset:offset+batch_size]

            imgs = []
            steering = []
            for i, batch_sample in enumerate(batch_samples):
                # set original data
                img = cv2.imread(batch_sample)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imgs.append(img)
                steering.append(batch_measurements[i])
                # set flipped image and steering
                imgs.append(cv2.flip(img, 1))
                steering.append(-1.0*batch_measurements[i])

                # mix-up
                """
                if i+1 < len(batch_samples):
                    lam = np.random.rand()
                    img1 = cv2.imread(batch_samples[i])
                    img2 = cv2.imread(batch_samples[i+1])
                    imgs.append(cv2.addWeighted(img1, lam, img2, 1-lam, 0))
                    steering.append(lam * batch_measurements[i] + (1-lam) * batch_measurements[i+1])
                """
                """
                for j in range(1):
                    lam = np.random.rand()
                    idx1 = np.random.randint(len(samples))
                    idx2 = np.random.randint(len(samples))
                    img1 = cv2.imread(samples[idx1])
                    img2 = cv2.imread(samples[idx2])
                    imgs.append(cv2.addWeighted(img1, lam, img2, 1-lam, 0))
                    steering.append(lam * measurements[idx1] + (1-lam) * measurements[idx2])
                """
            
            X_train = np.array(imgs)
            y_train = np.array(steering)
            yield sklearn.utils.shuffle(X_train, y_train)

# data preparation
BATCH_SIZE=64
from sklearn.model_selection import train_test_split
(train_samples, validation_samples, train_measurements, validation_measurements) = train_test_split(images, measurements,test_size=0.2)
train_generator = generator(train_samples, train_measurements, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, validation_measurements, batch_size=BATCH_SIZE)

# construct NN model
# base model: https://arxiv.org/pdf/1604.07316.pdf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(BatchNormalization())
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(BatchNormalization())
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(BatchNormalization())
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(BatchNormalization())
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(100))
model.add(BatchNormalization())
model.add(Dense(50))
model.add(Dense(10))
model.add(BatchNormalization())
model.add(Dense(1))

# for fine-tuning
#model.load_weights('model.h5')

# for plot model
from keras.utils import plot_model
plot_model(model, to_file='model.png')

# training
model.compile(loss='mse', optimizer='adam')
history = model.fit_generator(train_generator, samples_per_epoch=int(len(train_samples) / BATCH_SIZE), 
          validation_data=validation_generator, validation_steps=int(len(validation_samples) / BATCH_SIZE), nb_epoch=10)

# save the trained model
model.save('model.h5')

# print the keys contained in the history object
print(history.history.keys())

# plot the training and validation loss for each epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
