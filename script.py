import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

#Load train data
train_x = np.delete(np.genfromtxt('./input/train_x.csv', delimiter=','), 0,1)
train_y = np.delete(np.genfromtxt('./input/train_y.csv', delimiter=',')[1:], 0, 1)
train_x = train_x.reshape(train_x.shape[0], 32, 32, 3)

# Split the data
tr_imgs, val_imgs, tr_labels, val_labels = train_test_split(
        train_x, train_y, test_size=0.12, stratify=train_y)

# Normalize pixel values to between 0 and 1
train_images, val_images = tr_imgs / 255.0, val_imgs / 255.0

# Load model without classifier/fully connected layers
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Choose trainable layers
counter = 0 
for layer in vgg16.layers:
    counter = counter + 1
    layer.trainable = counter > 12
    
vgg16.summary()
model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.8))
model.add(Dense(1, activation = 'sigmoid' ))
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=1e-5), metrics=['accuracy'])
model.summary()

history = model.fit(x=train_images, y =tr_labels, epochs=11, validation_data=(val_images, val_labels))

# def show_results(history):
#     accuracy = history.history['accuracy']
#     val_accuracy = history.history['val_accuracy']
#     loss = history.history['loss']
#     val_loss = history.history['val_loss']
#     epochs = range(1,len(accuracy)+1)

#     plt.plot(epochs, accuracy, "bo", label="Training accuracy")
#     plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
#     plt.legend()
#     plt.title("Training and validation accuracy")
#     plt.figure()

#     plt.plot(epochs, loss, "bo", label="Training loss")
#     plt.plot(epochs, val_loss, "b", label="Validation loss")
#     plt.legend()
#     plt.title("Training and validation loss")
#     plt.figure()

#     plt.tight_layout()

#     plt.show()

# show_results(history)

test_x = np.delete(np.genfromtxt('./input/test_x.csv', delimiter=','), 0,1)
test_x = test_x.reshape(test_x.shape[0], 32, 32, 3)

prediction = map (lambda row : round(row[0]), model.predict(test_x))
def create_submission(predictions, filename):
    with open(filename + '.csv', 'w') as solution_file:
        solution_file.write('id,target\n')
        for i, prediction in enumerate(predictions):
            prediction = prediction
            solution_file.write(f"{i},{prediction}\n")

create_submission(prediction, 'submission')