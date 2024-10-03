import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import cv2
import time
def mapper(value):
    return reverse_mapping[value]

# Directory paths
directory = 'rpc_videos'
# Mapping labels
mapping = {"rock": 0, "paper": 1, "scissors": 2, "nothing": 3}
dataset = []

for label, class_id in mapping.items():
    class_directory = os.path.join(directory, label)
    for video_file in os.listdir(class_directory):
        if video_file.endswith(".mp4"):
            video_path = os.path.join(class_directory, video_file)
            cap = cv2.VideoCapture(video_path)
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1

                if frame_count % 2 == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (224, 224))

                    frame = img_to_array(frame)
                    frame = frame / 255.0
                    dataset.append([frame, class_id])
            cap.release()

data, labels = zip(*dataset)

# Count the number of frames for each class
class_counts = {class_id: sum(1 for label in labels if label == class_id) for class_id in mapping.values()}
print("Total no. of frames in dataset are", len(labels))
for label, count in class_counts.items():
    print(f"Number of frames of {label}: {count}")

# One-hot encode labels and convert data to numpy arrays
labels = to_categorical(labels, num_classes=len(mapping))
data = np.array(data)
labels = np.array(labels)

print("Data Shape:", data.shape)
print("Labels shape:", labels.shape)

# Split the dataset into training and testing sets
trainx, testx, trainy, testy = train_test_split(data, labels, test_size=0.2, random_state=44)

# Data augmentation
datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    fill_mode="nearest"
)

# Load MobileNetV2 without the top layers
pretrained_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet', pooling='avg')
pretrained_model.trainable = False

# Define the model
inputs = pretrained_model.input
x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

train_start = time.time()

# Train the model
his = model.fit(datagen.flow(trainx, trainy, batch_size=32), validation_data=(testx, testy), epochs=10)

train_end = time.time()

# Save the model
model.save('rock_paper_scissors_model1.h5')


eval_start = time.time()
# Evaluate the model
y_pred = model.predict(testx)
pred = np.argmax(y_pred, axis=1)
ground = np.argmax(testy, axis=1)
print(classification_report(ground, pred))
eval_end = time.time()

get_acc = his.history['accuracy']
value_acc = his.history['val_accuracy']
get_loss = his.history['loss']
validation_loss = his.history['val_loss']


print("For Learning Rate = 0.0001 batch size=32  epoch=10 Total Training Time: ",train_end-train_start)
print("For Learning Rate = 0.0001 batch size=32 epoch=10 Total Evaluating Time: ",eval_end-eval_start)


epochs = range(len(get_acc))
plt.plot(epochs, get_acc, 'r', label='Accuracy of Training data')
plt.plot(epochs, value_acc, 'b', label='Accuracy of Validation data')
plt.title('Training vs validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()


epochs = range(len(get_loss))
plt.plot(epochs, get_loss, 'r', label='Loss of Training data')
plt.plot(epochs, validation_loss, 'b', label='Loss of Validation data')
plt.title('Training vs validation loss')
plt.legend(loc=0)
plt.figure()

plt.show()


reverse_mapping = {0: 'rock', 1: 'paper', 2: 'scissors', 3: 'nothing'}

