import tensorflow as tf
from datetime import datetime
import numpy as np
import cv2
import os
from keras import layers
from keras.models import Sequential


def mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def printt(string):
    print(string)
    with open("log.txt", 'a') as f:
        f.write(f'{string}\n')


printt('speed test --> colab free')
printt('0:00:11.565765 --> create image 1000img/class x 2class = 2000img')
printt('0:12:25.589441 --> training 2000img 5epoch')
printt('0:00:00.086822 --> predict 1')
printt('0:00:00.025623 --> predict 2')
printt('')
printt('speed test --> Xeon E5-1620, RAM 24 GB, SSD sata')
printt('0:00:11.770384 --> create image 1000img/class x 2class = 2000img')
printt('0:04:09.808607 --> training 2000img 5epoch')
printt('0:00:00.272243 --> predict 1')
printt('0:00:00.026496 --> predict 2')
printt('')
printt('')

mkdir('image')
mkdir('image/a')
mkdir('image/b')

t1 = datetime.now()
for name in range(1000):
    r = np.random.rand(255, 255, 3)
    r = r * 255
    r = np.array(r, dtype=np.uint8)
    cv2.imwrite(f'image/a/{name}.png', r)
    r[::] = r[:1:]
    cv2.imwrite(f'image/b/{name}.png', r)
t2 = datetime.now()
printt(f'{t2 - t1} --> create image 1000img/class x 2class = 2000img')

normalization_layer = layers.Rescaling(1. / 255)
train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    'image',
    validation_split=0.2,
    subset="both",
    seed=123,
    image_size=(255, 255),
    batch_size=32)

class_names = train_ds.class_names
model = Sequential([
    layers.Rescaling(1. / 255, input_shape=(255, 255, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names))
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

t1 = datetime.now()
model.fit(train_ds, validation_data=val_ds, epochs=5)
t2 = datetime.now()
printt(f'{t2 - t1} --> training 2000img 5epoch')

r = np.random.rand(255, 255, 3)
r = r * 255
r = np.array(r, dtype=np.uint8)
img_array = tf.keras.utils.img_to_array(r)
img_array = tf.expand_dims(img_array, 0)
t1 = datetime.now()
predictions = model.predict_on_batch(img_array)
t2 = datetime.now()
printt(f'{t2 - t1} --> predict 1')
score = tf.nn.softmax(predictions[0])
print(class_names[np.argmax(score)], 100 * np.max(score))

r[::] = r[:1:]
img_array = tf.keras.utils.img_to_array(r)
img_array = tf.expand_dims(img_array, 0)
t1 = datetime.now()
predictions = model.predict_on_batch(img_array)
t2 = datetime.now()
printt(f'{t2 - t1} --> predict 2')
score = tf.nn.softmax(predictions[0])
print(class_names[np.argmax(score)], 100 * np.max(score))
