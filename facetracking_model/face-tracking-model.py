import os
import tensorflow as tf 
import matplotlib as plt
import os
import numpy
import cv2
print("Tensorflow version:", tf.__version__)   

TRAIN_DIRECTORY_LOCATION = r'Datasets\face_dataset_train_images'
VAL_DIRECTORY_LOCATION = r'Datasets\face_dataset_val_images'
IMG_SIZE=(600,600)
BATCH_SIZE = 32

train_dataset = tf.keras.utils.image_dataset_from_directory(TRAIN_DIRECTORY_LOCATION,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE)

validation_dataset = tf.keras.utils.image_dataset_from_directory(VAL_DIRECTORY_LOCATION,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  image_size=IMG_SIZE)

class_names = train_dataset.class_names

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

preprocess_input = tf.keras.applications.mobilenet.preprocess_input

# Create the base model from the pre-trained model MobileNet V2
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNet(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape)

base_model.trainable = False

base_model.summary()

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

inputs = tf.keras.Input(shape=(600, 600, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

loss0, accuracy0 = model.evaluate(validation_dataset)
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

EPOCHS = 15
history = model.fit(train_dataset,
                    epochs=EPOCHS,
                    validation_data=validation_dataset)

model.save("models/GTG-tracker1")


# opencv object that will detect faces for us
#face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades 
#                                     + 'haarcascade_frontalface_default.xml')
#video_capture = cv2.VideoCapture(0)  # webcamera
#
#if not video_capture.isOpened():
#    print("Unable to access the camera")
#else:
#    print("Access to the camera was successfully obtained")
#
#print("Streaming started")
#while True:
#    # Capture frame-by-frame
#    ret, frame = video_capture.read()
#    if not ret:
#        print("Can't receive frame (stream end?). Exiting ...")
#        break
#
#    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#        gray,
#        scaleFactor=1.3,
#        minNeighbors=5,
#        minSize=(100, 100),
#        flags=cv2.CASCADE_SCALE_IMAGE
#    )
#
#    for (x, y, w, h) in faces:
#        # for each face on the image detected by OpenCV
#        # draw a rectangle around the face
#        cv2.rectangle(frame, 
#                      (x, y), # start_point
#                      (x+w, y+h), # end_point
#                      (255, 0, 0),  # color in BGR
#                      2) # thickness in px
#        
    # Display the resulting frame
#    cv2.imshow("Face detector - to quit press ESC", frame)
#
    # Exit with ESC
#    key = cv2.waitKey(1)
#    if key % 256 == 27: # ESC code
#        break
        
# When everything done, release the capture
#video_capture.release()
#cv2.destroyAllWindows()
#print("Streaming ended")