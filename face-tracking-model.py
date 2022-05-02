import tensorflow as tf 
import tensorflow_addons as tfa

print("Tensorflow version:", tf.__version__)   

TRAIN_DIRECTORY_LOCATION = r'Datasets\face_dataset_train_images'
VAL_DIRECTORY_LOCATION = r'Datasets\face_dataset_val_images' 

IMG_SIZE=(250, 250)
BATCH_SIZE = 16


train_dataset = tf.keras.utils.image_dataset_from_directory(TRAIN_DIRECTORY_LOCATION,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE)

validation_dataset = tf.keras.utils.image_dataset_from_directory(VAL_DIRECTORY_LOCATION,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  image_size=IMG_SIZE)
#train_dataset = tf.image.resize(train_dataset, [250,250])
#validation_dataset = tf.image.resize(validation_dataset, [250,250])
class_names = train_dataset.class_names

      

data_augmentation = tf.keras.Sequential([
  #tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  #tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
  #tf.keras.layers.experimental.preprocessing.RandomZoom(-.3,-.1)
  tf.keras.layers.experimental.preprocessing.RandomContrast(factor=0.2),
  
      ])

preprocess_input = tf.keras.applications.mobilenet.preprocess_input

# Create the base model from the pre-trained model MobileNet V2
IMG_SHAPE = (250, 250,3)
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


prediction_layer = tf.keras.layers.Dense(3, activation = 'softmax')
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)



inputs = tf.keras.Input(shape=(250, 250, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)



base_learning_rate = 0.0005#changed from 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

loss0, accuracy0 = model.evaluate(validation_dataset)
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

EPOCHS = 3
history = model.fit(train_dataset,
                    epochs=EPOCHS,
                    validation_data=validation_dataset)

print(class_names)

model.save("facetracking_model/GTG_tracker_dir3")