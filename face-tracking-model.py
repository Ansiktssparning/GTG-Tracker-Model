import tensorflow as tf # importerar tensorflow

print("Tensorflow version:", tf.__version__)  # skriver ut tensorflow versionen 

TRAIN_DIRECTORY_LOCATION = r'Datasets\face_dataset_train_images' # bibliotek med träningsdata
VAL_DIRECTORY_LOCATION = r'Datasets\face_dataset_val_images' # bibliotek med valideringsdata

IMG_SIZE=(167, 250) # storlek vi vill ha på bilderna
BATCH_SIZE = 32 # variabel för batch storlek


train_dataset = tf.keras.utils.image_dataset_from_directory(TRAIN_DIRECTORY_LOCATION, # skapar träningsdataset
                                             shuffle=True, # blandar ordning av bilderna
                                             batch_size=BATCH_SIZE, # sätter batch storlek
                                             image_size=IMG_SIZE) # sätter bild storlek

validation_dataset = tf.keras.utils.image_dataset_from_directory(VAL_DIRECTORY_LOCATION, #skapar valideringsdataset
                                                  shuffle=True, # blandar ordning av bilderna
                                                  batch_size=BATCH_SIZE, # sätter batch storlek
                                                  image_size=IMG_SIZE) # sätter bild storlek

class_names = train_dataset.class_names # plockar ut klasserna från träningsdatan

      

data_augmentation = tf.keras.Sequential([ # data augmentation för att utöka datasetet och göra modellen mer generell
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2), # roterar bilderna
  tf.keras.layers.experimental.preprocessing.RandomZoom(-.3,-.1) # ändrar zoomen
  tf.keras.layers.experimental.preprocessing.RandomContrast(factor=0.2), # ändrar kontrasten
      ])

preprocess_input = tf.keras.applications.mobilenet.preprocess_input

IMG_SHAPE = (167, 250,3) # bildstorlek inklusive färg
base_model = tf.keras.applications.MobileNet(input_shape=IMG_SHAPE, #laddar bas modellen för transferlearning
                                               include_top=False,
                                               weights='imagenet')

image_batch, label_batch = next(iter(train_dataset)) 
feature_batch = base_model(image_batch)
print(feature_batch.shape) # plockar ut och skriver ut feature delen av modellen

base_model.trainable = False

base_model.summary()

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)


prediction_layer = tf.keras.layers.Dense(3, activation = 'softmax')
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)



inputs = tf.keras.Input(shape=IMG_SHAPE)
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

#loss0, accuracy0 = model.evaluate(validation_dataset)
#print("initial loss: {:.2f}".format(loss0))
#print("initial accuracy: {:.2f}".format(accuracy0))

EPOCHS = 5
history = model.fit(train_dataset,
                    epochs=EPOCHS,
                    validation_data=validation_dataset)

print(class_names)

model.save("facetracking_model/GTG_tracker_dir6.h5") 