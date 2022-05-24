import tensorflow as tf #Importerar tensorflow

print("Tensorflow version:", tf.__version__) #Skriver ut tensorflow versionen, användes bara för att se att det är installerat korrekt

TRAIN_DIRECTORY_LOCATION = r'Datasets\face_dataset_train_images' #Variabel som innehåller sökvägen för training datan
VAL_DIRECTORY_LOCATION = r'Datasets\face_dataset_val_images' #Variabel som innehåller sökvägen för validation datan

IMG_SIZE=(250, 250) #Variabel för bildstorleken som bilderna skalas ner till
BATCH_SIZE = 16 #Variabel för batch storleken, alltså 16 bilder per steg i epoken


train_dataset = tf.keras.utils.image_dataset_from_directory(TRAIN_DIRECTORY_LOCATION, #Importerar träningsdatan från sökvägen
                                             shuffle=True, #Blandar ordningen av bilderna
                                             batch_size=BATCH_SIZE, #Sätter batch storleken
                                             image_size=IMG_SIZE) #Sätter bildstorleken

validation_dataset = tf.keras.utils.image_dataset_from_directory(VAL_DIRECTORY_LOCATION, #Importerar validation datan från sökvägen
                                                  shuffle=True, #Blandar ordningen av bilderna
                                                  batch_size=BATCH_SIZE, #Sätter batch storleken
                                                  image_size=IMG_SIZE) #Sätter bildstorleken

class_names = train_dataset.class_names #Hämta labels för bilderna, alltså left, right och forward

      

data_augmentation = tf.keras.Sequential([ #Skapar fler bilder med de befintliga bilderna genom att ändra bildernas egenskaper
  tf.keras.layers.experimental.preprocessing.RandomContrast(factor=0.2), #Ändrar bildernas kontrast  
      ])

preprocess_input = tf.keras.applications.mobilenet.preprocess_input #Importerar funktion från mobilenet som skalar om pixelvärderna i bilderna för att fungera med basmodellen

#Skapar basmodellen från den förtränade modellen mobilenet
IMG_SHAPE = (250, 250, 3) #Variabel för bildstorleken för bilderna i den förtränade modellen, rgb
base_model = tf.keras.applications.MobileNet(input_shape=IMG_SHAPE, #Sätter bildstorleken
                                               include_top=False, #Gör att de helt anslutna utgångsskikten för basmodellen för att göra predictions inte laddas, vilket gör att ett nytt utgångsskikt kan läggas till och tränas
                                               weights='imagenet') #För att ladda in basmodellen förladdad med vikter tränade på ImageNet

image_batch, label_batch = next(iter(train_dataset)) #Kategoriserar all träningsdata, ger dem labels
feature_batch = base_model(image_batch) #Utför feature extraction
print(feature_batch.shape)#Skriver ut feature batch shape

base_model.trainable = False #Fryser convolution layers för att hindra vikterna från att bli uppdaterade under träningen

base_model.summary() #Skriver ut basmodellens arkitektur

global_average_layer = tf.keras.layers.GlobalAveragePooling2D() #Lägger till pooling lager för att reducera dimensioner och omvandlar till en 1024-element feature vektor per bild för att minska beräkningskraften som krävs
feature_batch_average = global_average_layer(feature_batch) #Utför feature extraction
print(feature_batch_average.shape) #Skriver ut feature batch average shape


prediction_layer = tf.keras.layers.Dense(3, activation = 'softmax') #Lägger till 3 dense layers för att vi har 3 klasser och sätter aktiveringsfunktionen till softmax då den är skapad för multiclass classification
prediction_batch = prediction_layer(feature_batch_average) #Utför feature extraction
print(prediction_batch.shape) #Skriver ut prediction batch shape


#Bygger den slutliga modellen genom att sammanställa data augmentation, rescaling, basmodellen och feature extractorn.
inputs = tf.keras.Input(shape=(250, 250, 3)) #Sätter input shape till 250x250, rgb
x = data_augmentation(inputs) #Lägger till data augmentation
x = preprocess_input(x) #Lägger till pre processing
x = base_model(x, training=False) #Lägger till basmodellen och påminner tensorflow att vi inte vill träna basmodellen
x = global_average_layer(x) #Lägger till data från global average layer
x = tf.keras.layers.Dropout(0.2)(x) #Lägger till dropout layer för att förhindra overfitting
outputs = prediction_layer(x) #Lägger till prediction layer
model = tf.keras.Model(inputs, outputs) #Skapar den slutliga modellen



base_learning_rate = 0.00005 #Variabel för att bestämma learning raten modellen använder
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate), #Lägger till optimizern Adam
              loss='sparse_categorical_crossentropy', #Lägger till loss funktionen sparse categorical crossentropy
              metrics=['accuracy']) #Används för att räkna ut accuracy genom att jämföra prediction mot det rätta svaret

model.summary() #Skriver ut modellens arkitektur

loss0, accuracy0 = model.evaluate(validation_dataset) #Räknar ut modellens loss och accuracy utan någon träning
print("initial loss: {:.2f}".format(loss0)) #Skriver ut initial loss
print("initial accuracy: {:.2f}".format(accuracy0)) #Skriver ut initial accuracy

EPOCHS = 5 #Variabel för antalet epoker som ska köras
#Modellen tränas
history = model.fit(train_dataset, #Använder träningsdata för träningen
                    epochs=EPOCHS, #Antalet epoker som ska köras
                    validation_data=validation_dataset) #Använder validation datan för validering

model.save("facetracking_model/GTG_tracker_dir8") #Sparar modellen i den angivna sökvägen

#tflite_models_dir = ("tflite_model")
#tflite_model_file = tflite_models_dir/'model1.tflite'
#tflite_model_file.write_bytes(tflite_model)

