import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model("facetracking_model/GTG_tracker_dir8")
tflite_model = converter.convert()

with open('tflite_model/model.tflite', 'wb') as f:
  f.write(tflite_model)