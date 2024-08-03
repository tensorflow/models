import tensorflow as tf

def convert_tflite_to_pb(tflite_model_path, pb_model_path):
    # Load the TFLite model
    with open(tflite_model_path, 'rb') as f:
        tflite_model = f.read()

    # Convert the TFLite model to a TensorFlow model
    converter = tf.lite.TFLiteConverter.from_saved_model(tflite_model)
    tf_model = converter.convert()

    # Save the TensorFlow model as a .pb file
    tf.saved_model.save(tf_model, pb_model_path)

if __name__ == "__main__":
    tflite_model_path = "face_landmark.tflite"
    pb_model_path = "face_landmark.pb"
    convert_tflite_to_pb(tflite_model_path, pb_model_path)
