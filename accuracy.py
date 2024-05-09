import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pathlib

# Directorios de los modelos y datos de validación
model_dir = "my_model"
tflite_model_path = "saved_model/model_original.tflite"
validation_dir = 'Imagenes/val'

# Cargar el modelo TensorFlow
model = tf.saved_model.load(model_dir)

# Crear una función para hacer predicciones con el modelo cargado
def predict_with_tf_saved_model(model, input_tensor):
    infer = model.signatures['serving_default']
    predictions = infer(input_tensor)
    return predictions['output_0']  # Ajusta esto según el nombre de la salida en tu modelo

# Cargar el modelo TFLite
interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Preparar los datos de validación
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(96, 96),
    class_mode='binary',
    color_mode='grayscale',
    shuffle=False  # Importante para comparar resultados
)

# Evaluar el modelo TensorFlow utilizando un enfoque de predicción manual
correct_predictions = 0
total_predictions = 0
for x, y_true in validation_generator:
    input_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
    preds = predict_with_tf_saved_model(model, input_tensor)
    preds = tf.sigmoid(preds)  # Aplica sigmoid si tu modelo finaliza con logits
    predicted_classes = tf.round(preds)  # Redondear a 0 o 1
    correct_predictions += tf.reduce_sum(tf.cast(tf.equal(predicted_classes, tf.expand_dims(y_true, axis=1)), tf.int32))
    total_predictions += x.shape[0]
    if total_predictions >= len(validation_generator.labels):
        break

tf_accuracy = correct_predictions / total_predictions
print("TensorFlow Model Accuracy:", tf_accuracy.numpy())

# Función para evaluar el modelo TFLite
def evaluate_tflite(interpreter, validation_generator):
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    total_seen = 0
    num_correct = 0

    for x, y_true in validation_generator:
        for i in range(x.shape[0]):
            input_data = x[i:i+1]
            y_true_single = y_true[i:i+1]

            interpreter.set_tensor(input_index, input_data)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_index)

            if predictions[0] > 0.5:
                prediction = 1
            else:
                prediction = 0

            if prediction == y_true_single:
                num_correct += 1
            total_seen += 1

        if total_seen >= len(validation_generator.labels):
            break

    return num_correct / total_seen

# Evaluar el modelo TFLite
tflite_accuracy = evaluate_tflite(interpreter, validation_generator)
print("TFLite Model Accuracy:", tflite_accuracy)

# Graficar las comparaciones
plt.bar(['TensorFlow', 'TFLite'], [tf_accuracy.numpy(), tflite_accuracy])
plt.title('Comparison of Model Accuracies')
plt.ylabel('Accuracy')
plt.show()
