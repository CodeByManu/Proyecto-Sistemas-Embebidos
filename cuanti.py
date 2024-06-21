import tensorflow as tf
import pathlib
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


validation_dir = 'Imagenes/val'

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(96, 96),
    batch_size=128,
    class_mode='binary',
    color_mode='grayscale'
)

# Directorio donde se guarda el modelo
export_dir = "saved_model"

# Función de datos representativos
def representative_data_gen():
    for _ in range(100):
        data, _ = next(validation_generator)
        yield [data]

# Convertir el modelo a TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # Entrada en int8
converter.inference_output_type = tf.int8  # Salida en int8

tflite_model_quant = converter.convert()

# Guardar el modelo cuantizado
tflite_model_file_quant = pathlib.Path('saved_model/model_quant.tflite')
tflite_model_file_quant.write_bytes(tflite_model_quant)

# Función para calcular el peso del modelo cuantizado
def get_quantized_model_size(tflite_model_file):
    model_size = tflite_model_file.stat().st_size
    return model_size / 1024  # Tamaño en KB

# Obtener el tamaño del modelo cuantizado
quantized_model_size = get_quantized_model_size(tflite_model_file_quant)
print(f"Peso del modelo cuantizado: {quantized_model_size:.2f} KB")

# Revisar el modelo usando el interpreter de TFLite

# Cargar el modelo de TFLite
interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
interpreter.allocate_tensors()

# Obtener detalles de entrada y salida
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)
print(output_details)

# Probar el modelo cuantizado
num_test_samples = 1000
y_interpreter_quant = np.zeros(num_test_samples)
y_out = np.zeros(num_test_samples)

for i in range(num_test_samples):
    # Obtener datos de prueba (modifica esta parte según tus datos de entrada)
    input_data, _ = next(validation_generator)
    
    # Asegurarse de que los datos de entrada estén en el rango y tipo correcto
    input_data = input_data * 255  # Desnormalizar
    input_data = input_data.astype(np.int8)  # Convertir a int8
    
    # Verifica que input_data tenga la forma correcta
    if input_data.shape[0] != 1:
        input_data = input_data[:1]  # Solo toma el primer batch
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    y_interpreter_quant[i] = interpreter.get_tensor(output_details[0]['index'])

# Puedes agregar código adicional para evaluar y comparar el rendimiento del modelo cuantizado
