import tensorflow as tf
import pathlib
import numpy as np

# # Asumiendo que ya tienes tu modelo definido como 'model'
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(96, 96, 1)),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(256, activation='relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])
# # Función para calcular el peso de cada capa
# def get_layer_weights(model):
#     layer_weights = {}
#     for layer in model.layers:
#         layer_name = layer.name
#         weights = layer.count_params()
#         layer_weights[layer_name] = weights * 4 / 1024  # Tamaño en KB (asumiendo float32)
#     return layer_weights

# # Obtener el peso de cada capa
# layer_weights = get_layer_weights(model)

# # Imprimir el peso de cada capa
# total_weights = 0
# print("Peso de cada capa (en KB):")
# for layer_name, weight in layer_weights.items():
#     print(f"{layer_name}: {weight:.2f} KB")
#     total_weights += weight

# print(f"\nPeso total del modelo: {total_weights:.2f} KB")

# # También podemos obtener un resumen del modelo
# model.summary()

tflite_model_file_quant = pathlib.Path('saved_model/model_original.tflite')

def get_quantized_model_size(tflite_model_file):
    model_size = tflite_model_file.stat().st_size
    return model_size / 1024  # Tamaño en KB

# Obtener el tamaño del modelo cuantizado
quantized_model_size = get_quantized_model_size(tflite_model_file_quant)
print(f"Peso del modelo cuantizado: {quantized_model_size:.2f} KB")