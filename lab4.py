import matplotlib.pyplot as plt

# Datos obtenidos del ESP32
total_inference_time = 1618  # Microsegundos
fc_time = 179  # Microsegundos
conv_time = 1399  # Microsegundos
pooling_time = 38  # Microsegundos
response_processing_time = 522  # Microsegundos

tasks = ['Total Inference', 'FullyConected', 'Convolution', 'Pooling']
times = [total_inference_time, fc_time, conv_time, pooling_time]

plt.bar(tasks, times)
plt.xlabel('Tasks')
plt.ylabel('Time (us)')
plt.title('Inference Time Breakdown')
plt.show()