import numpy as np
import matplotlib.pyplot as plt

# Datos proporcionados
subtasks = ['Conv1', 'Pool1', 'Conv2', 'Pool2', 'Conv3', 'Pool3', 'FC1', 'FC2', 'FC3']
n_mac = np.array([1272384, 106032, 9331200, 48600, 7372800, 19200, 1644800, 33024, 512])  # Número de operaciones MAC
total_bytes = np.array([150592, 176720, 100144, 80288, 41088, 32000, 6656, 384, 131])  # Total de bytes procesados
times_us = np.array([50000, 30000, 50000, 30000, 50000, 30000, 10000, 5000, 2000])  # Tiempo de ejecución en microsegundos

# Convertir tiempos a segundos
times_s = times_us * 1e-6

# Calcular OI y Performance
oi = n_mac / total_bytes
performance = n_mac / times_s / 1e9  # GFLOPs

# Parámetros del sistema (deben ser ajustados con valores reales del microcontrolador)
memory_bandwidth = 10  # en GB/s (ejemplo)
peak_flops = 1  # en GFLOPs (ejemplo)

# Crear el Roofline plot
plt.figure(figsize=(10, 8))

# Líneas de techo
oi_range = np.logspace(-2, 2, 100)
roofline_memory = memory_bandwidth * oi_range
roofline_compute = np.ones_like(oi_range) * peak_flops

plt.plot(oi_range, roofline_memory, label='Memory Bandwidth Roof', linestyle='--')
plt.plot(oi_range, roofline_compute, label='Compute Roof', linestyle='--')

# Puntos para los subtasks
plt.scatter(oi, performance, color='red')
for i, subtask in enumerate(subtasks):
    plt.annotate(subtask, (oi[i], performance[i]))

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Operational Intensity (FLOPs/Byte)')
plt.ylabel('Performance (GFLOPs)')
plt.title('Roofline Plot')
plt.legend()
plt.grid(True, which="both", ls="--")

plt.show()
