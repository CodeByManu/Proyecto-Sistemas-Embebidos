from collections import Counter
import matplotlib.pyplot as plt

def filter_instructions_by_labels(asm_file, start_label, end_label):
    with open(asm_file, 'r') as f:
        lines = f.readlines()

    in_section = False
    instructions = []

    for line in lines:
        if start_label in line:
            in_section = True
        if in_section:
            parts = line.split()
            if len(parts) > 0 and not parts[0].endswith(':'):
                instructions.append(parts[0])
        if end_label in line:
            in_section = False

    return instructions

def plot_instruction_pie_chart(instruction_counts, title):
    most_common_instructions = instruction_counts.most_common(10)
    instruction_labels, instruction_values = zip(*most_common_instructions)

    # Agrupar el resto de las instrucciones en "Otras"
    other_count = sum(instruction_counts.values()) - sum(instruction_values)
    instruction_labels += ('Others',)
    instruction_values += (other_count,)

    plt.figure(figsize=(10, 6))
    plt.pie(instruction_values, labels=instruction_labels, autopct='%1.1f%%')
    plt.title(title)
    plt.show()

# Filtrar instrucciones para cada subtask
quantization_instructions = filter_instructions_by_labels('output.asm', 'quantization_start', 'quantization_end')
conv2d_instructions = filter_instructions_by_labels('output.asm', 'conv2d_start', 'conv2d_end')
pooling_instructions = filter_instructions_by_labels('output.asm', 'pooling_start', 'pooling_end')
fullyconnected_instructions = filter_instructions_by_labels('output.asm', 'fullyconnected_start', 'fullyconnected_end')
response_instructions = filter_instructions_by_labels('output.asm', 'response_start', 'response_end')

# Contar instrucciones
quantization_counts = Counter(quantization_instructions)
conv2d_counts = Counter(conv2d_instructions)
pooling_counts = Counter(pooling_instructions)
fullyconnected_counts = Counter(fullyconnected_instructions)
response_counts = Counter(response_instructions)

# Graficar los resultados
plot_instruction_pie_chart(quantization_counts, 'Quantization Instructions')
plot_instruction_pie_chart(conv2d_counts, 'Conv2D Instructions')
plot_instruction_pie_chart(pooling_counts, 'Pooling Instructions')
plot_instruction_pie_chart(fullyconnected_counts, 'FullyConnected Instructions')
plot_instruction_pie_chart(response_counts, 'Response Processing Instructions')
