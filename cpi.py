from collections import Counter
import matplotlib.pyplot as plt

# Leer el archivo de ensamblador
with open('output.asm') as f:
    lines = f.readlines()

# Filtrar y contar instrucciones
instructions = [line.split()[0] for line in lines if len(line.split()) > 0 and line.split()[0].endswith(':') == False]
instruction_counts = Counter(instructions)

# Graficar las instrucciones más comunes
most_common_instructions = instruction_counts.most_common(10)
instruction_labels, instruction_values = zip(*most_common_instructions)

plt.figure(figsize=(10, 6))
plt.pie(instruction_values, labels=instruction_labels, autopct='%1.1f%%')
plt.title('Most Common Assembly Instructions')
plt.show()