import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter

# le dataset
# data = pd.read_csv("pdb_filtered_data.csv.zip", compression='zip')
data = pd.read_csv("pdb_filtered_top10_data.csv.zip", compression='zip')
# conta número de instâncias por classe
cnt = Counter(data.classification)
# mantém apenas famílias que possuem mais de 200 instâncias
classes  = {}
# ordena classes por quantia
sorted_classes = cnt.most_common()
# anda nas classes
for c in sorted_classes:
    # se tiver mais de 200, salva
    if c[1] > 200:
        # salva no dicionário
        classes[c[0]] = c[1]
fig, ax = plt.subplots(figsize=(13,5))
yticks = []
# hatches = ['//', '\\', '||', '|-', '++', 'xx', 'oo', 'OO', '..', '**']
for i, (classe, count) in enumerate(classes.items()):
    print(i+1, classe, '=', count)
    # plt.barh(i+1, count, 0.75, zorder=2, edgecolor="black", linewidth=1)#, hatch=hatches[i])
    plt.bar(i+1, count, 0.75, zorder=2, edgecolor="black", linewidth=1)#, hatch=hatches[i])
    if classe == 'STRUCTURAL GENOMICS, UNKNOWN FUNCTION':
        classe = "STRUCTURAL GENOMICS"
    classe = classe.replace(" ", "\n")
    print(classe)
    yticks.append(classe)
# ax.set_yticks(np.arange(1,len(yticks)+1), labels=yticks)
# ax.set_xticks(np.arange(0,10500,500))
# ax.set_xlabel('Número de Amostras')
# plt.ylim(0.3,10.8)
ax.set_xticks(np.arange(1,len(yticks)+1), labels=yticks)
ax.set_yticks(np.arange(0,10500,500))
ax.set_ylabel('Número de Amostras')
plt.xlim(0.4,10.6)
plt.title("Distribuição de Classes")
plt.grid(zorder=-10)
plt.tight_layout()
print(data)
plt.savefig("distribuicao.png", dpi=300)
plt.savefig("distribuicao.pdf", dpi=300)
plt.show()