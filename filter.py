import pandas as pd
from collections import Counter

# le dataset
df = pd.read_csv('pdb_data_no_dups.csv.zip', compression='zip').merge(pd.read_csv('pdb_data_seq.csv.zip', compression='zip'), how='inner', on='structureId').drop_duplicates(["sequence"]) # ,"classification"
# remove linhas que não possuem rótulos
df = df[[type(c) == type('') for c in df.classification.values]]
df = df[[type(c) == type('') for c in df.sequence.values]]
# select proteins
df = df[df.macromoleculeType_x == 'Protein']
# mantém sequências de tamanho 50 a 1200 apenas
df = df.loc[df.residueCount_x>50]
df = df.loc[df.residueCount_x<1200]
# conta número de instâncias por classe
cnt = Counter(df.classification)
# mantém apenas famílias que possuem mais de 200 instâncias
classes  = {}
# ordena classes por quantia
sorted_classes = cnt.most_common()[:5]
# anda nas classes
for c in sorted_classes:
    # se tiver mais de 200, salva
    if c[1] > 200:
        # salva no dicionário
        classes[c[0]] = c[1]
for i, (classe, count) in enumerate(classes.items()):
    print(i+1, classe, '=', count)
# filtra dados dentro das famílias
df = df[[c in classes.keys() for c in df.classification]]
print("Total:", str(df.shape[0]))

df.to_csv("pdb_filtered_top5_data.csv.zip", index=False, compression="zip")