from model.analize import Analize
from model.data_loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import itertools as it
from pandas import DataFrame


X_trained = ['Milk', 'Frozen']
label = ['Customer']
from model.preprocess import Preprocess

data = DataLoader('./data.csv', ';', ',')
file = data.read_file()




prepared = Preprocess().prepare_input(X_trained, file)




model = Analize().fit(prepared)


rs = file['Milk']
rs = rs.to_frame(name='Milk')

rs2 = file['Frozen']
rs2 = rs2.to_frame(name='Frozen')

lab = file['Customer ']
lab = lab.to_frame(name='Customer')

rs3 = pd.DataFrame(model)
rs3.columns = ['Cluster']


data = pd.concat([lab, rs, rs2, rs3], axis=1, join='inner')
print(data)

groups = data.groupby('Cluster')

fig, ax = plt.subplots()
ax.margins(0.1)
plt.scatter(rs, rs2)

for lab, group in groups:
    ax.plot(group.Milk, group.Frozen, marker='*', linestyle='', ms=10, label=lab)
    def labels(Milk, Frozen, Customer, ax):
        a = pd.concat({'Milk': Milk, 'Frozen': Frozen, 'Customer': Customer}, axis=1)
        for i, point in a.iterrows():
            ax.text(point['Milk'], point['Frozen'], str(point['Customer']))
    labels (group.Milk, group.Frozen, group.Customer, ax)
ax.legend()
plt.show()

