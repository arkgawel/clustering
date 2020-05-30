from model.analize import Analize
from model.data_loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame


X_trained = ['Milk', 'Frozen']
label = ['Customer']
from model.preprocess import Preprocess

data = DataLoader('./data.csv', ';', ',')
file = data.read_file()

prepared = Preprocess().prepare_input(X_trained, file)
#print(prepared)



model = Analize().fit(prepared)
#print(model)

rs = file['Milk']
rs = rs.to_frame(name='Milk')
rs2 = file['Frozen']
rs2 = rs2.to_frame(name='Frozen')

rs3 = pd.DataFrame(model)
rs3.columns = ['Cluster']


data = pd.concat([rs, rs2, rs3], axis=1, join='inner')
print(data)

groups = data.groupby('Cluster')

fig, ax = plt.subplots()
ax.margins(0.1)
for name, group in groups:
    ax.plot(group.Milk, group.Frozen, marker='*', linestyle='', ms=10, label=name)
ax.legend()

plt.show()