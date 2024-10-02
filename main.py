import pandas as pd

dataset = pd.read_csv('./assets/dataset.csv')

performance_dummies = pd.get_dummies(dataset['Performance'], prefix='', prefix_sep='')
tm_dummies = pd.get_dummies(dataset['Tm'], prefix='Tm')
pos_dummies = pd.get_dummies(dataset['Pos'], prefix='Pos')
dataset = pd.concat([dataset, performance_dummies, tm_dummies, pos_dummies], axis=1)
dataset.drop(['Player', 'Performance', 'Tm', 'Pos'], axis=1, inplace=True)
dataset.replace({False: 0, True: 1}, inplace=True)
dataset = dataset.applymap(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x)
dataset = dataset.astype(float)

print(dataset)