import pandas as pd
import math
import random
import math
import random

dataset = pd.read_csv('./assets/dataset.csv')

# Normalização dos dados em float
performance_dummies = pd.get_dummies(dataset['Performance'], prefix='', prefix_sep='')
tm_dummies = pd.get_dummies(dataset['Tm'], prefix='Tm')
pos_dummies = pd.get_dummies(dataset['Pos'], prefix='Pos')
dataset = pd.concat([dataset, performance_dummies, tm_dummies, pos_dummies], axis=1)
dataset.drop(['Player', 'Performance', 'Tm', 'Pos'], axis=1, inplace=True)
dataset.replace({False: 0, True: 1}, inplace=True)
dataset = dataset.applymap(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x)
dataset = dataset.astype(float)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def initialize_weights(input_size, hidden_size, output_size):
    weights_input_hidden = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(input_size)]
    weights_hidden_output = [[random.uniform(-1, 1) for _ in range(output_size)] for _ in range(hidden_size)]

    return weights_input_hidden, weights_hidden_output

input_size = dataset.shape[1] 
hidden_size = 20
output_size = 1

weights_input_hidden, weights_hidden_output = initialize_weights(input_size, hidden_size, output_size)
