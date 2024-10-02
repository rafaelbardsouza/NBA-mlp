import pandas as pd
import math
import random

dataset = pd.read_csv('./assets/dataset.csv')

# Normalização dos dados em float
performance_dummies = pd.get_dummies(dataset['Performance'], prefix='', prefix_sep='')
tm_dummies = pd.get_dummies(dataset['Tm'], prefix='Tm')
pos_dummies = pd.get_dummies(dataset['Pos'], prefix='Pos')
dataset = pd.concat([dataset, performance_dummies, tm_dummies, pos_dummies], axis=1)
players = dataset['Player']
dataset.drop(['Player', 'Performance', 'Tm', 'Pos'], axis=1, inplace=True)
dataset.replace({False: 0, True: 1}, inplace=True)
dataset = dataset.applymap(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x)
dataset = dataset.apply(pd.to_numeric, errors='coerce')
dataset = dataset.fillna(0)  # Replace NaNs with 0 to prevent calculation issues

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + math.exp(-max(min(x, 500), -500)))  # Clip to prevent overflow

def sigmoid_derivative(x):
    return x * (1 - x)

def loss_function(y_true, y_pred):
    return 0.5 * (y_true - y_pred) ** 2

# Initialize weights
def initialize_weights(input_size, hidden_size, output_size):
    limit_input_hidden = math.sqrt(6 / (input_size + hidden_size))
    limit_hidden_output = math.sqrt(6 / (hidden_size + output_size))
    
    weights_input_hidden = [[random.uniform(-limit_input_hidden, limit_input_hidden) for _ in range(hidden_size)] for _ in range(input_size)]
    weights_hidden_output = [[random.uniform(-limit_hidden_output, limit_hidden_output) for _ in range(output_size)] for _ in range(hidden_size)]
    
    return weights_input_hidden, weights_hidden_output

# Backpropagation
def backpropagation(dataset, weights_input_hidden, weights_hidden_output, learning_rate=0.01, epochs=10):
    for epoch in range(epochs):
        total_loss = 0
        for index, row in dataset.iterrows():
            input_layer = row.values[:-1]
            target = row.values[-1]

            # Forward pass
            hidden_layer = [sigmoid(sum([input_layer[i] * weights_input_hidden[i][j] for i in range(input_size)])) for j in range(hidden_size)]
            output_layer = [sigmoid(sum([hidden_layer[j] * weights_hidden_output[j][k] for j in range(hidden_size)])) for k in range(output_size)]

            # Loss calculation
            loss = loss_function(target, output_layer[0])
            total_loss += loss

            # Backward pass
            output_error = [target - output_layer[k] for k in range(output_size)]
            output_delta = [output_error[k] * sigmoid_derivative(output_layer[k]) for k in range(output_size)]

            hidden_error = [sum([output_delta[k] * weights_hidden_output[j][k] for k in range(output_size)]) for j in range(hidden_size)]
            hidden_delta = [hidden_error[j] * sigmoid_derivative(hidden_layer[j]) for j in range(hidden_size)]

            # Update weights
            for j in range(hidden_size):
                for k in range(output_size):
                    weights_hidden_output[j][k] += learning_rate * output_delta[k] * hidden_layer[j]

            for i in range(input_size):
                for j in range(hidden_size):
                    weights_input_hidden[i][j] += learning_rate * hidden_delta[j] * input_layer[i]

        # Print loss after every epoch to track progress
        print(f'Epoch: {epoch+1}/{epochs} | Total Loss: {total_loss}')

# Define the size of each layer
input_size = dataset.shape[1] - 1
hidden_size = 20
output_size = 1

# Initialize the weights
weights_input_hidden, weights_hidden_output = initialize_weights(input_size, hidden_size, output_size)

# Run backpropagation
backpropagation(dataset, weights_input_hidden, weights_hidden_output)
