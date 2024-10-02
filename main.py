import pandas as pd
import math
import random

# Load dataset
dataset = pd.read_csv('./assets/dataset.csv')

# Normalize data
performance_dummies = pd.get_dummies(dataset['Performance'], prefix='', prefix_sep='')
tm_dummies = pd.get_dummies(dataset['Tm'], prefix='Tm')
pos_dummies = pd.get_dummies(dataset['Pos'], prefix='Pos')
dataset = pd.concat([dataset, performance_dummies, tm_dummies, pos_dummies], axis=1)

# Store players for later use
players = dataset['Player']

# Drop unnecessary columns for training
dataset.drop(['Performance', 'Tm', 'Pos', 'Player'], axis=1, inplace=True)
dataset.replace({False: 0, True: 1}, inplace=True)
dataset = dataset.applymap(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x)
dataset = dataset.apply(pd.to_numeric, errors='coerce')
dataset = dataset.fillna(0)

# Manual Min-Max Normalization for the dataset
def min_max_normalize(column):
    return (column - column.min()) / (column.max() - column.min())

# Apply normalization to all numeric columns
for column in dataset.columns:
    if dataset[column].dtype != 'object':  # Ensure it's a numeric column
        dataset[column] = min_max_normalize(dataset[column])

# ReLU function
def relu(x):
    return max(0, x)

def relu_derivative(x):
    return 1 if x > 0 else 0

# Loss function
def loss_function(y_true, y_pred):
    return 0.5 * (y_true - y_pred) ** 2

# Initialize weights
def initialize_weights(input_size, hidden_size, output_size):
    limit_input_hidden = math.sqrt(6 / (input_size + hidden_size))
    limit_hidden_output = math.sqrt(6 / (hidden_size + output_size))
    
    weights_input_hidden = [[random.uniform(-limit_input_hidden, limit_input_hidden) for _ in range(hidden_size)] for _ in range(input_size)]
    weights_hidden_output = [[random.uniform(-limit_hidden_output, limit_hidden_output) for _ in range(output_size)] for _ in range(hidden_size)]
    
    return weights_input_hidden, weights_hidden_output

# Backpropagation with ReLU
def backpropagation(X_train, y_train, weights_input_hidden, weights_hidden_output, learning_rate=0.01, epochs=10):
    for epoch in range(epochs):
        total_loss = 0
        for index in range(len(X_train)):
            input_layer = X_train.iloc[index].values
            target = y_train.iloc[index]

            # Forward pass
            hidden_layer = [relu(sum([input_layer[i] * weights_input_hidden[i][j] for i in range(input_size)])) for j in range(hidden_size)]
            output_layer = relu(sum([hidden_layer[j] * weights_hidden_output[j][0] for j in range(hidden_size)]))

            # Loss calculation
            loss = loss_function(target, output_layer)
            total_loss += loss

            # Backward pass
            output_error = target - output_layer
            output_delta = output_error * relu_derivative(output_layer)

            hidden_error = [output_delta * weights_hidden_output[j][0] for j in range(hidden_size)]
            hidden_delta = [hidden_error[j] * relu_derivative(hidden_layer[j]) for j in range(hidden_size)]

            # Update weights
            for j in range(hidden_size):
                weights_hidden_output[j][0] += learning_rate * output_delta * hidden_layer[j]

            for i in range(input_size):
                for j in range(hidden_size):
                    weights_input_hidden[i][j] += learning_rate * hidden_delta[j] * input_layer[i]

        print(f'Epoch: {epoch + 1}/{epochs} | Total Loss: {total_loss}')

# Evaluate the model
def evaluate(X_test, weights_input_hidden, weights_hidden_output):
    predictions = []
    for index in range(len(X_test)):
        input_layer = X_test.iloc[index].values

        # Forward pass
        hidden_layer = [relu(sum([input_layer[i] * weights_input_hidden[i][j] for i in range(input_size)])) for j in range(hidden_size)]
        output_layer = relu(sum([hidden_layer[j] * weights_hidden_output[j][0] for j in range(hidden_size)]))

        prediction = 1 if output_layer > 0.5 else 0
        predictions.append(prediction)

    return predictions

# Shuffle and split the dataset
dataset = dataset.sample(frac=1).reset_index(drop=True)
split_ratio = 0.8
split_index = int(split_ratio * len(dataset))
train_data = dataset.iloc[:split_index]
test_data = dataset.iloc[split_index:]

# Separate features and target
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

# Define layer sizes
input_size = X_train.shape[1]
hidden_size = 20
output_size = 1

# Initialize weights
weights_input_hidden, weights_hidden_output = initialize_weights(input_size, hidden_size, output_size)

# Train the model
backpropagation(X_train, y_train, weights_input_hidden, weights_hidden_output, learning_rate=0.01, epochs=100)

# Evaluate the model
predictions = evaluate(X_test, weights_input_hidden, weights_hidden_output)

# Prepare results
results = pd.DataFrame({'Player': players[split_index:].reset_index(drop=True), 'Predicted Performance': predictions})
results['Predicted Performance'] = results['Predicted Performance'].replace({0: 'Bad', 1: 'Good'})

print(results)
