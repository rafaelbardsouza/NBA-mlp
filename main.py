import pandas as pd
import random
import math

dataset = pd.read_csv('./assets/dataset.csv')

performance_dummies = pd.get_dummies(dataset['Performance'], prefix='', prefix_sep='')
""" tm_dummies = pd.get_dummies(dataset['Tm'], prefix='Tm')
pos_dummies = pd.get_dummies(dataset['Pos'], prefix='Pos')

dataset = pd.concat([dataset, performance_dummies, tm_dummies, pos_dummies], axis=1) """

dataset.drop(['Player', 'Performance', 'Tm', 'Pos'], axis=1, inplace=True)

# Função de ativação sigmoide
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Derivada da função sigmoide
def sigmoid_derivative(x):
    return x * (1 - x)

# Inicialização dos pesos e biases
def initialize_parameters(layer_dimensions):
    parameters = {}
    L = len(layer_dimensions)

    for l in range(1, L):
        parameters['W' + str(l)] = [[random.uniform(-1, 1) for _ in range(layer_dimensions[l - 1])] for _ in range(layer_dimensions[l])]
        parameters['b' + str(l)] = [0.0 for _ in range(layer_dimensions[l])]

    return parameters

# Propagação para frente
def forward_propagation(X, parameters):
    cache = {'A0': X}
    L = len(parameters) // 2

    for l in range(1, L + 1):
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        A_prev = cache['A' + str(l - 1)]

        Z = []
        for i in range(len(W)):
            Z.append(sum([W[i][j] * A_prev[j] for j in range(len(A_prev))]) + b[i])

        A = [sigmoid(z) for z in Z]

        cache['Z' + str(l)] = Z
        cache['A' + str(l)] = A

    return A, cache

# Função de custo (erro quadrático médio)
def compute_cost(A, Y):
    m = len(Y)
    cost = sum((A[i] - Y[i]) ** 2 for i in range(m)) / m
    return cost

# Propagação para trás
def backward_propagation(parameters, cache, X, Y):
    grads = {}
    L = len(parameters) // 2
    m = len(Y)

    A_final = cache['A' + str(L)]
    dA = [(2 * (A_final[i] - Y[i])) for i in range(len(A_final))]

    for l in reversed(range(1, L + 1)):
        dZ = [dA[i] * sigmoid_derivative(cache['A' + str(l)][i]) for i in range(len(dA))]
        dW = [[dZ[i] * cache['A' + str(l-1)][j] for j in range(len(cache['A' + str(l-1)]))] for i in range(len(dZ))]
        db = [sum(dZ) for _ in range(len(dZ))]

        grads['dW' + str(l)] = dW
        grads['db' + str(l)] = db

        if l > 1:
            dA = [sum(parameters['W' + str(l)][i][j] * dZ[i] for i in range(len(dZ))) for j in range(len(parameters['W' + str(l)][0]))]

    return grads

# Atualização dos pesos e biases
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    
    for l in range(1, L + 1):
        for i in range(len(parameters['W' + str(l)])):
            for j in range(len(parameters['W' + str(l)][0])):
                parameters['W' + str(l)][i][j] -= learning_rate * grads['dW' + str(l)][i][j]
            parameters['b' + str(l)][i] -= learning_rate * grads['db' + str(l)][i]
    
    return parameters

# Função de treinamento
def train(X, Y, layer_sizes, learning_rate=0.01, num_iterations=10000):
    parameters = initialize_parameters(layer_sizes)
    
    for i in range(num_iterations):
        A, cache = forward_propagation(X, parameters)
        cost = compute_cost(A, Y)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, learning_rate)
        
        if i % 1000 == 0:
            print(f"Iteration {i}, Cost: {cost}")
    
    return parameters

# Exemplo de uso
if __name__ == "__main__":
    # Dados de exemplo
    X = [[random.random() for _ in range(3)]]  # 1 exemplo com 3 features
    Y = [random.random()]  # 1 saída
    
    # Arquitetura da rede: 3 neurônios na entrada, 4 na camada oculta, 1 na saída
    layer_sizes = [3, 4, 1]
    
    # Treinamento da rede
    parameters = train(X[0], Y, layer_sizes, learning_rate=0.01, num_iterations=10000)
    
    # Predição
    A, _ = forward_propagation(X[0], parameters)
    print("Predictions:", A)