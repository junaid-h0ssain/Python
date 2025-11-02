import csv
import random
import math

# Simple MLPRegressor implementation without external dependencies
class SimpleMLPRegressor:
    def __init__(self, hidden_sizes=[10], learning_rate=0.01, epochs=1000):
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = []
        self.biases = []
        
    def _sigmoid(self, x):
        return 1 / (1 + math.exp(-max(-500, min(500, x))))
    
    def _sigmoid_derivative(self, x):
        s = self._sigmoid(x)
        return s * (1 - s)
    
    def _initialize_weights(self, input_size):
        # Initialize weights and biases
        layer_sizes = [input_size] + self.hidden_sizes + [1]
        
        for i in range(len(layer_sizes) - 1):
            # Xavier initialization
            limit = math.sqrt(6 / (layer_sizes[i] + layer_sizes[i + 1]))
            weights = [[random.uniform(-limit, limit) for _ in range(layer_sizes[i + 1])] 
                      for _ in range(layer_sizes[i])]
            biases = [random.uniform(-limit, limit) for _ in range(layer_sizes[i + 1])]
            
            self.weights.append(weights)
            self.biases.append(biases)
    
    def _forward(self, X):
        activations = [X[:]]
        
        for layer_idx in range(len(self.weights)):
            layer_input = activations[-1]
            layer_output = []
            
            for neuron_idx in range(len(self.weights[layer_idx][0])):
                weighted_sum = sum(layer_input[i] * self.weights[layer_idx][i][neuron_idx] 
                                 for i in range(len(layer_input)))
                weighted_sum += self.biases[layer_idx][neuron_idx]
                
                if layer_idx < len(self.weights) - 1:  # Hidden layers
                    activation = self._sigmoid(weighted_sum)
                else:  # Output layer (linear)
                    activation = weighted_sum
                
                layer_output.append(activation)
            
            activations.append(layer_output)
        
        return activations
    
    def fit(self, X, y):
        if not self.weights:
            self._initialize_weights(len(X[0]))
        
        for epoch in range(self.epochs):
            total_loss = 0
            
            for i in range(len(X)):
                # Forward pass
                activations = self._forward(X[i])
                prediction = activations[-1][0]
                
                # Calculate loss
                loss = (prediction - y[i]) ** 2
                total_loss += loss
                
                # Backward pass (simplified)
                error = prediction - y[i]
                
                # Update output layer weights
                for j in range(len(self.weights[-1])):
                    for k in range(len(self.weights[-1][j])):
                        gradient = error * activations[-2][j]
                        self.weights[-1][j][k] -= self.learning_rate * gradient
                
                # Update output bias
                self.biases[-1][0] -= self.learning_rate * error
            
            if epoch % 100 == 0:
                avg_loss = total_loss / len(X)
                print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
    
    def predict(self, X):
        predictions = []
        for sample in X:
            activations = self._forward(sample)
            predictions.append(activations[-1][0])
        return predictions

# Load and process data
def load_csv_data(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data

# Load the advertising data
print("Loading advertising data...")
data = load_csv_data('Advertising Budget and Sales.csv')

# Extract features and target
X = []
y = []

for row in data:
    features = [
        float(row['TV Ad Budget ($)']),
        float(row['Radio Ad Budget ($)']),
        float(row['Newspaper Ad Budget ($)'])
    ]
    target = float(row['Sales ($)'])
    
    X.append(features)
    y.append(target)

print(f"Loaded {len(X)} samples with {len(X[0])} features each")

# Normalize features (simple min-max scaling)
def normalize_features(X):
    X_norm = []
    mins = [min(X[i][j] for i in range(len(X))) for j in range(len(X[0]))]
    maxs = [max(X[i][j] for i in range(len(X))) for j in range(len(X[0]))]
    
    for sample in X:
        normalized = [(sample[j] - mins[j]) / (maxs[j] - mins[j]) if maxs[j] != mins[j] else 0 
                     for j in range(len(sample))]
        X_norm.append(normalized)
    
    return X_norm, mins, maxs

X_norm, mins, maxs = normalize_features(X)

# Split data (80% train, 20% test)
split_idx = int(0.8 * len(X_norm))
X_train = X_norm[:split_idx]
X_test = X_norm[split_idx:]
y_train = y[:split_idx]
y_test = y[split_idx:]

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Create and train model
print("\nTraining MLP Regressor...")
model = SimpleMLPRegressor(hidden_sizes=[10, 5], learning_rate=0.01, epochs=500)
model.fit(X_train, y_train)

# Make predictions
print("\nMaking predictions...")
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Calculate metrics
def calculate_mse(actual, predicted):
    return sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual)

def calculate_r2(actual, predicted):
    mean_actual = sum(actual) / len(actual)
    ss_tot = sum((a - mean_actual) ** 2 for a in actual)
    ss_res = sum((a - p) ** 2 for a, p in zip(actual, predicted))
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

train_mse = calculate_mse(y_train, train_predictions)
test_mse = calculate_mse(y_test, test_predictions)
train_r2 = calculate_r2(y_train, train_predictions)
test_r2 = calculate_r2(y_test, test_predictions)

print(f"\nModel Performance:")
print(f"Training MSE: {train_mse:.3f}")
print(f"Testing MSE: {test_mse:.3f}")
print(f"Training R²: {train_r2:.3f}")
print(f"Testing R²: {test_r2:.3f}")

print(f"\nSample Predictions vs Actual (Test Set):")
for i in range(min(5, len(y_test))):
    print(f"Actual: {y_test[i]:.2f}, Predicted: {test_predictions[i]:.2f}")