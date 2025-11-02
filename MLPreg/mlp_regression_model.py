import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

# Load the data
df = pd.read_csv('Advertising Budget and Sales.csv', index_col=0)
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Prepare features and target
X = df[['TV Ad Budget ($)', 'Radio Ad Budget ($)', 'Newspaper Ad Budget ($)']]
y = df['Sales ($)']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train MLPRegressor
mlp = MLPRegressor(
    hidden_layer_sizes=(100, 50),  # Two hidden layers
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size='auto',
    learning_rate='constant',
    learning_rate_init=0.001,
    max_iter=1000,
    random_state=42
)

print("\nTraining MLPRegressor...")
mlp.fit(X_train_scaled, y_train)

# Make predictions
y_train_pred = mlp.predict(X_train_scaled)
y_test_pred = mlp.predict(X_test_scaled)

# Calculate metrics
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"\nModel Performance:")
print(f"Training MSE: {train_mse:.3f}")
print(f"Testing MSE: {test_mse:.3f}")
print(f"Training R²: {train_r2:.3f}")
print(f"Testing R²: {test_r2:.3f}")

# Create comprehensive plots
plt.figure(figsize=(15, 12))

# 1. Actual vs Predicted scatter plot
plt.subplot(2, 3, 1)
plt.scatter(y_test, y_test_pred, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Sales ($)')
plt.ylabel('Predicted Sales ($)')
plt.title('Actual vs Predicted Sales (Test Set)')
plt.grid(True, alpha=0.3)

# 2. Residuals plot
plt.subplot(2, 3, 2)
residuals = y_test - y_test_pred
plt.scatter(y_test_pred, residuals, alpha=0.7, color='green')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Sales ($)')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.grid(True, alpha=0.3)

# 3. Training loss curve
plt.subplot(2, 3, 3)
plt.plot(mlp.loss_curve_)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True, alpha=0.3)

# 4. Feature importance (using permutation-like approach)
plt.subplot(2, 3, 4)
feature_names = ['TV Ad Budget', 'Radio Ad Budget', 'Newspaper Ad Budget']

# Simple feature importance based on correlation with target
correlations = [abs(df['TV Ad Budget ($)'].corr(df['Sales ($)'])),
                abs(df['Radio Ad Budget ($)'].corr(df['Sales ($)'])),
                abs(df['Newspaper Ad Budget ($)'].corr(df['Sales ($)']))]

plt.bar(feature_names, correlations, color=['skyblue', 'lightgreen', 'lightcoral'])
plt.xlabel('Features')
plt.ylabel('Absolute Correlation with Sales')
plt.title('Feature Correlations')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# 5. Distribution of predictions vs actual
plt.subplot(2, 3, 5)
plt.hist(y_test, alpha=0.7, label='Actual', bins=10, color='blue')
plt.hist(y_test_pred, alpha=0.7, label='Predicted', bins=10, color='red')
plt.xlabel('Sales ($)')
plt.ylabel('Frequency')
plt.title('Distribution: Actual vs Predicted')
plt.legend()
plt.grid(True, alpha=0.3)

# 6. Model architecture visualization
plt.subplot(2, 3, 6)
layers = [3] + list(mlp.hidden_layer_sizes) + [1]  # Input + hidden + output
layer_names = ['Input\n(3 features)'] + [f'Hidden {i+1}\n({size} neurons)' for i, size in enumerate(mlp.hidden_layer_sizes)] + ['Output\n(Sales)']

x_positions = range(len(layers))
y_positions = [layer/2 for layer in layers]

for i, (x, y, name) in enumerate(zip(x_positions, y_positions, layer_names)):
    plt.scatter(x, 0, s=layers[i]*50, alpha=0.7, color=plt.cm.viridis(i/len(layers)))
    plt.text(x, -0.5, name, ha='center', va='top', fontsize=8)

plt.xlim(-0.5, len(layers)-0.5)
plt.ylim(-1, 1)
plt.title('MLP Architecture')
plt.axis('off')

plt.tight_layout()
plt.show()

# Additional detailed analysis
print(f"\nDetailed Analysis:")
print(f"Number of iterations: {mlp.n_iter_}")
print(f"Number of layers: {mlp.n_layers_}")
print(f"Number of outputs: {mlp.n_outputs_}")

# Feature statistics
print(f"\nFeature Statistics:")
for i, feature in enumerate(feature_names):
    print(f"{feature}: correlation = {correlations[i]:.3f}")