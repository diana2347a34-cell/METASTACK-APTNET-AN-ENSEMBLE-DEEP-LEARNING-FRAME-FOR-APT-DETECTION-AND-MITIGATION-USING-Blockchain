import numpy as np
import tensorflow as tf
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt

# ==== Simulated Input Data from CSV ====
# Assuming 'data.csv' contains features and 'Activity' column with string labels
data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Diana (1)/part-00000-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv/concatenated_output4.csv')  # Replace with your CSV path
X = data.drop(columns=['label']).values  # Features
y = data['label'].values  # Labels

# ==== Label Encoding to convert string labels to numeric labels ====
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Encoding the string labels

# Split into Train and Test using encoded labels
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# ==== 1. Train Individual Models with Training/Testing Accuracy and Loss Tracking ====

# 1.1 XGBoost Model
def train_xgb(learning_rate=0.2, max_depth=7, n_estimators=100):
    xgb_model = XGBClassifier(learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators, use_label_encoder=False, eval_metric="logloss")
    xgb_model.fit(X_train, y_train)
    train_acc = xgb_model.score(X_train, y_train)
    test_acc = xgb_model.score(X_test, y_test)
    return xgb_model, train_acc, test_acc

# 1.2 Logistic Regression Model
def train_lr(C=0.7):
    lr_model = LogisticRegression(C=C)
    lr_model.fit(X_train, y_train)
    train_acc = lr_model.score(X_train, y_train)
    test_acc = lr_model.score(X_test, y_test)
    return lr_model, train_acc, test_acc

# 1.3 CNN Model
def train_cnn(learning_rate=0.001, dropout_rate=0.3):
    cnn_model = Sequential([
        Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(1, activation='sigmoid')
    ])
    cnn_model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    history = cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, validation_data=(X_test, y_test))
    train_acc = history.history['accuracy']
    test_acc = history.history['val_accuracy']
    train_loss = history.history['loss']
    test_loss = history.history['val_loss']
    return cnn_model, train_acc, test_acc, train_loss, test_loss

# 1.4 LSTM Model
def train_lstm(learning_rate=0.01, dropout_rate=0.4, units=50):
    X_train_lstm = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test_lstm = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    lstm_model = Sequential([
        LSTM(units, input_shape=(1, X_train.shape[1]), return_sequences=False, dropout=dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    lstm_model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    history = lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, verbose=0, validation_data=(X_test_lstm, y_test))
    train_acc = history.history['accuracy']
    test_acc = history.history['val_accuracy']
    train_loss = history.history['loss']
    test_loss = history.history['val_loss']
    return lstm_model, train_acc, test_acc, train_loss, test_loss

# 1.5 Binary Neural Network (BNN) Model
def train_bnn():
    bnn_model = Sequential([
        Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(1, activation='sigmoid')
    ])
    bnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    history = bnn_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, validation_data=(X_test, y_test))
    train_acc = history.history['accuracy']
    test_acc = history.history['val_accuracy']
    train_loss = history.history['loss']
    test_loss = history.history['val_loss']
    return bnn_model, train_acc, test_acc, train_loss, test_loss

# ==== 2. Obtain Predictions ====
def get_predictions():
    # Train models with fixed hyperparameters
    xgb_model, _, _ = train_xgb()
    lr_model, _, _ = train_lr()
    cnn_model, _, _, _, _ = train_cnn()
    lstm_model, _, _, _, _ = train_lstm()
    bnn_model, _, _, _, _ = train_bnn()

    # Get model predictions
    P_xgb = xgb_model.predict_proba(X_test)[:, 1]
    P_lr = lr_model.predict_proba(X_test)[:, 1]
    P_cnn = cnn_model.predict(X_test).flatten()
    P_lstm = lstm_model.predict(X_test.reshape(X_test.shape[0], 1, X_test.shape[1])).flatten()
    P_bnn = bnn_model.predict(X_test).flatten()

    return P_xgb, P_lr, P_cnn, P_lstm, P_bnn

# ==== 3. Weighted Voting (CNN, LSTM, BNN) ====
def weighted_voting(P_cnn, P_lstm, P_bnn, w_cnn=0.33, w_lstm=0.33, w_bnn=0.34):
    P_combined = w_cnn * P_cnn + w_lstm * P_lstm + w_bnn * P_bnn
    return P_combined

# ==== 4. Final Weighted Voting (Combined + XGB + LR) ====
def final_voting(P_combined, P_xgb, P_lr, w_combined=0.4, w_xgb=0.3, w_lr=0.3):
    P_final = w_combined * P_combined + w_xgb * P_xgb + w_lr * P_lr
    predicted_classes = (P_final > 0.5).astype(int)
    return predicted_classes

# ==== Addax Optimization Algorithm (AOA) ====
class AddaxOptimizationAlgorithm:
    def __init__(self, population_size, num_variables, max_iter, lower_bound, upper_bound):
        self.population_size = population_size
        self.num_variables = num_variables
        self.max_iter = max_iter
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.population = np.random.rand(self.population_size, self.num_variables) * (upper_bound - lower_bound) + lower_bound
        self.best_solution = None
        self.best_value = np.inf

    def update_position_for_foraging(self, i, SAi, xi):
        """Phase 1: Foraging Process (Exploration Phase)"""
        r = np.random.rand(self.num_variables)
        I = np.random.choice([1, 2], size=self.num_variables)  # Randomly selecting 1 or 2 for each dimension
        xi_new = xi + r * (SAi - xi) * I
        return xi_new

    def update_position_for_digging(self, i, xi):
        """Phase 2: Digging Process (Exploitation Phase)"""
        gamma = 0.1  # Adjust based on your needs
        delta = 0.1  # Adjust based on your needs
        zeta = 0.5  # Adjust based on your needs
        tau = 0.2  # Adjust based on your needs

        # Calculate the flooding and removal terms
        flooding = zeta * (np.abs(self.upper_bound - self.lower_bound) * np.random.rand(self.num_variables))
        worst_solution = np.max(self.population, axis=0)  # Worst solution is the one with highest objective function value
        removal = -tau * (xi - worst_solution)

        # Update position based on the new equation
        xi_new = xi + gamma * flooding + delta * removal
        # Clip the position to stay within bounds
        xi_new = np.clip(xi_new, self.lower_bound, self.upper_bound)
        return xi_new

    def update_population(self, objective_function):
        """Update population positions and evaluate objective function"""
        for i in range(self.population_size):
            # Foraging (Exploration Phase)
            SAi = self.population[np.random.randint(self.population_size)]
            xi_new_1 = self.update_position_for_foraging(i, SAi, self.population[i])

            # Check if new position from foraging is better
            Fi_new_1 = objective_function(xi_new_1)
            Fi_current = objective_function(self.population[i])

            if Fi_new_1 < Fi_current:
                self.population[i] = xi_new_1

            # Digging (Exploitation Phase)
            xi_new_2 = self.update_position_for_digging(i, self.population[i])

            # Check if new position from digging is better
            Fi_new_2 = objective_function(xi_new_2)
            if Fi_new_2 < Fi_current:
                self.population[i] = xi_new_2

        # Update the best solution found so far
        for i in range(self.population_size):
            value = objective_function(self.population[i])
            if value < self.best_value:
                self.best_solution = self.population[i]
                self.best_value = value

        return self.best_solution, self.best_value


# Example Objective Function to Minimize (Mean Squared Error)
def objective_function(xi):
    # Example: MSE on predictions (this should be customized for your scenario)
    P_xgb, P_lr, P_cnn, P_lstm, P_bnn = get_predictions()
    P_combined = weighted_voting(P_cnn, P_lstm, P_bnn)
    predicted_classes = final_voting(P_combined, P_xgb, P_lr)
    mse = np.mean((predicted_classes - y_test) ** 2)  # Mean Squared Error as objective
    return mse

# Perform the optimization
aoa = AddaxOptimizationAlgorithm(population_size=10, num_variables=3, max_iter=10, lower_bound=0.0, upper_bound=1.0)
best_solution, best_value = aoa.update_population(objective_function)

# Final voting results
P_xgb, P_lr, P_cnn, P_lstm, P_bnn = get_predictions()
P_combined = weighted_voting(P_cnn, P_lstm, P_bnn)
final_predictions = final_voting(P_combined, P_xgb, P_lr)

# Calculate final accuracy
final_accuracy = accuracy_score(y_test, final_predictions)
# ==== Run model training and evaluation ====
xgb_model, train_acc_xgb, test_acc_xgb = train_xgb()
lr_model, train_acc_lr, test_acc_lr = train_lr()
cnn_model, train_acc_cnn, test_acc_cnn, train_loss_cnn, test_loss_cnn = train_cnn()
lstm_model, train_acc_lstm, test_acc_lstm, train_loss_lstm, test_loss_lstm = train_lstm()
bnn_model, train_acc_bnn, test_acc_bnn, train_loss_bnn, test_loss_bnn = train_bnn()

# Show accuracy and loss for all models
print("XGBoost: Train Accuracy: {:.4f}, Test Accuracy: {:.4f}".format(train_acc_xgb, test_acc_xgb))
print("Logistic Regression: Train Accuracy: {:.4f}, Test Accuracy: {:.4f}".format(train_acc_lr, test_acc_lr))
print("CNN: Train Accuracy: {:.4f}, Test Accuracy: {:.4f}".format(train_acc_cnn[-1], test_acc_cnn[-1]))
print("LSTM: Train Accuracy: {:.4f}, Test Accuracy: {:.4f}".format(train_acc_lstm[-1], test_acc_lstm[-1]))
print("BNN: Train Accuracy: {:.4f}, Test Accuracy: {:.4f}".format(train_acc_bnn[-1], test_acc_bnn[-1]))
# Get predictions and combine results
P_xgb, P_lr, P_cnn, P_lstm, P_bnn = get_predictions()
P_combined = weighted_voting(P_cnn, P_lstm, P_bnn)
predicted_classes = final_voting(P_combined, P_xgb, P_lr)
# Track and display best training accuracy
best_train_accuracy = max(train_acc_xgb, train_acc_lr, train_acc_cnn[-1], train_acc_lstm[-1], train_acc_bnn[-1])
print(f"Best Accuracy: {best_train_accuracy:.4f}")