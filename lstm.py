   import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Embedding
import numpy as np
import pickle
import re

# Open the file with proper syntax (missing comma fixed)
file = open('Sherlock Holmes.txt', 'r', encoding='utf8')

lines = []
for i in file:
    lines.append(i)

# Join all lines into a single string separated by space
data = ' '.join(lines)

print(data[1:1000])  # Print characters 1 to 1000 from the text


data = data.replace('\n', '').replace('\r', '').replace('\ufeff', '').replace('"','')

data = data.split()
data = ''.join(data)


# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])


# Save tokenizer
pickle.dump(tokenizer, open('token.pkl', 'wb'))

# Convert text to sequences
sequence_data = tokenizer.texts_to_sequences([data])[0]
print("Sequence data:", sequence_data[:10])
print("Length of sequence data:", len(sequence_data))

vocab_size = len(tokenizer.word_index) + 1
print("Vocabulary size:", vocab_size)
# Create sequences for training
sequences = []
for i in range(3, len(sequence_data)):
    words = sequence_data[i-3:i+1]
    sequences.append(words)
    
print("Number of sequences:", len(sequences))

# Create X and y
X = []
y = []
for i in sequences:
    X.append(i[0:3])
    y.append(i[3])
 
X = np.array(X)
y = np.array(y)
y_categorical = to_categorical(y, num_classes=vocab_size)
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"y_categorical shape: {y_categorical.shape}")

# Create Keras Embedding layer
embedding_layer = Embedding(input_dim=vocab_size, output_dim=100, input_length=3)
# Build the embedding layer
embedding_layer.build(input_shape=(None, 3))
print("Keras Embedding layer created!")

# Manual LSTM Implementation with Gates (FIXED)
class ManualLSTM:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        
        # Weights for input gate
        self.W_xi = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hi = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_i = np.zeros((hidden_size, 1))
        
        # Weights for forget gate  
        self.W_xf = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hf = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_f = np.zeros((hidden_size, 1))
        
        # Weights for output gate
        self.W_xo = np.random.randn(hidden_size, input_size) * 0.01
        self.W_ho = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_o = np.zeros((hidden_size, 1))
        
        # Weights for cell state
        self.W_xc = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hc = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_c = np.zeros((hidden_size, 1))
        
        # Initialize gradients
        self.reset_gradients()
    
    def reset_gradients(self):
        self.dW_xi = np.zeros_like(self.W_xi)
        self.dW_hi = np.zeros_like(self.W_hi)
        self.db_i = np.zeros_like(self.b_i)
        self.dW_xf = np.zeros_like(self.W_xf)
        self.dW_hf = np.zeros_like(self.W_hf)
        self.db_f = np.zeros_like(self.b_f)
        self.dW_xo = np.zeros_like(self.W_xo)
        self.dW_ho = np.zeros_like(self.W_ho)
        self.db_o = np.zeros_like(self.b_o)
        self.dW_xc = np.zeros_like(self.W_xc)
        self.dW_hc = np.zeros_like(self.W_hc)
        self.db_c = np.zeros_like(self.b_c)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -50, 50)))
    
    def tanh(self, x):
        return np.tanh(x)
    
    def forward(self, x, h_prev, c_prev):
        """
        x shape: (batch_size, input_size)
        h_prev shape: (batch_size, hidden_size) 
        c_prev shape: (batch_size, hidden_size)
        """
        # Input gate
        i = self.sigmoid(np.dot(x, self.W_xi.T) + np.dot(h_prev, self.W_hi.T) + self.b_i.T)
        
        # Forget gate
        f = self.sigmoid(np.dot(x, self.W_xf.T) + np.dot(h_prev, self.W_hf.T) + self.b_f.T)
        
        # Output gate
        o = self.sigmoid(np.dot(x, self.W_xo.T) + np.dot(h_prev, self.W_ho.T) + self.b_o.T)
        
        # Cell candidate
        c_candidate = self.tanh(np.dot(x, self.W_xc.T) + np.dot(h_prev, self.W_hc.T) + self.b_c.T)
        
        # Update cell state
        c = f * c_prev + i * c_candidate
        
        # Update hidden state
        h = o * self.tanh(c)
        
        # Cache for backward pass
        self.cache = (x, h_prev, c_prev, i, f, o, c_candidate, c, h)
        
        return h, c
    
    def backward(self, dh, dc):
        x, h_prev, c_prev, i, f, o, c_candidate, c, h = self.cache
        batch_size = x.shape[0]
        
        # Gradients from next layer
        do = dh * self.tanh(c)
        d_tanh_c = dh * o * (1 - self.tanh(c)**2)
        dc = dc + d_tanh_c
        
        # Gate gradients
        df = dc * c_prev
        di = dc * c_candidate
        dc_candidate = dc * i
        
        # Gate derivatives
        di_input = di * i * (1 - i)
        df_input = df * f * (1 - f)
        do_input = do * o * (1 - o)
        dc_candidate_input = dc_candidate * (1 - c_candidate**2)
        
        # Weight gradients
        self.dW_xi += np.dot(di_input.T, x)
        self.dW_hi += np.dot(di_input.T, h_prev)
        self.db_i += np.sum(di_input, axis=0, keepdims=True).T
        
        self.dW_xf += np.dot(df_input.T, x)
        self.dW_hf += np.dot(df_input.T, h_prev)
        self.db_f += np.sum(df_input, axis=0, keepdims=True).T
        
        self.dW_xo += np.dot(do_input.T, x)
        self.dW_ho += np.dot(do_input.T, h_prev)
        self.db_o += np.sum(do_input, axis=0, keepdims=True).T
        
        self.dW_xc += np.dot(dc_candidate_input.T, x)
        self.dW_hc += np.dot(dc_candidate_input.T, h_prev)
        self.db_c += np.sum(dc_candidate_input, axis=0, keepdims=True).T
        
        # Input gradients - RETURNS 3 VALUES!
        dx = (np.dot(di_input, self.W_xi) + 
              np.dot(df_input, self.W_xf) + 
              np.dot(do_input, self.W_xo) + 
              np.dot(dc_candidate_input, self.W_xc))
        
        dh_prev = (np.dot(di_input, self.W_hi) + 
                   np.dot(df_input, self.W_hf) + 
                   np.dot(do_input, self.W_ho) + 
                   np.dot(dc_candidate_input, self.W_hc))
        
        dc_prev = dc * f
        
        return dx, dh_prev, dc_prev
    
    def update(self, learning_rate):
        self.W_xi -= learning_rate * self.dW_xi
        self.W_hi -= learning_rate * self.dW_hi
        self.b_i -= learning_rate * self.db_i
        
        self.W_xf -= learning_rate * self.dW_xf
        self.W_hf -= learning_rate * self.dW_hf
        self.b_f -= learning_rate * self.db_f
        
        self.W_xo -= learning_rate * self.dW_xo
        self.W_ho -= learning_rate * self.dW_ho  # Fixed typo
        self.b_o -= learning_rate * self.db_o
        
        self.W_xc -= learning_rate * self.dW_xc
        self.W_hc -= learning_rate * self.dW_hc
        self.b_c -= learning_rate * self.db_c
        
        self.reset_gradients()

        self.bias = npC.zeros((output_size, 1))

# Manual Dense Layer
class ManualDense:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) * 0.01
        self.bias = npC.zeros((output_size, 1))
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)
    
    def forward(self, x):
        self.input = x.T if x.ndim > 1 else x.reshape(-1, 1)
        return np.dot(self.weights, self.input) + self.bias
    
    def backward(self, dout):
        batch_size = dout.shape[1]
        self.grad_weights += np.dot(dout, self.input.T) / batch_size
        self.grad_bias += np.sum(dout, axis=1, keepdims=True) / batch_size
        return np.dot(self.weights.T, dout)
    
    def update(self, learning_rate):
        self.weights -= learning_rate * self.grad_weights
        self.bias -= learning_rate * self.grad_bias
        self.grad_weights.fill(0)
        self.grad_bias.fill(0)

# Fixed Manual Adam Optimizer
class ManualAdam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = {}
        self.v = {}
        self._initialized = False
    
    def initialize_moments(self, layers):
        for name, layer in layers.items():
            # Handle LSTM parameters
            if hasattr(layer, 'W_xi'):
                lstm_params = ['W_xi', 'W_hi', 'b_i', 'W_xf', 'W_hf', 'b_f', 
                              'W_xo', 'W_ho', 'b_o', 'W_xc', 'W_hc', 'b_c']
                for param in lstm_params:
                    if hasattr(layer, param):
                        param_value = getattr(layer, param)
                        key = f"{name}_{param}"
                        self.m[key] = np.zeros_like(param_value)
                        self.v[key] = np.zeros_like(param_value)
            
            # Handle Dense parameters
            if hasattr(layer, 'weights'):
                key_weights = f"{name}_weights"
                self.m[key_weights] = np.zeros_like(layer.weights)
                self.v[key_weights] = np.zeros_like(layer.weights)
                
                if hasattr(layer, 'bias'):
                    key_bias = f"{name}_bias"
                    self.m[key_bias] = np.zeros_like(layer.bias)
                    self.v[key_bias] = np.zeros_like(layer.bias)
        
        self._initialized = True
    
    def update(self, layers):
        if not self._initialized:
            self.initialize_moments(layers)
            
        self.t += 1
        
        for name, layer in layers.items():
            # Update LSTM parameters
            if hasattr(layer, 'W_xi'):
                lstm_params = ['W_xi', 'W_hi', 'b_i', 'W_xf', 'W_hf', 'b_f', 
                              'W_xo', 'W_ho', 'b_o', 'W_xc', 'W_hc', 'b_c']
                for param in lstm_params:
                    if hasattr(layer, f'd{param}'):
                        grad = getattr(layer, f'd{param}')
                        key = f"{name}_{param}"
                        self._update_param(layer, param, grad, key)
            
            # Update Dense parameters
            if hasattr(layer, 'grad_weights'):
                key_weights = f"{name}_weights"
                self._update_param(layer, 'weights', layer.grad_weights, key_weights)
                
                if hasattr(layer, 'grad_bias'):
                    key_bias = f"{name}_bias"
                    self._update_param(layer, 'bias', layer.grad_bias, key_bias)
    
    def _update_param(self, layer, param_name, grad, key):
        # Initialize moments if they don't exist
        if key not in self.m:
            param_value = getattr(layer, param_name)
            self.m[key] = np.zeros_like(param_value)
            self.v[key] = np.zeros_like(param_value)
        
        # Update moments
        self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
        self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)
        
        # Bias correction
        m_hat = self.m[key] / (1 - self.beta1 ** self.t)
        v_hat = self.v[key] / (1 - self.beta2 ** self.t)
        
        # Update parameter
        current_value = getattr(layer, param_name)
        updated_value = current_value - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        setattr(layer, param_name, updated_value)

# Complete Model using Keras Embedding + Manual LSTM (FIXED backward)
class HybridLSTMModel:
    def __init__(self, vocab_size, hidden_size):
        # Use Keras Embedding layer
        self.embedding = Embedding(input_dim=vocab_size, output_dim=100)
        self.embedding.build(input_shape=(None, 3))
        
        # Manual LSTM layers
        self.lstm1 = ManualLSTM(100, hidden_size)
        self.lstm2 = ManualLSTM(hidden_size, hidden_size)
        self.dense = ManualDense(hidden_size, vocab_size)
        self.hidden_size = hidden_size
        
        self.layers = {
            'lstm1': self.lstm1,
            'lstm2': self.lstm2, 
            'dense': self.dense
        }
    
    def forward(self, x_sequence):
        batch_size, seq_length = x_sequence.shape
        
        # Use Keras Embedding for forward pass
        embedded = self.embedding(x_sequence).numpy()
        
        # Initialize hidden states
        h1 = np.zeros((batch_size, self.hidden_size))
        c1 = np.zeros((batch_size, self.hidden_size))
        h2 = np.zeros((batch_size, self.hidden_size))
        c2 = np.zeros((batch_size, self.hidden_size))
        
        # Process sequence through manual LSTM layers
        for t in range(seq_length):
            x_t = embedded[:, t, :]
            h1, c1 = self.lstm1.forward(x_t, h1, c1)
            h2, c2 = self.lstm2.forward(h1, h2, c2)
        
        # Final output through manual dense layer
        output = self.dense.forward(h2)
        self.last_hidden = (h1, c1, h2, c2)
        
        return output.T  # (batch_size, vocab_size)
    
    def backward(self, dout):
        # Backward through manual dense layer
        d_dense = self.dense.backward(dout.T)
        
        # Initialize gradients for LSTM backward pass
        dh2 = d_dense.T
        dc2 = np.zeros_like(dh2)
        
        # Backward through manual LSTM layers - FIXED: unpack 3 values
        dx_lstm2, dh1, dc1 = self.lstm2.backward(dh2, dc2)
        dx_embedded, dh_prev, dc_prev = self.lstm1.backward(dh1, dc1)
        
        # Note: We don't backprop through Keras embedding for manual training

# Training function with proper parameter naming
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def categorical_crossentropy(y_pred, y_true):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))

def compute_accuracy(y_pred, y_true):
    predictions = np.argmax(y_pred, axis=1)
    targets = np.argmax(y_true, axis=1)
    return np.mean(predictions == targets)

def train_hybrid_model(model, X, y_onehot, epochs=20, batch_size=32, learning_rate=0.001):
    optimizer = ManualAdam(learning_rate=learning_rate)
    optimizer.initialize_moments(model.layers)
    
    num_samples = len(X)
    num_batches = num_samples // batch_size
    losses = []
    accuracies = []
    
    for epoch in range(epochs):
        total_loss = 0
        total_accuracy = 0
        
        # Shuffle data
        permutation = np.random.permutation(num_samples)
        X_shuffled = X[permutation]
        y_shuffled = y_onehot[permutation]  # Use y_onehot
        
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = start_idx + batch_size
            
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]  # This is one-hot encoded
            
            # Forward pass
            outputs = model.forward(X_batch)
            probs = softmax(outputs)
            
            # Calculate loss and accuracy
            batch_loss = categorical_crossentropy(probs, y_batch)
            batch_accuracy = compute_accuracy(probs, y_batch)
            
            # Backward pass
            d_final_out = (probs - y_batch) / batch_size
            model.backward(d_final_out)
            
            # Update weights with Adam
            optimizer.update(model.layers)
            
            total_loss += batch_loss
            total_accuracy += batch_accuracy
            
            if batch % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch}, Loss: {batch_loss:.4f}, Accuracy: {batch_accuracy:.4f}")
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        losses.append(avg_loss)
        accuracies.append(avg_accuracy)
        
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.4f}")
    
    return losses, accuracies

# Save the trained model
def save_model(model, tokenizer, filename):
    model_data = {
        'embedding_weights': model.embedding.weights,
        'lstm1_weights': {
            'W_xi': model.lstm1.W_xi, 'W_hi': model.lstm1.W_hi, 'b_i': model.lstm1.b_i,
            'W_xf': model.lstm1.W_xf, 'W_hf': model.lstm1.W_hf, 'b_f': model.lstm1.b_f,
            'W_xo': model.lstm1.W_xo, 'W_ho': model.lstm1.W_ho, 'b_o': model.lstm1.b_o,
            'W_xc': model.lstm1.W_xc, 'W_hc': model.lstm1.W_hc, 'b_c': model.lstm1.b_c,
        },
        'lstm2_weights': {
            'W_xi': model.lstm2.W_xi, 'W_hi': model.lstm2.W_hi, 'b_i': model.lstm2.b_i,
            'W_xf': model.lstm2.W_xf, 'W_hf': model.lstm2.W_hf, 'b_f': model.lstm2.b_f,
            'W_xo': model.lstm2.W_xo, 'W_ho': model.lstm2.W_ho, 'b_o': model.lstm2.b_o,
            'W_xc': model.lstm2.W_xc, 'W_hc': model.lstm2.W_hc, 'b_c': model.lstm2.b_c,
        },
        'dense_weights': model.dense.weights,
        'dense_bias': model.dense.bias,
        'tokenizer': tokenizer,
        'vocab_size': vocab_size
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to {filename}")

save_model(model, tokenizer, 'sherlock_lstm_numpy.pkl')
print("Model saved successfully!") 
import matplotlib.pyplot as plt

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(losses) + 1), losses, 'b-', linewidth=2, marker='o')
plt.title('LSTM Training Loss (NumPy Implementation)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.xticks(range(1, len(losses) + 1))
plt.show()Training Summary:
Initial Loss: 9.8465
Final Loss: 8.3040
Total Improvement: 1.5426
Percentage Improvement: 15.67%

print(f"Training Summary:")
print(f"Initial Loss: {losses[0]:.4f}")
print(f"Final Loss: {losses[-1]:.4f}")
print(f"Total Improvement: {losses[0] - losses[-1]:.4f}")
print(f"Percentage Improvement: {((losses[0] - losses[-1]) / losses[0]) * 100:.2f}%")
