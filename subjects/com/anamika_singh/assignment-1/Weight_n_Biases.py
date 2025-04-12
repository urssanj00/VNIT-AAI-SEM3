import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

x = np.linspace(-10, 10, 500)

# Weight variations
plt.figure(figsize=(12, 6))
for w in [0.5, 1, 2, 5]:
    plt.plot(x, sigmoid(w*x), label=f'w={w}, b=0')
plt.title('Sigmoid Response to Weight Changes (b=0)')
plt.xlabel('x')
plt.ylabel('σ(ωx + b)')
plt.legend()
plt.grid(True)
plt.savefig('response_to_weight_changes.png')
plt.close()

# Bias variations
plt.figure(figsize=(12, 6))
for b in [-2, 0, 2, 5]:
    plt.plot(x, sigmoid(x + b), label=f'ω=1, b={b}')
plt.title('Sigmoid Response to Bias Changes (ω=1)')
plt.xlabel('x')
plt.ylabel('σ(ωx + b)')
plt.legend()
plt.grid(True)
plt.savefig('response_to_bias_changes.png')
plt.close()

