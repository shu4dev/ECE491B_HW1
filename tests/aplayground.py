import torch

# Example: A 3D tensor of shape (batch_size, m, n)
# and a 2D matrix of shape (n, k)
batch_size, m, n, k = 3, 4, 5, 2

# Create a random 3D tensor
A = torch.randn(batch_size, m, n)  # Shape: (3, 4, 5)

# Create a random 2D matrix
B = torch.randn(n, k)              # Shape: (5, 2)

# Multiply A and B using torch.matmul.
# This will perform matrix multiplication on the last dimension of A and the first dimension of B,
# resulting in a tensor of shape (batch_size, m, k)
result = torch.matmul(A, B)

print("Shape of A:", A.shape)         # (3, 4, 5)
print("Shape of B:", B.shape)         # (5, 2)
print("Shape of result:", result.shape)  # (3, 4, 2)
print("Result:\n", result)