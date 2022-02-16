from autograd.tensor import Tensor


x = Tensor.eye(3)
y = Tensor([[2.0, 1.0, 5.1]], requires_grad=True)
z = y.dot(x)

print(f'dot product between: {y.shape} x {x.shape}')
print(f'resulting shape: {z.shape}')

print(f'z is: {z.data}')
z = z.sum()
z.backward()

print(f'performed backwards pass on graph')
print(f'dz/dx = {x.grad}')
print(f'dz/dy = {y.grad}')

