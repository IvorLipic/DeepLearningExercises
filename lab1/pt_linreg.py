import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


## Definicija računskog grafa
# podaci i parametri, inicijalizacija parametara
a = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

X = torch.tensor([1., 2., 3., 5., 4.])
Y = torch.tensor([3., 5., 7., 8., 6.])

'''
A = (1, 3), B = (2, 5)
----------------------------------
y - 3 = [(5 - 3)/(2 - 1)] * (x - 1)
y - 3 = 2x - 2
----------------
[y = 2x + 1] -> a = 2, b = 1
----------------

GRAD:
E = sumN_i((h(x_i) - y_i))^2 
  = sumN_i((ax_i + b - y_i)^2) 
  = sumN_i((ax_i)^2 + 2ax_ib - 2ax_iy_i - 2by_i + b^2  + y_i^2)

dE/da = sumN_i(2ax_i^2 + 2x_ib - 2x_iy_i) = 2 * sumN_i(ax_i^2 + x_ib - x_iy_i)
dE/db = sumN_i(2ax_i - 2y_i + 2b)         = 2 * sumN_i(ax_i   - y_i  + b)
'''

# optimizacijski postupak: gradijentni spust
optimizer = optim.SGD([a, b], lr=0.01)

for i in range(100):
    # afin regresijski model
    Y_ = a*X + b

    diff = (Y-Y_)

    # kvadratni gubitak
    loss = torch.sum(diff**2)

    # računanje gradijenata
    loss.backward()
    grad_a = 2 * np.sum(a.item()*X.detach().numpy()**2 + X.detach().numpy()*b.item() - X.detach().numpy()*Y.detach().numpy())
    grad_b = 2 * np.sum(a.item()*X.detach().numpy()    - Y.detach().numpy()          + b.item())

    # korak optimizacije
    optimizer.step()
    print(f'''
          step: {i}, loss: {loss.item()}, 
          Y_: {Y_.detach().numpy()}, 
          a: {a.item()}, 
          b: {b.item()}
          grad_a: {a.grad.item()}
          \t- manual: {grad_a}, 
          grad_b: {b.grad.item()}
          \t- manual: {grad_b},''')
    # Postavljanje gradijenata na nulu
    optimizer.zero_grad()

plt.scatter(X.detach().numpy(), Y.detach().numpy(), color='blue')
x_range = torch.linspace(0, torch.max(X), 100)
y_range = a.detach().numpy() * x_range.detach().numpy() + b.detach().numpy()
plt.plot(x_range.detach().numpy(), y_range, color='red')
plt.show()