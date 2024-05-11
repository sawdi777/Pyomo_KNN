import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from pyomo.environ import *


x, y = make_classification(n_classes=2, n_samples=400, n_features=2, n_informative=2, n_redundant=0,
                           n_repeated=0, shuffle=False, weights=[0.55, 0.5], random_state=30)


y = 2*y - 1


model = ConcreteModel()
model.dual = Suffix(direction=Suffix.IMPORT)


N = len(y)
model.alpha = Var(range(N), domain=NonNegativeReals)


model.y = Param(range(N), initialize=lambda model, i: y[i])
model.x = Param(range(N), range(2), initialize=lambda model, i, j: x[i][j])
model.C = Param(initialize=1)


model.objective = Objective(expr=sum(model.alpha[i] for i in range(N)) - 0.5*sum(model.y[i]*model.y[j]*model.alpha[i]*model.alpha[j]*(np.dot(x[i], x[j])) for i in range(N) for j in range(N)), sense=maximize)


model.constraints = ConstraintList()
for i in range(N):
    model.constraints.add(expr=model.alpha[i] <= model.C)
model.constraints.add(expr=sum(model.alpha[i]*model.y[i] for i in range(N)) == 0)


solver = SolverFactory('ipopt')
solver.solve(model, tee=True)


w = np.zeros(2)
for i in range(N):
    w += model.alpha[i].value * model.y[i] * x[i]
b = np.mean([y[i] - np.dot(w, x[i]) for i in range(N) if 0 < model.alpha[i].value < model.C])


print('w0:', b)
print('w1:', w[0])
print('w2:', w[1])


plt.scatter(x[:, 0], x[:, 1], c=y)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = np.dot(xy, w) + b
Z = Z.reshape(XX.shape)
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
plt.show()
