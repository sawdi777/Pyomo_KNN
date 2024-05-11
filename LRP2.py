from sklearn.datasets import make_classification
import numpy as np
from pyomo.environ import *
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score

X, y = make_classification(n_classes=2, n_samples=400, n_features=2,
                           n_informative=2, n_redundant=0, n_repeated=0,
                           shuffle=False, weights=[0.55, 0.5], random_state=30)

y = np.where(y==0, -1, y)

model = ConcreteModel()
model.dual = Suffix(direction=Suffix.IMPORT)
model.a = Var(range(len(X)), domain=NonNegativeReals)
model.b = Var()

C = 1.0
model.constraints = ConstraintList()
for i in range(len(X)):
    model.constraints.add(expr=y[i]*(sum(model.a[j]*y[j]*np.dot(X[j], X[i]) for j in range(len(X))) - model.b) >= 1 - model.a[i]/C)

solver = SolverFactory('ipopt')
solver.solve(model)

SV = []
for i in range(len(X)):
    if model.a[i].value > 1e-5:
        SV.append(i)

w = sum(model.a[i].value*y[i]*X[i] for i in SV)
b = model.b.value

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = np.dot(xy, w) - b
Z = Z.reshape(XX.shape)
ax.contour(XX, YY, Z, colors='black', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
plt.show()

y_pred = np.sign(np.dot(X, w) - b)

precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
accuracy = accuracy_score(y, y_pred)

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'Accuracy: {accuracy}')

w1, w2 = w
w0 = b
print(f'w0: {w0}')
print(f'w1: {w1}')
print(f'w2: {w2}')