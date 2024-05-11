from pyomo.environ import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

X, y = make_classification(n_classes=2, n_samples=400, n_features=2,
                           n_informative=2, n_redundant=0, n_repeated=0,
                           shuffle=False, weights=[0.55, 0.5], random_state=30)

X_data = np.array(X)
y_data = np.array(y)

X_data = np.array(X)
y_data = np.array(y)
lambda_val = 1


model = ConcreteModel()
model.d = Param(initialize=X_data.shape[1])
model.lambda_val = Param(initialize=lambda_val)

model.w = Var(range(model.d.value), within=Reals)
model.w0 = Var(within=Reals)

def objective_rule(m):
    return sum((1/(1 + exp(-m.w0 - sum(m.w[i]*X_data[j, i] for i in range(int(m.d.value))))) - y_data[j])**2 for j in range(len(y_data))) + m.lambda_val*sum(m.w[i]**2 for i in range(int(m.d.value)))

model.objective = Objective(rule=objective_rule, sense=minimize)

solver = SolverFactory('ipopt')
solver.solve(model)

print("Optimal w0: ", model.w0.value)
print("Optimal w1: ", model.w[1].value)
#print("Optimal w2: ", model.w[2].value)

plt.figure(figsize=(10, 6))
plt.scatter(X[:,0],X[:,1], c=y)
plt.plot(X_data, 1/(1 + np.exp(-model.w0.value - sum(model.w[i].value*X_data for i in range(int(model.d.value))))), 'b:')
plt.show()

from sklearn.metrics import accuracy_score, precision_score, recall_score
probabilities = np.array([1 / (1 + np.exp(-model.w0.value - sum(model.w[i].value*X_data[j, i] for i in range(int(model.d.value)))))
                          for j in range(len(y_data))])

y_pred = np.array([1 if prob > 0.5 else 0 for prob in probabilities])

accuracy = accuracy_score(y_data, y_pred)
precision = precision_score(y_data, y_pred)
recall = recall_score(y_data, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")