from sklearn import linear_model
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import polars as pl



# random values for testing
X = np.random.rand(100, 1) * 10  # 100 random
y = 2.5 * X + np.random.randn(100, 1) * 2  # Linear relation with some noise


frame = pl.DataFrame({
    "X": X.flatten(),
    "y": y.flatten()
})


linear_model = linear_model.LinearRegression()
linear_model.fit(frame[["X"]], frame["y"])


# plotting the results
plt.scatter(frame["X"], frame["y"], color="blue", label="Data Points")
plt.plot(frame["X"], linear_model.predict(frame[["X"]]), color="red", label="Regression Line")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression Example")
plt.legend()
plt.show()