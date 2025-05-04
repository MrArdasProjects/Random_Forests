import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings
from mpl_toolkits.mplot3d import Axes3D

# (5 pts) Build X and y
df = pd.read_csv("Football_players.csv", encoding="ISO-8859-1")
X = df[["Age", "Height", "Mental", "Skill"]].to_numpy()
y = df["Salary"].to_numpy()

# (10 pts Bonus) Handle warnings
warnings.filterwarnings("error", category=UserWarning)

results = []

# (5 pts) Loop over n (trees) and d (depth)
for n in range(10, 101):
    for d in range(3, 13):
        try:
            rf = RandomForestRegressor(
                n_estimators=n,
                max_depth=d,
                oob_score=True,
                bootstrap=True,
                random_state=42,
                n_jobs=-1
            )
            # (5 pts) Fit model
            rf.fit(X, y)
            # (20 pts) Evaluate using OOB
            oob_pred = rf.oob_prediction_
            mse = mean_squared_error(y, oob_pred)
            results.append((n, d, mse))
            # (5 pts Bonus) Observe runtime results
            print(f"n = {n}, d = {d}, MSE = {mse:.2f}")
        except Exception:
            continue


results_df = pd.DataFrame(results, columns=["n_estimators", "max_depth", "mse"])

# (5 pts Bonus) Find best result for plotting
best_row = results_df.loc[results_df["mse"].idxmin()]


fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')

# (25 pts) Plot all results / (5 pts Bonus) Highlight best result / (5 pts) Set labels and title
ax.scatter(results_df["n_estimators"], results_df["max_depth"], results_df["mse"], c='red', label="All Results")
ax.scatter(best_row["n_estimators"], best_row["max_depth"], best_row["mse"], c='blue', s=60, label="Best (min MSE)")
ax.set_xlabel("Number of Trees")
ax.set_ylabel("Depth")
ax.set_zlabel("MSE")
ax.set_title("Random Forest - Parameter Optimization")
ax.legend()
plt.tight_layout()
plt.show()

# (25 pts) Depth analysis
depth_means = results_df.groupby("max_depth")["mse"].mean().reset_index()
for d in range(3, 13):
    avg_mse = depth_means.loc[depth_means["max_depth"] == d, "mse"].values[0]
    print(f"Average MSE for d = {d}: {avg_mse:.2f}")

