# Decision Tree Regression Comparison
# This script demonstrates Decision Tree Regression in two contexts:
# 1. Single-output regression to show the effect of max_depth on overfitting.
# 2. Multi-output regression to approximate a circle, again highlighting max_depth.

# Author: Wallace de Holanda Costa
# License: MIT License

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor


class DecisionTreeComparison:
    """
    Class to demonstrate Single-Output and Multi-Output Decision Tree Regression
    and the effect of the 'max_depth' parameter.
    """

    def __init__(self, random_seed=1):
        self.rng = np.random.RandomState(random_seed)
        self.output_colors = {
            2: '#4A90E2',  #Blue
            5: '#F95D6A',  #Red/Pink
            8: '#4BB38E'  #Green / Teal
        }

    def _create_single_output_data(self, n_samples=50):
        """Creates a simple, noisy sine wave dataset for single-output regression."""
        X = np.sort(5 * self.rng.rand(n_samples, 1), axis=0)
        y = np.sin(X).ravel()

        y[::5] += 3 * (0.5 - self.rng.rand(n_samples // 5))
        return X, y

    def run_single_output_demo(self, max_depth_1=2, max_depth_2=5):
        """
        Runs the single-output regression demo, showing overfitting vs. generalization.
        """
        print("--- Running Single-Output Regression Demo ---")
        X, y = self._create_single_output_data()

        # 1. Fit regression models
        regr_1 = DecisionTreeRegressor(max_depth=max_depth_1)
        regr_2 = DecisionTreeRegressor(max_depth=max_depth_2)

        regr_1.fit(X, y)
        regr_2.fit(X, y)

        # 2. Get predictions on the test set (fine grid)
        X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
        y_1 = regr_1.predict(X_test)
        y_2 = regr_2.predict(X_test)

        # 3. Plot the results
        plt.figure(figsize=(8, 6), facecolor='#f8f8f8')

        # Data points
        plt.scatter(X, y, s=40, edgecolor="black", c="#FFD700", label="Training Data", zorder=2)  # Gold data points

        # Predictions
        plt.plot(X_test, y_1, color=self.output_colors[max_depth_1],
                 label=f"max_depth={max_depth_1} (General)", linewidth=3, linestyle='-')
        plt.plot(X_test, y_2, color=self.output_colors[max_depth_2],
                 label=f"max_depth={max_depth_2} (Overfit)", linewidth=3, linestyle='--')

        plt.xlabel("Feature (X)")
        plt.ylabel("Target (y)")
        plt.title("Decision Tree Regression: Overfitting Demonstration", fontsize=15, fontweight='bold')
        plt.legend(loc="upper left")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.show()

    def run_multi_output_demo(self, max_depths=[2, 5, 8]):
        """
        Runs the multi-output regression demo, approximating a circle.
        """
        print("\n--- Running Multi-Output Regression Demo (Circle Approximation) ---")

        # 1. Create the random dataset (Noisy Circle)
        # X: 100 samples in the range [-100, 100]. Y: (sin(X), cos(X))
        X = np.sort(200 * self.rng.rand(100, 1) - 100, axis=0)
        y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
        # Add noise
        y[::5, :] += 0.5 - self.rng.rand(20, 2)

        # 2. Fit regression models for different depths
        regr_models = {}
        for depth in max_depths:
            regr = DecisionTreeRegressor(max_depth=depth)
            regr.fit(X, y)
            regr_models[depth] = regr

        # 3. Predict on the test set
        X_test = np.arange(-100.0, 100.0, 0.01)[:, np.newaxis]
        y_preds = {depth: model.predict(X_test) for depth, model in regr_models.items()}

        # 4. Plot the results
        s = 35  # Marker size
        plt.figure(figsize=(8, 8), facecolor='#f8f8f8')

        # Original Data (Circle with Noise)
        plt.scatter(y[:, 0], y[:, 1], c="gold", s=s, edgecolor="black", alpha=0.8, label="Training Data")

        # Predictions (Discrete segments approximating the curve)
        for depth, y_pred in y_preds.items():
            plt.scatter(
                y_pred[:, 0],
                y_pred[:, 1],
                c=self.output_colors[depth],
                s=s,
                edgecolor="black",
                alpha=0.7,
                label=f"max_depth={depth}",
            )

        plt.xlim([-4.5, 4.5])
        plt.ylim([-4.5, 4.5])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel("Target 1 ($\pi \cdot \sin(X)$)")
        plt.ylabel("Target 2 ($\pi \cdot \cos(X)$)")
        plt.title("Multi-Output Decision Tree Regression: Circle Approximation", fontsize=15, fontweight='bold')
        plt.legend(loc="best")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.show()


# --- Execution Block ---
if __name__ == "__main__":
    demo = DecisionTreeComparison()

    # First demonstration
    demo.run_single_output_demo()

    # Second demonstration
    demo.run_multi_output_demo()
