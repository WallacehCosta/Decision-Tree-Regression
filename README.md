# Decision Tree Regression Comparison: Overfitting and Multi-Output

This repository contains a concise, object-oriented Python demonstration of **Decision Tree Regression**. It highlights the crucial role of the `max_depth` hyperparameter in two practical scenarios:

1.  **Overfitting vs. Generalization** (Single-Output Regression).
2.  **Approximation of a Complex Curve** (Multi-Output Regression).

---

## What Does This Project Do?

The script `decision_tree_regression_demo.py` is structured into a class, `DecisionTreeComparison`, which executes two distinct machine learning demonstrations:

### 1. Single-Output Demo: Overfitting
This section generates a noisy **sine wave** dataset. It fits two models:
* A **shallow tree (`max_depth=2`)** that captures the general trend (Generalization).
* A **deep tree (`max_depth=5`)** that fits the noise in the training data too closely (Overfitting).
The resulting plot clearly shows the trade-off between bias and variance.

### 2. Multi-Output Demo: Circle Approximation
This section demonstrates the Decision Tree's ability to handle **multiple target variables simultaneously**. It generates a dataset where the outputs form a noisy circle defined by sine and cosine functions.
* The Decision Tree learns to approximate the curved relationship by creating a series of **local linear segments**, with deeper trees resulting in more segments and a smoother (but potentially overfit) approximation.
---

## Key Concepts Demonstrated

| Concept | Description |
| :--- | :--- |
| **Max Depth** | The primary control parameter determining tree complexity and model capacity. |
| **Overfitting** | Observed when a high `max_depth` model fits the noise in the training data instead of the underlying trend. |
| **Piecewise Constant** | The Decision Tree's fundamental output; it makes predictions based on the average value of data points in the terminal leaf nodes, resulting in "steps" or linear segments in the prediction curve. |
| **Multi-Output** | The ability of a single Decision Tree model to predict an output vector (e.g., coordinates $y_1$ and $y_2$) from a single input feature ($X$). |

---

## Getting Started

Follow these steps to set up the environment and run the demo.

### Prerequisites

* Python 3.8+
* `pip` for package installation

### Installation

1.  **Clone the repository** (or download the files).
    ```bash
    git clone [https://github.com/WallacehCosta/Decision-Tree-Regression.git](https://github.com/WallacehCosta/Decision-Tree-Regression.git)
    cd Decision-Tree-Regression
    ```

2.  **Install dependencies** using `pip  install`.
    ```bash
    pip install numpy matplotlib scikit-learn
    ```

### Execution

Run the main Python script:

```bash
python decision_tree_regression_demo.py