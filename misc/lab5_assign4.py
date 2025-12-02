import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("measurement_errors_cyan.csv")

# Extract variables
d = df["meas_d"].values
theta = df["meas_theta"].values
e_d = df["err_d"].values
e_theta = df["err_theta"].values

# Compute squared errors
e_d2 = e_d**2
e_theta2 = e_theta**2

# ----------------------------
# Linear regression for σ_d²(d)
# ----------------------------
A = np.vstack([np.ones_like(d), d]).T
coef_d, _, _, _ = np.linalg.lstsq(A, e_d2, rcond=None)
a0, a1 = coef_d

print("Distance variance model:")
print(f"sigma_d^2(d) = {a0:.6f} + {a1:.6f} * d")

# -------------------------------
# Linear regression for σ_theta²(d)
# -------------------------------
coef_theta, _, _, _ = np.linalg.lstsq(A, e_theta2, rcond=None)
b0, b1 = coef_theta

print("\nBearing variance model:")
print(f"sigma_theta^2(d) = {b0:.6f} + {b1:.6f} * d")

# -------------------------
# Optional plotting section
# -------------------------

plt.figure()
plt.scatter(d, e_d2, s=3, label="Measured err_d²")
plt.plot(d, a0 + a1*d, label="Fitted variance model", linewidth=2)
plt.xlabel("Measured distance d (m)")
plt.ylabel("err_d²")
plt.title("Distance Error Variance Model")
plt.legend()
plt.grid(True)

plt.figure()
plt.scatter(d, e_theta2, s=3, label="Measured err_theta²")
plt.plot(d, b0 + b1*d, label="Fitted variance model", linewidth=2)
plt.xlabel("Measured distance d (m)")
plt.ylabel("err_theta²")
plt.title("Bearing Error Variance Model")
plt.legend()
plt.grid(True)

plt.show()
