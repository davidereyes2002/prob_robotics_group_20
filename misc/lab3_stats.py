import numpy as np

def load_values(filename):
    values = []
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("data:"):
                values.append(float(line.split()[1]))
    return np.array(values)

closed_vals = load_values("door_closed.txt")
open_vals = load_values("door_opened.txt")

def summarize(name, values):
    print(f"\n{name}:")
    print(f"  Count: {len(values)}")
    print(f"  Mean: {np.mean(values):.2f}")
    print(f"  Std Dev: {np.std(values):.2f}")
    print(f"  Min: {np.min(values):.2f}")
    print(f"  Max: {np.max(values):.2f}")

summarize("Door Closed", closed_vals)
summarize("Door Opened", open_vals)
