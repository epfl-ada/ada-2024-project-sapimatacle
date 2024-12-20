# Re-import necessary libraries to ensure the environment is set up correctly
import numpy as np
import matplotlib.pyplot as plt

# Define the rectangular pulse x1(t) and its periodic repetition x2(t)
t_range = np.arange(-10, 10, 0.1)  # Define a time range for plotting
T_period = 5                       # Period for the periodic signal

# Define x1(t) as a rectangular pulse from -2 to 2 (nonzero only in this interval)
x1 = np.where(np.abs(t_range) <= 2, 1, 0)

# Define x2(t) as periodic repetition of x1(t) every T_period (5 units)
x2_periodic = np.tile(x1, int(np.ceil(T_period / 0.1)))  # Repeat enough to cover time range
x2_periodic = x2_periodic[:len(t_range)]  # Clip to match t_range length

# Initialize figure for the flip-and-drag convolution visualization
fig, axes = plt.subplots(3, 1, figsize=(12, 8))
fig.suptitle("Flip-and-Drag Convolution of $x_1(t)$ and $x_2(t)$")

# Plot x1(t) as the original pulse
axes[0].plot(t_range, x1, color="purple")
axes[0].set_title("$x_1(t)$: Rectangular Pulse")
axes[0].set_xlabel("Time (t)")
axes[0].set_ylabel("Amplitude")
axes[0].grid(True)

# Plot x2(t) as the periodic repetition of x1(t)
axes[1].plot(t_range, x2_periodic, color="blue")
axes[1].set_title("$x_2(t)$: Periodic Signal Created from $x_1(t)$")
axes[1].set_xlabel("Time (t)")
axes[1].set_ylabel("Amplitude")
axes[1].grid(True)

# Plot convolution result using the flip-and-drag concept
# Convolution output as a triangular shape for each overlap
x1_flipped_and_dragged = np.convolve(x1, x2_periodic, mode='same') * 0.1
axes[2].plot(t_range, x1_flipped_and_dragged, color="green")
axes[2].set_title("Convolution of $x_1(t)$ with $x_2(t)$ using Flip-and-Drag")
axes[2].set_xlabel("Time (t)")
axes[2].set_ylabel("Amplitude")
axes[2].grid(True)

# Show plot
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
