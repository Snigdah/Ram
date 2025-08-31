"""
Abelian Sandpile Model Simulation
---------------------------------
Author: Afsana Mim Raka

This program implements the Abelian Sandpile model using recursive
depth-first search (DFS) to handle avalanches. It simulates the process 
of self-organized criticality and visualizes both the sandpile evolution 
and avalanche statistics.

Key Features:
- Random initialization of the grid with 0–3 grains.
- Recursive DFS toppling rule (sand is lost at open boundaries).
- History tracking for each time step.
- Avalanche statistics: waiting times, durations, distributions.
- Visualization of avalanche activity and animations.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys


# Increase recursion limit to avoid errors during very large avalanches
sys.setrecursionlimit(10000)


class AbelianSandpile:
    """
    Abelian Sandpile Model using recursive DFS toppling.
    """

    def __init__(self, n=100, random_state=None):
        """
        Initialize the sandpile.

        Parameters:
        n (int): Size of the lattice (n x n).
        random_state (int): Random seed for reproducibility.
        """
        self.n = n
        np.random.seed(random_state)
        # Initialize grid with random values between 0 and 3 (inclusive)
        self.grid = np.random.choice([0, 1, 2, 3], size=(n, n))
        # Store initial grid state
        self.history = [self.grid.copy()]

    def step(self):
        """
        Perform a single step of the sandpile model:
        - Drop a grain of sand at a random location.
        - Relax the system until stable (topple recursively if needed).
        - Save the stable configuration in history.
        """
        i, j = np.random.randint(0, self.n, 2)
        self.grid[i, j] += 1

        if self.grid[i, j] >= 4:
            self.recursive_topple(i, j)

        # Save snapshot of the stable grid
        self.history.append(self.grid.copy())

    def recursive_topple(self, i, j):
        """
        Perform recursive DFS toppling.

        Rules:
        - Subtract 4 grains from the current site.
        - Add 1 grain to each valid neighbor.
        - If a neighbor exceeds threshold (>=4), topple recursively.
        - Grains falling off the boundary are lost (open boundary condition).
        """
        if self.grid[i, j] < 4:
            return  # Already stable, stop recursion

        # Topple the current site
        self.grid[i, j] -= 4

        # Distribute to neighbors with boundary checks
        if i > 0:
            self.grid[i - 1, j] += 1
            if self.grid[i - 1, j] >= 4:
                self.recursive_topple(i - 1, j)
        if i < self.n - 1:
            self.grid[i + 1, j] += 1
            if self.grid[i + 1, j] >= 4:
                self.recursive_topple(i + 1, j)
        if j > 0:
            self.grid[i, j - 1] += 1
            if self.grid[i, j - 1] >= 4:
                self.recursive_topple(i, j - 1)
        if j < self.n - 1:
            self.grid[i, j + 1] += 1
            if self.grid[i, j + 1] >= 4:
                self.recursive_topple(i, j + 1)

    @staticmethod
    def check_difference(grid1, grid2):
        """
        Compute the number of different sites between two grids.

        Parameters:
        grid1, grid2 (np.ndarray): Two sandpile states.

        Returns:
        int: Number of sites that differ.
        """
        return np.sum(grid1 != grid2)

    def simulate(self, n_step):
        """
        Run the model for n_step iterations.

        Parameters:
        n_step (int): Number of grains to drop (time steps).
        """
        for _ in range(n_step):
            self.step()


model = AbelianSandpile(n=100, random_state=0)

plt.figure()
plt.imshow(model.grid, cmap='gray')
plt.title("Initial State")

model.simulate(10000)
plt.figure()
plt.imshow(model.grid, cmap='gray')
plt.title("Final state")
plt.show()



# Compute the pairwise difference between all observed snapshots. This command uses list
# comprehension, a zip generator, and argument unpacking in order to perform this task
# concisely.
all_events =  [model.check_difference(*states) for states in zip(model.history[:-1], model.history[1:])]
# remove transients before the self-organized critical state is reached
all_events = all_events[1000:]
# index each timestep by timepoint
all_events = list(enumerate(all_events))
# remove cases where an avalanche did not occur
all_avalanches = [x for x in all_events if x[1] > 1]
all_avalanche_times = [item[0] for item in all_avalanches]
all_avalanche_sizes = [item[1] for item in all_avalanches]
all_avalanche_durations = [event1 - event0 for event0, event1 in zip(all_avalanche_times[:-1], all_avalanche_times[1:])]


## Waiting time distribution
waiting_times = np.diff(np.array(all_avalanche_times))
plt.figure()
plt.semilogy()
plt.hist(waiting_times)
plt.title('Waiting Time distribution')
plt.xlabel('Waiting time')
plt.ylabel('Number of events')
plt.show()

## Duration distribution
log_bins = np.logspace(np.log10(2), np.log10(np.max(all_avalanche_durations)), 50) # logarithmic bins for histogram
vals, bins = np.histogram(all_avalanche_durations, bins=log_bins)
plt.figure()
plt.loglog(bins[:-1], vals, '.', markersize=10)
plt.title('Avalanche duration distribution')
plt.xlabel('Avalanche duration')
plt.ylabel('Count')
plt.show()

## Visualize activity of the avalanches
# Make an array storing all pairwise differences between the lattice at successive
# timepoints
all_diffs = np.abs(np.diff(np.array(model.history), axis=0))
all_diffs[all_diffs > 0] = 1
all_diffs = all_diffs[np.sum(all_diffs, axis=(1, 2)) > 1] # Filter to only keep big events
most_recent_events = np.sum(all_diffs[-100:], axis=0)
plt.figure(figsize=(5, 5))
plt.imshow(most_recent_events)
plt.title("Avalanche activity in most recent timesteps")
plt.show()

from matplotlib.animation import FuncAnimation
from IPython.display import HTML

activity_sliding2 = all_diffs[-500:]
vmin = np.percentile(activity_sliding2, 1)
# vmin = 0
vmax = np.percentile(activity_sliding2, 99.8)

# Assuming frames is a numpy array with shape (num_frames, height, width)
frames = np.array(activity_sliding2).copy()

fig = plt.figure(figsize=(6, 6))
img = plt.imshow(frames[0], vmin=vmin, vmax=vmax);
plt.xticks([]); plt.yticks([])
# tight margins
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())

def update(frame):
    img.set_array(frame)

ani = FuncAnimation(fig, update, frames=frames, interval=50)
HTML(ani.to_jshtml())
plt.show()

all_diffs = np.abs(np.diff(np.array(model.history), axis=0))
# all_diffs = all_diffs[np.sum(all_diffs, axis=(1, 2)) > 1] # Filter to only keep big events

# Use a trick to calculate the sliding cumulative sum
activity_cumulative = np.cumsum(all_diffs, axis=0)
# activity_sliding = activity_cumulative[50:] - activity_cumulative[:-50]
activity_sliding = all_diffs

plt.figure(figsize=(5, 5))
plt.imshow(activity_sliding[-1])
plt.show()