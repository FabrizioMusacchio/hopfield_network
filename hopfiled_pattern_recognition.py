"""
Just a simple example of Hopfield network for pattern recognition using Hebbian learning rule.

author: Fabrizio Musacchio
date: Feb 20, 2024

For reproducibility:

conda create -n hopfield python=3.10
conda activate hopfield
conda install -y mamba
mamba install -y numpy matplotlib

"""
# %% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
# %% DEFINE HOPFIELD NETWORK AND PATTERNS
# define the Hopfield network class:
class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))
    
    def train(self, patterns):
        for pattern in patterns:
            pattern = np.reshape(pattern, (self.size, 1)) # reshape to a column vector
            self.weights += np.dot(pattern, pattern.T) # update weights based on Hebbian learning rule, i.e., W = W + p*p'
        self.weights[np.diag_indices(self.size)] = 0 # set diagonal to 0 in order to avoid self-connections
        self.weights /= self.size # normalize weights by the number of neurons to ensure stability of th network
        
    def predict(self, pattern, steps=10):
        pattern = pattern.copy()
        for _ in range(steps):
            for i in range(self.size):
                raw_value = np.dot(self.weights[i, :], pattern)
                pattern[i] = 1 if raw_value >= 0 else -1
        return pattern

    def visualize_patterns(self, patterns, title):
        fig, ax = plt.subplots(1, len(patterns), figsize=(10, 5))
        for i, pattern in enumerate(patterns):
            ax[i].matshow(pattern.reshape((int(np.sqrt(self.size)), -1)), cmap='binary')
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        fig.suptitle(title)
        plt.savefig(f'{title}.png')
        plt.show()

# define test patterns
pattern1 = np.array([-1,1,1,1,1,-1,-1,1,-1]) # representing a simple shape
pattern2 = np.array([1,1,1,-1,1,1,-1,-1,-1]) # another simple shape
patterns = [pattern1, pattern2]
# %% INITIALIZATION
# initialize Hopfield network:
network_size = 9 # this should be square to easily visualize patterns
hn = HopfieldNetwork(network_size)
# %% TRAINING AND PREDICTION
# train the network:
hn.train(patterns)

# corrupt patterns slightly:
corrupted_pattern1 = np.array([-1,1,-1,1,1,-1,-1,1,-1]) # slightly modified version of pattern1
corrupted_pattern2 = np.array([1,1,1,-1,-1,1,-1,-1,-1]) # slightly modified version of pattern2
corrupted_patterns = [corrupted_pattern1, corrupted_pattern2]

# predict (recover) from corrupted patterns:
recovered_patterns = [hn.predict(p) for p in corrupted_patterns]
# %% VISUALIZATION
# visualize original, corrupted, and recovered patterns:
hn.visualize_patterns(patterns, 'Original Patterns')
hn.visualize_patterns(corrupted_patterns, 'Corrupted Patterns')
hn.visualize_patterns(recovered_patterns, 'Recovered Patterns')
# %% END