import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, Subset
import torch
import sigkernel
import copy
import MNIST_funcs

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import TwoLocal, StatePreparation

class StrokeDataset(Dataset):
    def __init__(self, inputs, targets):
        # label_map: dict, maps original labels to binary labels
        self.inputs = np.array(inputs, dtype=object)
        self.targets = np.array(targets, dtype=np.int64)  # shape: (N,)

    def __getitem__(self, idx):
        path = self.inputs[idx]           # shape: (T, 2)
        label = self.targets[idx]         # int
        return path, label
    
    def __len__(self):
        return len(self.inputs)
    
class BinaryMappedDataset(Dataset):
    def __init__(self, dataset, compression_lvl, allowed_labels=(0,1), label_map={0:-1, 1:1}):
        # compression_lvl: int, how many points to keep in the compressed trajectory
        self.inputs = []
        self.targets = []
        self.kernel = []
        i = 0
        for x,y in dataset:
            # if y in allowed_labels:
            if y in allowed_labels and i < 1000:  # limit to 1000 samples
                i += 1
                # compress the trajectory
                pen_lift = x[:, -1]
                compressed_x = MNIST_funcs.stream_normalise_mean_and_range(x[:, :2])
                compressed_x = np.column_stack((compressed_x, pen_lift))
                # compressed_x = compress_digit_by_arclength_without_interpolation(compressed_x, max_points=compression_lvl)[0]
                compressed_x = compress_digit_by_arclength(compressed_x, max_points=compression_lvl)[0]
                
                if compressed_x.shape != (compression_lvl, 3): # try to debug
                    print(f"sample {i} with shape {compressed_x.shape} != {compression_lvl}, 3)")
                    continue
                self.inputs.append(compressed_x)
                self.targets.append(label_map[y])
        self.inputs = np.array(self.inputs, dtype=np.float64)
        self.targets = np.array(self.targets, dtype=np.int64)

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        path = self.inputs[idx]           # shape: (T, 2)
        label = self.targets[idx]         # int
        return path, label
    
    def set_kernel(self, kernel):
        self.kernel = np.array(kernel, dtype=np.float64)  # shape: (N, T, D)


class CompressedDataset(Dataset):
    def __init__(self, dataset, compression_lvl, label_map={0:-1, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1}, max_per_class=1000):
        # label_map: dict, maps original labels to binary labels
        self.inputs = []
        self.targets = []  # use original targets
        self.kernel = []
        self.targets_bin = []
        
        count_0 = 0
        count_other = 0
        for x,y in dataset:
            if y == 0 and count_0 < max_per_class:
                count_0 += 1
            elif y != 0 and count_other < max_per_class:
                count_other += 1
            else:
                continue

            x = np.cumsum(x, axis=0)  # convert to absolute position
            # compress the trajectory

            # normalize first compress after
            # pen_lift = x[:, -1]
            # compressed_x = MNIST_funcs.stream_normalise_mean_and_std(x[:, :2])
            # compressed_x = np.column_stack((compressed_x, pen_lift))
            # # compressed_x = compress_digit_by_arclength_without_interpolation(x, max_points=compression_lvl)[0]
            # compressed_x = compress_digit_by_arclength(compressed_x, max_points=compression_lvl)[0]

            # compress first normalize after
            # Since the compress algotithm is based on arc length, it is better to compress first
            compressed_x = compress_digit_by_arclength(x, max_points=compression_lvl)[0]
            pen_lift = compressed_x[:, -1]
            compressed_x = MNIST_funcs.stream_normalise_mean_and_std(compressed_x[:, :2])
            compressed_x = np.column_stack((compressed_x, pen_lift))
            # compressed_x = compress_digit_by_arclength_without_interpolation(x, max_points=compression_lvl)[0]
            padded_x = compressed_x
            if compressed_x.shape[0] < compression_lvl: # if vec is with wrong dim copy the last element to correct 
                print(f"sample {count_0+count_other} with shape {compressed_x.shape} != {compression_lvl}, 3), padding {compression_lvl-compressed_x.shape[0]} times to solve")
                pad_rows = np.repeat(compressed_x[-1:], compression_lvl-compressed_x.shape[0], axis=0)
                padded_x = np.vstack([compressed_x, pad_rows])

            self.inputs.append(padded_x)
            self.targets_bin.append(label_map[y])
            self.targets.append(y)

            # stop early if we have enough samples
            if count_0 >= max_per_class and count_other >= max_per_class:
                break

        # Convert lists to numpy arrays
        self.inputs = np.array(self.inputs, dtype=np.float64)
        self.targets_bin = np.array(self.targets_bin, dtype=np.int64)
        self.targets = np.array(self.targets, dtype=np.int64)  # keep original targets
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        path = self.inputs[idx]           # shape: (T, 2)
        label = self.targets[idx]         # int
        return path, label
    
    def set_kernel(self, kernel):
        self.kernel = np.array(kernel, dtype=np.float64)  # shape: (N, T, D)
        
def plot_pen_drawn_digit(data, ax=None, abs_positions=True):
    """
    Plot digit trajectory from stroke-based data.
    
    Parameters:
    - data: np.ndarray of shape (T, 3), where:
        data[:, 0] = dx (horizontal displacement)
        data[:, 1] = dy (vertical displacement)
        data[:, 2] = stroke_end (1 if stroke ends at this point)
    - ax: optional matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Step 1: Compute absolute positions from relative displacements
    if abs_positions:
        # If data is already absolute positions, use directly
        # it is the case for compressed data
        positions = data[:, :2]
    else:
        positions = np.cumsum(data[:, :2], axis=0)

    # Step 2: Split into strokes using stroke_end
    strokes = []
    current_stroke = [positions[0]]
    for i in range(1, len(positions)):
        current_stroke.append(positions[i])
        if data[i, 2] == 1:
            strokes.append(np.array(current_stroke))
            if i + 1 < len(positions):
                current_stroke = [positions[i + 1]]
    
    # Append last stroke if needed
    if current_stroke:
        strokes.append(np.array(current_stroke))

    # Step 3: Plot strokes
    for stroke in strokes:
        ax.scatter(stroke[:, 0], -stroke[:, 1])  # invert y for image-like orientation

    ax.axis('equal')
    # ax.axis('off')
    return ax

def compress_digit_by_arclength_without_interpolation(data, max_points=10):
    """
    ATTENTION: This function is not updated (use compress_digit_by_arclength instead).
    Compress pen trajectory using arc-length parameterization.

    Parameters:
    - data: np.ndarray of shape (T, 3) with [dx, dy, stroke_end]
    - max_points: number of points to keep (default 10)

    Returns:
    - List of np.ndarrays of shape (<=max_points, 3) for each trajectory
    """
    if data.ndim == 2:
        data = data[None, :] # add batch dimension if needed
    
    compressed_list = []
    for sample in data:
        # Step 1: Compute absolute positions
        # pos = np.cumsum(sample[:, :2], axis=0)  # (T, 2)
        pos = sample[:, :2] # trying to use absolute positions directly
        stroke_end = sample[:, 2]

        # Step 2: Compute arc length
        deltas = np.diff(pos, axis=0)
        segment_lengths = np.linalg.norm(deltas, axis=1)
        arc_length = np.concatenate([[0], np.cumsum(segment_lengths)])  # shape (T,)

        # Step 3: Sample uniformly along arc length
        if arc_length[-1] == 0 or len(pos) <= max_points:
            compressed_list.append(np.hstack([pos, stroke_end.reshape(-1, 1)]))
            continue

        sampled_lengths = np.linspace(0, arc_length[-1], max_points)

        # Step 4: Find closest points instead of interpolating
        closest_indices = [np.argmin(np.abs(arc_length - s)) for s in sampled_lengths]
        closest_indices = np.unique(closest_indices)  # remove duplicates

        compressed_pos = pos[closest_indices]
        sampled_stroke = stroke_end[closest_indices]

        compressed = np.hstack([compressed_pos, sampled_stroke[:, None]])
        compressed_list.append(compressed)
    return compressed_list


def compress_digit_by_arclength(data, max_points=10):
    """
    Compress pen trajectory using arc-length parameterization,
    interpolating only along pen-down segments (drawn path).
    
    Parameters:
    - data: np.ndarray of shape (T, 3) with [pos_x, pos_y, stroke_end]
    - max_points: number of points to keep (default 10)

    Returns:
    - List of np.ndarrays of shape (<=max_points, 3)
    """
    if data.ndim == 2:
        data = data[None, :]

    compressed_list = []
    for sample in data:
        # pos = np.cumsum(sample[:, :2], axis=0)
        pos = sample[:, :2] # trying to use absolute positions directly
        stroke_end = sample[:, 2]

        # Compute arc length ignoring pen-up transitions
        arc_length = np.zeros(len(pos))
        total_length = 0.0
        for i in range(1, len(pos)):
            if stroke_end[i - 1] == stroke_end[i]:
                dist = np.linalg.norm(pos[i] - pos[i - 1])
                total_length += dist
            arc_length[i] = total_length

        # Skip if too short
        if total_length == 0 or len(pos) <= max_points:
            compressed_list.append(np.hstack([pos, stroke_end.reshape(-1, 1)]))
            continue

        # Sample positions along arc length
        sampled_lengths = np.linspace(0, total_length, max_points)
        x_interp = np.interp(sampled_lengths, arc_length, pos[:, 0])
        y_interp = np.interp(sampled_lengths, arc_length, pos[:, 1])
        compressed_pos = np.stack([x_interp, y_interp], axis=1)

        # Map stroke_end intelligently
        sampled_indices = np.searchsorted(arc_length, sampled_lengths, side='right') - 1
        sampled_indices = np.clip(sampled_indices, 0, len(stroke_end) - 1)
        sampled_stroke = np.zeros(max_points)

        # Recover stroke_end locations
        original_breaks = np.where(stroke_end == 1)[0]
        used = set()
        for break_idx in original_breaks:
            for i in range(1, max_points):
                if sampled_indices[i - 1] <= break_idx < sampled_indices[i]:
                    if i - 1 not in used:
                        sampled_stroke[i - 1] = 1
                        used.add(i - 1)
                    break

        compressed = np.hstack([compressed_pos, sampled_stroke[:, None]])
        compressed_list.append(compressed)

    return compressed_list

def compare_compression(original, compressed, title="Digit Compression Comparison"):
    """
    Plot the original and compressed digit side by side.
    
    Parameters:
    - original: np.ndarray of shape (T, 3) with [dx, dy, stroke_end]
    - compressed: np.ndarray of shape (<=T, 3) with [x, y, stroke_end] or [dx, dy, stroke_end]
    - title: optional title for the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(title)

    # Left: original
    plot_pen_drawn_digit(original, ax=axes[0], abs_positions=True)
    axes[0].set_title("Original")

    # Right: compressed
    # If already absolute positions, use directly
    plot_pen_drawn_digit(compressed, ax=axes[1], abs_positions=True)   
    axes[1].set_title("Compressed")

    plt.tight_layout()
    plt.show()

    
def plot_kernel_matrix(matrix, label, idx):
    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title(f"Kernel matrix (Label: {label}) - Sample {idx}")
    plt.xlabel("Reference point index")
    plt.ylabel("Input point index")
    plt.show()

def qclassifier(params, x=None):
    n_layer, n_qubit, _ = params.shape
    qc = QuantumCircuit(n_qubit)
    if x is not None:
        # Initialize the circuit with the input vector x
        if len(x) != 2**n_qubit:
            raise ValueError(f"Input vector x must have length {2**n_qubit}, got {len(x)}")
        qc.initialize(x, None, normalize=True)

    for i in range(n_layer):
        if i==0:
            for j in range(n_qubit-1):
                qc.cx(j, j+1)
        
        # Variational layers
        for j in range(n_qubit):
            theta, phi, lam = params[i, j]
            qc.u(theta, phi, lam, j)

        for j in range(n_qubit-1):
            qc.cx(j, j+1)

    return qc

from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

# Cost function: Mean Squared Error (MSE)
def mse_loss(predict, label):
    return np.mean((predict - label)**2)

def gen_eval_circuit(param_val, kernel_mat, estimator: StatevectorEstimator):
    """
    Generate and evaluate circuit for a given parameter value and input kernel.
    The kernel is a matrix, that represents a statevector in the computational basis, where
    each cell corresponds to a basis state amplitude.
    
    Returns the expectation value of the observable.
    param_val: numpy array of shape (n_layer, n_qubit, 3)
    kernel_mat: numpy array of shape (S, T);
    estimator: StatevectorEstimator instance
    """
    # Prepare the input state from the kernel
    s, t = kernel_mat.shape
    n_layer, n_qubit, _ = param_val.shape
    assert s*t <= 2**n_qubit, "Kernel matrix size exceeds the number of qubits"
    
    kernel_mat = kernel_mat.flatten()
    if s*t != 2**n_qubit:
        # If the number of cells in the kernel does not match the number of qubits,
        # pad the state vector with zeros
        kernel_mat = np.pad(kernel_mat, (0, 2**n_qubit - s*t), mode='constant')
        kernel_mat = kernel_mat / np.linalg.norm(kernel_mat)  # Normalize the state vector
    
    # Create the quantum circuit for the classifier
    qc = qclassifier(param_val, kernel_mat)
    observable = SparsePauliOp.from_list([("Z" + "I" * (qc.num_qubits - 1), 1.0)])
    result = estimator.run([(qc, observable)]).result()
    exp = dict(result[0].data.items())['evs'].item()
    return exp

def cost(param_val, X, y, estimator: StatevectorEstimator):
    exp = [gen_eval_circuit(param_val, X[i], estimator) for i in range(len(X))]
    return mse_loss(np.array(exp), y)

# Accuracy is defined as the proportion of correctly classified samples out of the total'
def accuracy(predicts, labels):
    assert len(predicts) == len(labels)
    return np.sum((np.sign(predicts)*labels+1)/2)/len(predicts)


class OptimizerLog:
    """Log to store optimization results."""
    def __init__(self):
        self.losses = []
        self.accuracies = []

    def update(self, _nfevs, _theta, ftheta, *_):
        self.losses.append(ftheta)



def plot_compression_comparison_side_by_side(digit, original_dataset, compressed_dataset):
    # Filter samples of the specified digit
    original_samples = [(x, i) for i, (x, y) in enumerate(original_dataset) if y == digit]
    compressed_samples = [(x, i) for i, (x, y) in enumerate(compressed_dataset) if y == digit]

    # Take the first 9 samples (ensure we have enough)
    n = min(9, len(original_samples), len(compressed_samples))
    if n < 9:
        print(f"Warning: only found {n} samples for digit {digit}.")

    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(8, 2.5 * n))

    for i in range(n):
        orig_path, _ = original_samples[i]
        comp_path, _ = compressed_samples[i]
        
        # if i == 0:
            # print(orig_path)
        
        # Plot original
        ax_orig = axes[i, 0]
        # ax_orig.plot(orig_path[:, 0], orig_path[:, 1], label='Original', alpha=0.8)
        plot_pen_drawn_digit(orig_path, ax=ax_orig)
        ax_orig.set_title(f"Original - Sample {i+1}")
        ax_orig.axis('equal')
        ax_orig.axis('off')

        # Plot compressed
        ax_comp = axes[i, 1]
        ax_comp.plot(comp_path[:, 0], -comp_path[:, 1], 'o--', label='Compressed', alpha=0.8)
        ax_comp.set_title(f"Compressed - Sample {i+1}")
        ax_comp.axis('equal')
        ax_comp.axis('off')

    plt.tight_layout()
    plt.show()