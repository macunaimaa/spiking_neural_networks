from dataclasses import dataclass

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from jax import grad, jit, lax, vmap
from jax._src.interpreters.batching import batch
from jax._src.numpy.lax_numpy import PadValue
from sympy.external.ntheory import j
from sympy.matrices.expressions.determinant import per

# Training hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 128
NUM_EPOCHS = 1000

# Dataset parameters
Data_DT = 0.1
DT = 0.1
SAMPLING_SIZE = 5
SEQ_LENGTH = 200
N_SAMPLES = 1000

# Network architecture
LAYER_SIZES = [SAMPLING_SIZE, 64, 64, 1]  # input -> hidden1 -> hidden2 -> output


# Network initialization
WEIGHT_SCALE = 0.8  # Scale for weight initialization
BIAS_SCALE = 0.0  # Scale for bias initialization

# Input encoding
RATE_ENCODING_MAX = 1.0  # Maximum spike rate for input encoding

# Surrogate gradient can also be set for 1-5 for more unstable NN
SURROGATE_SIGMA = 5.0  # Sharpness of surrogate gradient

# LIF neuron parameters
LIF_THRESHOLD = 1.0  # Spike threshold (Pth)
LIF_DECAY = 0.50  # Membrane potential decay (D)
LIF_SPIKE_VALUE = 1.0  # Spike reset value (Pspike)
LIF_REFRACTORY = 0.5  # Refractory period (t_ref)
LIF_RESET = 0.2  # Reset potential (Pref)
LIF_MIN = -0.5  # Minimum potential (Pmin)


@dataclass(frozen=True)  # JAX: frozen=True makes dataclass hashable for jit
class SpikingNN:
    """LIF neuron configuration (same as main.py)"""

    T: float = 200.0
    dt: float = DT
    Pref: float = LIF_RESET
    Pmin: float = LIF_MIN
    Pth: float = LIF_THRESHOLD
    D: float = LIF_DECAY
    Pspike: float = LIF_SPIKE_VALUE
    t_ref: float = LIF_REFRACTORY

    def time(self):
        return jnp.arange(0, self.T + self.dt, self.dt)


# JAX: Register SpikingNN as a pytree so it can be used with jit
jax.tree_util.register_dataclass(
    SpikingNN,
    data_fields=["T", "dt", "Pref", "Pmin", "Pth", "D", "Pspike", "t_ref"],
    meta_fields=[],
)


# JAX: custom_vjp allows defining custom gradients for non-differentiable operations
@jax.custom_vjp
def spike_function(P, threshold):
    """Spike generation with surrogate gradient.

    Forward: True Heaviside step (maintains biological accuracy)
    Backward: Smooth surrogate for gradient flow
    """
    return (P >= threshold).astype(jnp.float32)


def spike_fwd(P, threshold):
    spikes = (P >= threshold).astype(jnp.float32)
    return spikes, (P, threshold)


def spike_bwd(res, g):
    """Backward pass: use fast sigmoid as surrogate gradient"""
    P, threshold = res
    # Use hyperparameter for gradient sharpness
    surrogate_grad = (
        SURROGATE_SIGMA / (1.0 + jnp.abs(SURROGATE_SIGMA * (P - threshold))) ** 2
    )
    return (g * surrogate_grad, None)  # None = no gradient for threshold


spike_function.defvjp(spike_fwd, spike_bwd)


def init_layer_params(key, in_dim, out_dim):
    """Initialize weights and biases for one layer.

    JAX: Pytree parameters (nested dicts/arrays) work seamlessly with optimizers
    """
    key_w, key_b = jax.random.split(key)
    W = jax.random.normal(key_w, (out_dim, in_dim)) * WEIGHT_SCALE
    b = jax.random.normal(key_b, (out_dim,)) * BIAS_SCALE
    return {"W": W, "b": b}


def init_network_params(key, layer_sizes):
    """Initialize all network parameters as a pytree"""
    keys = jax.random.split(key, len(layer_sizes) - 1)
    params = {}
    layer_names = ["input", "hidden", "output"]

    for i, (in_dim, out_dim) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        params[layer_names[i]] = init_layer_params(keys[i], in_dim, out_dim)

    return params


def spiking_layer(params, state, inputs, config):
    """Single spiking layer with LIF dynamics.

    JAX: lax.scan provides efficient temporal iteration with O(1) memory
    """
    seq_length = inputs.shape[1]  # inputs is (input_dim, seq_length)

    time = jnp.arange(seq_length) * config.dt

    n_neurons = params["W"].shape[0]

    def step(carry, t_and_input):
        P_prev, t_rest_prev = carry
        S_i, t = t_and_input

        # Compute synaptic input: I = W @ input + b
        I_syn = params["W"] @ S_i + params["b"]

        in_refractory = t <= t_rest_prev

        P_normal = jnp.where(P_prev > config.Pmin, P_prev + I_syn - config.D, 0.0)
        P = jnp.where(in_refractory, config.Pref, P_normal)

        # Use surrogate gradient spike function
        spiked = spike_function(P, config.Pth)

        # Update potential and refractory period
        P_new = jnp.where(spiked > 0.5, P + config.Pspike, P)
        t_rest_new = jnp.where(spiked > 0.5, t + config.t_ref, t_rest_prev)

        return (P_new, t_rest_new), spiked

    # Initialize state for all neurons in this layer
    init_carry = state

    # JAX: lax.scan iterates over the first dimension of each array in the tuple
    inputs_with_time = (
        inputs.T,
        time,
    )

    final_state, spike_train = lax.scan(step, init_carry, inputs_with_time)
    return final_state, spike_train


def forward_pass(params, config, inputs):
    """Full network forward pass through all layers.

    Args:
        params: Network parameters (pytree)
        config: SpikingNN configuration
        inputs: Input sequence [seq_length, input_dim]

    Returns:
        predictions: Continuous output predictions [seq_length,]
    """
    seq_length = inputs.shape[0]

    # Initialize states for each layer
    state_input = (
        jnp.zeros(LAYER_SIZES[1]),
        jnp.zeros(LAYER_SIZES[1]),
    )
    state_hidden = (
        jnp.zeros(LAYER_SIZES[2]),
        jnp.zeros(LAYER_SIZES[2]),
    )

    # Layer 1: Input encoding to spikes
    _, spikes_input = spiking_layer(params["input"], state_input, inputs.T, config)

    # Layer 2: Hidden layer processing
    _, spikes_hidden = spiking_layer(
        params["hidden"], state_hidden, spikes_input.T, config
    )

    # Layer 3: Output decoding (linear readout, no spiking)
    # Simple linear transformation of spike rates
    output = params["output"]["W"] @ spikes_hidden.T + params["output"]["b"][:, None]

    return output.squeeze()


def generate_sine_wave(key, n_samples, seq_length, dt, sampling_size):
    """Generate sine wave time series for sliding window prediction.

    Returns:
        inputs: [n_samples, seq_length, sampling_size] - sliding windows
        targets: [n_samples, seq_length, 1] - next values to predict
    """
    key_freq, key_phase = jax.random.split(key)

    # Random frequencies and phases for variety
    freq = jax.random.uniform(key_freq, (n_samples,), minval=0.5, maxval=2.0)
    phase = jax.random.uniform(key_phase, (n_samples,), minval=0, maxval=2 * jnp.pi)

    t = jnp.arange(seq_length + sampling_size) * dt

    # Generate sine waves: [n_samples, seq_length + sampling_size]
    signals = jnp.sin(2 * jnp.pi * freq[:, None] * t[None, :] + phase[:, None])

    # Input: sliding windows, Target: next value after each window
    window_idx = jnp.arange(seq_length)[:, None] + jnp.arange(sampling_size)[None, :]
    inputs = jnp.take(signals, window_idx, axis=1)
    targets = signals[:, sampling_size : sampling_size + seq_length, None]

    return jnp.array(inputs, dtype=jnp.float32), jnp.array(targets, dtype=jnp.float32)


def rate_encode(signal):
    """Convert continuous signal to spike rate encoding.

    Normalizes sine wave from [-1, 1] to [0, RATE_ENCODING_MAX]
    """
    return (signal + 1.0) / 2.0 * RATE_ENCODING_MAX


def mse_loss(params, config, inputs, targets):
    encoded_inputs = rate_encode(inputs)

    # Forward pass
    predictions = forward_pass(params, config, encoded_inputs)

    # Compute MSE
    loss = jnp.mean((predictions - targets.squeeze()) ** 2)
    return loss


def plot_predictions(predictions, targets, title="Predictions vs Targets"):
    """Plot model predictions against targets for a single sequence."""
    pred = jnp.array(predictions).squeeze()
    tgt = jnp.array(targets).squeeze()

    plt.figure(figsize=(10, 4))
    plt.plot(pred, label="predictions")
    plt.plot(tgt, label="targets", alpha=0.8)
    plt.title(title)
    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()


def batched_mse_loss(params, config, inputs, targets):
    per_sample = jax.vmap(mse_loss, in_axes=(None, None, 0, 0))(
        params, config, inputs, targets
    )

    return jnp.mean(per_sample)


@jax.jit(static_argnames=("config"))
def train_step(params, opt_state, config, inputs, targets):
    loss, grads = jax.value_and_grad(batched_mse_loss)(params, config, inputs, targets)

    # Apply optimizer (Adam) updates
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss


if __name__ == "__main__":
    print("Initializing Spiking Neural Network")

    print(f"Architecture: {LAYER_SIZES}")

    print(
        f"Learning rate: {LEARNING_RATE}, Batch size: {BATCH_SIZE}, Epochs: {NUM_EPOCHS}\n"
    )

    # 1. Initialize keys
    key = jax.random.PRNGKey(42)
    key, key_data, key_params = jax.random.split(key, 3)

    # 2. Sine wave dataset
    print("Generating dataset")
    train_inputs, train_targets = generate_sine_wave(
        key_data, N_SAMPLES, SEQ_LENGTH, dt=Data_DT, sampling_size=SAMPLING_SIZE
    )
    print(f"Dataset shape: inputs={train_inputs.shape}, targets={train_targets.shape}")

    # 3. Initialize network parameters
    print("\nInitializing network parameters")
    params = init_network_params(key_params, LAYER_SIZES)
    config = SpikingNN(T=SEQ_LENGTH * DT, dt=DT)

    # Print parameter shapes
    for layer_name, layer_params in params.items():
        print(
            f"  {layer_name}: W={layer_params['W'].shape}, b={layer_params['b'].shape}"
        )

    # 4. Setup Adam optimizer with optax
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(params)

    # 5. Training loop
    print(f"\nStarting training for {NUM_EPOCHS} epochs")
    num_batches = N_SAMPLES // BATCH_SIZE

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0

        # Shuffle data each epoch
        key, shuffle_key = jax.random.split(key)
        perm = jax.random.permutation(shuffle_key, N_SAMPLES)
        train_inputs_shuffled = train_inputs[perm]
        train_targets_shuffled = train_targets[perm]

        # Batch iteration
        for batch_idx in range(num_batches):
            batch_start = batch_idx * BATCH_SIZE
            batch_end = batch_start + BATCH_SIZE

            batch_inputs = train_inputs_shuffled[batch_start:batch_end]
            batch_targets = train_targets_shuffled[batch_start:batch_end]

            # For simplicity, training on just the first sample vmap implementation requires better state management
            params, opt_state, loss = train_step(
                params, opt_state, config, batch_inputs, batch_targets
            )
            epoch_loss += loss

        avg_loss = epoch_loss / num_batches

        # Log progress
        if epoch % 10 == 0 or epoch == NUM_EPOCHS - 1:
            print(f"Epoch {epoch:3d}/{NUM_EPOCHS} - Loss: {avg_loss:.6f}")

    print("\nTraining completed!")

    # 6. Test on a sample
    print("\nTesting on sample")
    test_input = train_inputs[0]
    test_target = train_targets[-1]

    encoded_test = rate_encode(test_input)
    test_pred = forward_pass(params, config, encoded_test)

    test_mse = jnp.mean((test_pred - test_target.squeeze()) ** 2)
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Sample predictions (first 10): {test_pred[:10]}")
    print(f"Sample targets (first 10): {test_target.squeeze()[:10]}")
    plot_predictions(test_pred, test_target, title="Test Predictions vs Targets")
