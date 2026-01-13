from dataclasses import dataclass
from this import s

import jax
import jax.numpy as jnp
import optax
from jax import grad, jit, lax, vmap
from jax._src.checkify import functionalization_error
from jax._src.typing import StaticIndex


@dataclass(frozen=True)
class SpikingNN:
    """LIF neuron configuration (same as main.py)"""

    T: float = 200.0
    dt: float = 0.125
    Pref: float = 0.0
    Pmin: float = -1.0
    Pth: float = 10.0
    D: float = 0.25
    Pspike: float = 4.0
    t_ref: float = 5.0

    def time(self):
        return jnp.arange(0, self.T + self.dt, self.dt)


# JAX: Register SpikingNN as a pytree so it can be used with jit
jax.tree_util.register_dataclass(
    SpikingNN,
    data_fields=["T", "dt", "Pref", "Pmin", "Pth", "D", "Pspike", "t_ref"],
    meta_fields=[],
)


# Hyperparameters
LEARNING_RATE = 0.002
BATCH_SIZE = 32
NUM_EPOCHS = 100
LAYER_SIZES = [1, 128, 64, 1]  # input -> hidden1 -> hidden2 -> output
SEQ_LENGTH = 100
N_SAMPLES = 1000


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
    sigma = 10.0  # Controls gradient sharpness
    surrogate_grad = sigma / (1.0 + jnp.abs(sigma * (P - threshold))) ** 2
    return (g * surrogate_grad, None)  # None = no gradient for threshold


spike_function.defvjp(spike_fwd, spike_bwd)


def init_layer_params(key, in_dim, out_dim, scale=0.5):
    """Initialize weights and biases for one layer.

    JAX: Pytree parameters (nested dicts/arrays) work seamlessly with optimizers
    """
    key_w, key_b = jax.random.split(key)
    W = jax.random.normal(key_w, (out_dim, in_dim)) * scale
    b = jax.random.normal(key_b, (out_dim,)) * 0.1
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
    print("state_input shape:", state_input)
    print("state_hidden shape:", state_hidden)

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


# =============================================================================
# Dataset Generation
# =============================================================================


def generate_sine_wave(key, n_samples, seq_length, dt):
    """Generate sine wave time series for prediction task.

    Returns:
        inputs: [n_samples, seq_length, 1] - current values
        targets: [n_samples, seq_length, 1] - next values to predict
    """
    key_freq, key_phase = jax.random.split(key)

    # Random frequencies and phases for variety
    freq = jax.random.uniform(key_freq, (n_samples,), minval=0.5, maxval=2.0)
    phase = jax.random.uniform(key_phase, (n_samples,), minval=0, maxval=2 * jnp.pi)

    t = jnp.arange(seq_length + 1) * dt

    # Generate sine waves: [n_samples, seq_length + 1]
    signals = jnp.sin(2 * jnp.pi * freq[:, None] * t[None, :] + phase[:, None])

    # Input: all but last timestep, Target: all but first timestep
    inputs = signals[:, :-1, None]  # [n_samples, seq_length, 1]
    targets = signals[:, 1:, None]  # [n_samples, seq_length, 1]

    return jnp.array(inputs, dtype=jnp.float32), jnp.array(targets, dtype=jnp.float32)


def rate_encode(signal, max_rate=10.0):
    """Convert continuous signal to spike rate encoding.

    Normalizes sine wave from [-1, 1] to [0, max_rate]
    """
    return (signal + 1.0) / 2.0 * max_rate


def mse_loss(params, config, inputs, targets):
    """Mean squared error loss for regression.

    JAX: Automatic differentiation will compute gradients through this function
    """
    # Encode inputs as spike rates
    encoded_inputs = rate_encode(inputs)

    # Forward pass
    predictions = forward_pass(params, config, encoded_inputs)

    # Compute MSE
    loss = jnp.mean((predictions - targets.squeeze()) ** 2)
    return loss


@jax.jit(static_argnames=("config"))
def train_step(params, opt_state, config, inputs, targets):
    """Single training step with gradient descent.

    JAX: value_and_grad computes both loss and gradients in one pass
    """
    loss, grads = jax.value_and_grad(mse_loss)(params, config, inputs, targets)

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
        key_data, N_SAMPLES, SEQ_LENGTH, dt=0.1
    )
    print(f"Dataset shape: inputs={train_inputs.shape}, targets={train_targets.shape}")

    # 3. Initialize network parameters
    print("\nInitializing network parameters")
    params = init_network_params(key_params, LAYER_SIZES)
    config = SpikingNN(T=SEQ_LENGTH * 0.1, dt=0.1)

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

            # JAX: vmap vectorizes train_step across the batch dimension
            # This processes all samples in parallel
            batched_train_step = vmap(
                lambda inp, tgt: train_step(params, opt_state, config, inp, tgt),
                in_axes=(0, 0),
            )

            # For simplicity, training on just the first sample vmap implementation requires better state management
            params, opt_state, loss = train_step(
                params, opt_state, config, batch_inputs[0], batch_targets[0]
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
    test_target = train_targets[0]

    encoded_test = rate_encode(test_input)
    test_pred = forward_pass(params, config, encoded_test)

    test_mse = jnp.mean((test_pred - test_target.squeeze()) ** 2)
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Sample predictions (first 10): {test_pred[:10]}")
    print(f"Sample targets (first 10): {test_target.squeeze()[:10]}")
