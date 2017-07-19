from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


def binary_decay(learning_rate, global_step, decay_steps, decay_rate,
                       staircase=False, name=None):

    """
    Applies inverse time decay to the initial learning rate.

    When training a model, it is often recommended to lower the learning rate as
    the training progresses.  This function applies an inverse decay function
    to a provided initial learning rate.  It requires an `global_step` value to
    compute the decayed learning rate.  You can just pass a TensorFlow variable
    that you increment at each training step.

    The function returns the decayed learning rate.  It is computed as:

    ```python
    decayed_learning_rate = learning_rate / (1 + decay_rate * t)
    ```

    Example: decay 1/t with a rate of 0.5:

    ```python
    ...
    global_step = tf.Variable(0, trainable=False)
    learning_rate = 0.1
    k = 0.5
    learning_rate = tf.train.inverse_time_decay(learning_rate, global_step, k)

    # Passing global_step to minimize() will increment it at each step.
    learning_step = (
    tf.train.GradientDescentOptimizer(learning_rate)
    .minimize(...my loss..., global_step=global_step)
    )
    ```

    Args:
    learning_rate: A scalar `float32` or `float64` `Tensor` or a
    Python number.  The initial learning rate.
    global_step: A Python number.
    Global step to use for the decay computation.  Must not be negative.
    decay_steps: How often to apply decay.
    decay_rate: A Python number.  The decay rate.
    staircase: Whether to apply decay in a discrete staircase, as opposed to
    continuous, fashion.
    name: String.  Optional name of the operation.  Defaults to
    'InverseTimeDecay'.

    Returns:
    A scalar `Tensor` of the same type as `learning_rate`.  The decayed
    learning rate.

    Raises:
    ValueError: if `global_step` is not supplied.
    """
    if global_step is None:
        raise ValueError("global_step is required for inverse_time_decay.")
    with ops.name_scope(name, "BinaryDecay",
                  [learning_rate, global_step, decay_rate]) as name:
        learning_rate = ops.convert_to_tensor(learning_rate, name="learning_rate")
        dtype = learning_rate.dtype
        global_step = math_ops.cast(global_step, dtype)
        decay_steps = math_ops.cast(decay_steps, dtype)
        decay_rate = math_ops.cast(decay_rate, dtype)
        p = global_step / decay_steps
        p = math_ops.floor(p)
        const = math_ops.cast(constant_op.constant(0.5), learning_rate.dtype)
        min_lr = math_ops.cast(constant_op.constant(1e-30), learning_rate.dtype)
        exp = math_ops.pow(const,p)
        lr =  math_ops.multiply(learning_rate, exp, name=name)
        return math_ops.maximum(min_lr,lr)