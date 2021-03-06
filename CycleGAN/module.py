import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras

# ==============================================================================
# =                                  networks                                  =
# ==============================================================================


# def _get_norm_layer(norm):
#    if norm == 'none':
#        return lambda: lambda x: x
#    elif norm == 'batch_norm':
#        return keras.layers.BatchNormalization
#    elif norm == 'instance_norm':
#        return tfa.layers.InstanceNormalization
#    elif norm == 'layer_norm':
#        return keras.layers.LayerNormalization
# ==============================================================================
#                   Replacing instance normalization
#                   See: https://github.com/NVlabs/stylegan2/blob/7d3145d23013607b987db30736f89fb1d3e10fad/training/networks_stylegan.py
# From StyleGAN2 paper https://arxiv.org/pdf/1912.04958.pdf
# ==============================================================================

def Norm(x):
    epsilon = 1e-8
    orig_dtype = x.dtype
    x = tf.cast(x, tf.float32)
    x -= tf.reduce_mean(x, axis=[2, 3], keepdims=True)
    epsilon = tf.constant(epsilon, dtype=x.dtype, name='epsilon')
    x *= tf.math.rsqrt(tf.reduce_mean(tf.square(x),
                                 axis=[2, 3], keepdims=True) + epsilon)
    x = tf.cast(x, orig_dtype)
    return x

# specific implementation taken from https://github.com/ialhashim/StyleGAN-Tensorflow2/blob/master/stylegan.py,
# a tensorflow2 reimplementation. The official repo is in TF1.


# ==============================================================================
#                  Spatial Pyramid Attentive Pooling
#
#        Attempting to implement https://arxiv.org/pdf/1901.06322.pdf
# ==============================================================================
# norm='instance_norm'
# Norm = _get_norm_layer(norm)

def _attentive_fuse(a, b, network):
    dim = a.shape[-1]
    n = tf.concat([a, b], 3)
    n = keras.layers.Conv2D(
        dim, 2, strides=1, padding='same', use_bias=False)(n)
    n = Norm(n)
    if network == "g":
        n = tf.nn.relu(n)
    else:
        n = tf.nn.leaky_relu(n)
    a = keras.layers.multiply([a, n])
    b = keras.layers.multiply([b, n])
    return keras.layers.add([a, b])


def _dilated_conv(a, level, network):
    dim = a.shape[-1]
    a = keras.layers.Conv2D(
        dim, 1, strides=1, dilation_rate=level, padding='same', use_bias=False)(a)
    a = Norm(a)
    if network == "g":
        a = tf.nn.relu(a)
    else:
        a = tf.nn.leaky_relu(a)
    return a


def spap(h, iterations, network):
    x = h
    dim = x.shape[-1]
    levels = iterations - 1
    x = _dilated_conv(h, iterations, network)
    while levels > 0:
        x = _attentive_fuse(x, _dilated_conv(h, levels, network), network)
        levels = levels - 1

    x = keras.layers.add([x, h])
    return x


def spap_rev(h, iterations, network):
    x = h
    dim = x.shape[-1]
    x = _dilated_conv(h, 1, network)
    levels = 2
    while levels <= iterations:
        x = _attentive_fuse(x, _dilated_conv(h, levels, network), network)
        levels += 1

    x = keras.layers.add([x, h])
    return x
# ==============================================================================


def ResnetGenerator(input_shape=(512, 512, 4),
                    output_channels=4,
                    dim=64,
                    n_downsamplings=1,  # originally 2
                    norm='instance_norm'):

    # 0
    h = inputs = keras.Input(shape=input_shape)

    # 0.5 - Reducing from 512 to 256.
    h = tf.image.resize(
        h, (256, 256), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # 1
    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h = keras.layers.Conv2D(dim, 7, padding='valid', use_bias=False)(h)
    h = Norm(h)
    h = tf.nn.relu(h)

    # 2
    for _ in range(n_downsamplings):
        dim *= 2
        h = keras.layers.Conv2D(
                    dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm(h)
        h = tf.nn.relu(h)

    # 3 - replacing the residual block network with the SPAP network
    h = spap(h, 3, "g")
    h = Norm(h)
    h = tf.nn.relu(h)

    # 4
    for _ in range(n_downsamplings):
        dim //= 2
        new_size = h.shape[1]
        #h = keras.layers.Conv2DTranspose(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = tf.image.resize(h, (new_size * 2, new_size * 2),
                            tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        h = keras.layers.Conv2D(dim, 3, strides=1, padding='same')(h)
        h = Norm(h)
        h = tf.nn.relu(h)

    # 4.5 - reupscaling
    h = tf.image.resize(
            h, (512, 512), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    h = keras.layers.Conv2D(dim, 2, strides=1, padding='same')(h)
    h = Norm(h)
    h = tf.nn.relu(h)

    # 5
    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h = keras.layers.Conv2D(output_channels, 7, padding='valid')(h)
    h = tf.tanh(h)

    return keras.Model(inputs=inputs, outputs=h)


def ConvDiscriminator(input_shape=(512, 512, 4),
                      dim=64,
                      n_downsamplings=4,
                      norm='instance_norm'):
    dim_ = dim
    # Norm = _get_norm_layer(norm)

    # 0
    h = inputs = keras.Input(shape=input_shape)

    # 1
    h = keras.layers.Conv2D(dim, 4, strides=2, padding='same')(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)

    for _ in range(n_downsamplings - 1):
        dim = min(dim * 2, dim_ * 8)
        h = keras.layers.Conv2D(
            dim, 4, strides=2, padding='same', use_bias=False)(h)
        h = Norm(h)
        h = tf.nn.leaky_relu(h, alpha=0.2)

    # 1.5

    h = spap(h,2, "d")
    h = Norm(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)

    # 2
    dim = min(dim * 2, dim_ * 8)
    h = keras.layers.Conv2D(
        dim, 4, strides=1, padding='same', use_bias=False)(h)
    h = Norm(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)

    # 3
    h = keras.layers.Conv2D(1, 4, strides=1, padding='same')(h)

    return keras.Model(inputs=inputs, outputs=h)


# ==============================================================================
# =                          learning rate scheduler                           =
# ==============================================================================

class LinearDecay(keras.optimizers.schedules.LearningRateSchedule):
    # if `step` < `step_decay`: use fixed learning rate
    # else: linearly decay the learning rate to zero

    def __init__(self, initial_learning_rate, total_steps, step_decay):
        super(LinearDecay, self).__init__()
        self._initial_learning_rate = initial_learning_rate
        self._steps = total_steps
        self._step_decay = step_decay
        self.current_learning_rate = tf.Variable(
            initial_value=initial_learning_rate, trainable=False, dtype=tf.float32)

    def __call__(self, step):
        self.current_learning_rate.assign(tf.cond(
            step >= self._step_decay,
            true_fn=lambda: self._initial_learning_rate *
            (1 - 1 / (self._steps - self._step_decay) * (step - self._step_decay)),
            false_fn=lambda: self._initial_learning_rate
        ))
        return self.current_learning_rate
