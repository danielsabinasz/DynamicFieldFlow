import logging
import dff

import tensorflow as tf
from dff.simulation.weight_patterns import compute_kernel_gauss_tensor

def convolve(x, w):
    if dff.config.use_fft:
        return convolve_fft(x, w)
    else:
        return convolve_classic(x, w)

#@tf.function(input_signature=(
#    tf.TensorSpec(shape=(None,), dtype=tf.float32),
#    tf.TensorSpec(shape=(None,), dtype=tf.float32)
#))
@tf.function
def convolve_classic(x, w):
    """Performs a convolution

    :param Tensor x: tensor to convolve
    :param Tensor w: convolution kernel
    :return Tensor: convolution result
    """
    logging.debug(f"trace convolution x={x} w={w}")

    x_shape = tf.shape(x)
    w_shape = tf.shape(w)

    rank = len(x.shape)

    if rank == 0:
        # TODO: check why this is traced
        return tf.math.multiply(x, w)
    elif rank == 1:
        x = tf.reshape(x, [1, x_shape[0], 1])
        w = tf.reshape(w, [w_shape[0], 1, 1])
        conv = tf.nn.conv1d(x, w, stride=1, padding="SAME")
        conv = tf.reshape(conv, [x_shape[0]])
    elif rank == 2:
        x = tf.reshape(x, [1, x_shape[0], x_shape[1], 1])
        w = tf.reshape(w, [w_shape[0], w_shape[1], 1, 1])
        conv = tf.nn.conv2d(x, w, strides=[1,1], padding="SAME")
        conv = tf.reshape(conv, [x_shape[0], x_shape[1]])
    elif rank == 3:
        x = tf.reshape(x, [1, x_shape[0], x_shape[1], x_shape[2], 1])
        w = tf.reshape(w, [w_shape[0], w_shape[1], w_shape[2], 1, 1])
        conv = tf.nn.conv3d(x, w, strides=[1,1,1,1,1], padding="SAME")
        conv = tf.reshape(conv, [x_shape[0], x_shape[1], x_shape[2]])
    else:
        raise Exception("Unsupported rank " + str(rank))

    """import matplotlib.pyplot as plt
    plt.imshow(x[0,:,:,0])
    plt.show()
    plt.imshow(w.numpy()[:,:,0,0,0])
    plt.colorbar()
    plt.show()
    plt.imshow(conv[:,:,0])
    plt.show()"""

    return conv


@tf.function
def convolve_fft(x, w):
    print(x.shape, w.shape, "CONV")
    """Performs a convolution using the fast fourier transform

    :param Tensor x: The tensor to convolve
    :param Tensor w: The convolution kernel
    :return Tensor: The convolution result
    """
    logging.debug(f"trace convolution_fft x={x} w={w}")

    ndim = len(x.shape)
    if ndim == 2:
        fft_length1 = tf.shape(x)[0]
        fft_length2 = tf.shape(x)[1]
        x_fft = tf.signal.rfft2d(x, fft_length=[fft_length1, fft_length2])
        w_fft = tf.signal.rfft2d(w, fft_length=[fft_length1, fft_length2])
        conv = tf.signal.irfft2d(x_fft * w_fft, [fft_length1, fft_length2])
        conv = tf.roll(conv, shift=[int(conv.shape[0]/2), int(conv.shape[1]/2)], axis=[0, 1])
    elif ndim == 3:
        fft_length1 = tf.shape(x)[0]
        fft_length2 = tf.shape(x)[1]
        fft_length3 = tf.shape(x)[2]
        x_fft = tf.signal.rfft3d(x, fft_length=[fft_length1, fft_length2, fft_length3])
        w_fft = tf.signal.rfft3d(w, fft_length=[fft_length1, fft_length2, fft_length3])
        conv = tf.signal.irfft3d(x_fft * w_fft, [fft_length1, fft_length2, fft_length3])
        conv = tf.roll(conv, shift=[int(conv.shape[0]/2), int(conv.shape[1]/2), int(conv.shape[2]/2)], axis=[0, 1, 2])
    elif ndim == 1:
        fft_length = tf.shape(x)[0]
        x_fft = tf.signal.rfft(x, fft_length=fft_length)
        w_fft = tf.signal.rfft(w, fft_length=fft_length)
        conv = tf.signal.irfft(x_fft * w_fft, fft_length)
        conv = tf.roll(conv, shift=int(conv.shape[0]/2), axis=0)
    return conv

@tf.function
def convolve_fft_args(args):
    x, w = args
    return convolve_fft(x, w)

@tf.function(jit_compile=True)
def convolve_fft_stacked_old(x_stack, w_stack):
    return tf.map_fn(convolve_fft_args, (x_stack, w_stack), dtype=tf.float32)

@tf.function
def convolve_fft_stacked_new(x_stack, w_stack):
    x_slices = tf.split(x_stack, num_or_size_splits=x_stack.shape[0], axis=0)
    w_slices = tf.split(w_stack, num_or_size_splits=w_stack.shape[0], axis=0)

    x_shape = x_stack[0].shape
    w_shape = w_stack[0].shape

    conv_slices = [
        tf.reshape(
            tf.nn.conv2d(input=tf.reshape(input_slice, [1, x_shape[0], x_shape[1], 1]),
                     filters=tf.reshape(filter_slice, [w_shape[0], w_shape[1], 1, 1]), strides=[1, 1], padding='SAME')
        , [x_shape[0], x_shape[1]])
        for input_slice, filter_slice in zip(x_slices, w_slices)
    ]
    return tf.stack(conv_slices, axis=0)



@tf.function
def convolve_classic_stacked(X, w):
    """Performs a convolution

    :param Tensor x: tensor to convolve
    :param Tensor w: convolution kernel
    :return Tensor: convolution result
    """

    x_shape = tf.shape(X[0])
    w_shape = tf.shape(w)

    rank = len(X[0].shape)

    #if rank == 1:
    #    x = tf.reshape(x, [1, x_shape[0], 1])
    #    w = tf.reshape(w, [w_shape[0], 1, 1])
    #    conv = tf.nn.conv1d(x, w, stride=1, padding="SAME")
    #    conv = tf.reshape(conv, [x_shape[0]])
    if rank == 2:
        x = tf.reshape(X, [X.shape[0], x_shape[0], x_shape[1], 1])
        w = tf.reshape(w, [w_shape[0], w_shape[1], 1, 1])
        conv = tf.nn.conv2d(x, w, strides=[1,1], padding="SAME")
        conv = tf.reshape(conv, [X.shape[0], x_shape[0], x_shape[1]])
    elif rank == 3:
        x = tf.reshape(X, [X.shape[0], x_shape[0], x_shape[1], x_shape[2], 1])
        w = tf.reshape(w, [w_shape[0], w_shape[1], w_shape[2], 1, 1])
        conv = tf.nn.conv3d(x, w, strides=[1,1,1,1,1], padding="SAME")
        conv = tf.reshape(conv, [X.shape[0], x_shape[0], x_shape[1], x_shape[2]])
    #else:
    #    raise Exception("Unsupported rank " + str(rank))

    """import matplotlib.pyplot as plt
    plt.imshow(x[0,:,:,0])
    plt.show()
    plt.imshow(w.numpy()[:,:,0,0,0])
    plt.colorbar()
    plt.show()
    plt.imshow(conv[:,:,0])
    plt.show()"""

    return conv
