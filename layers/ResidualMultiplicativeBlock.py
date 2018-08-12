import tensorflow as tf
from layers.MultiplicativeUnit import MultiplicativeUnit


class ResidualMultiplicativeBlock():
    """Initialize the residual multiplicative block without mask.
    Args:
       layer_name: layer names for different residual multiplicative block.
       filter_size: int tuple of the height and width of the filter.
       num_hidden: number of units in output tensor.
    """
    def __init__(self, layer_name, num_hidden, filter_size):
        self.layer_name = layer_name
        self.num_features = num_hidden
        self.filter_size = filter_size

    def __call__(self, h, reuse=False):
        with tf.variable_scope(self.layer_name, reuse=reuse):
            h1 = tf.layers.conv2d(
                h, self.num_features, 1, padding='same', activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(), name='h1')
            h2 = MultiplicativeUnit('multiplicative_unit_1', self.num_features, self.filter_size)(h1, reuse=reuse)
            h3 = MultiplicativeUnit('multiplicative_unit_2', self.num_features, self.filter_size)(h2, reuse=reuse)
            h4 = tf.layers.conv2d(
                h3, 2 * self.num_features, 1, padding='same', activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(), name='h4')
            rmb = tf.add(h, h4)
            return rmb
