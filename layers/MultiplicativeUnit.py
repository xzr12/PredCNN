import tensorflow as tf


class MultiplicativeUnit():
    """Initialize the multiplicative unit.
    Args:
       layer_name: layer names for different multiplicative units.
       filter_size: int tuple of the height and width of the filter.
       num_hidden: number of units in output tensor.
    """
    def __init__(self, layer_name, num_hidden, filter_size):
        self.layer_name = layer_name
        self.num_features = num_hidden
        self.filter_size = filter_size

    def __call__(self, h, reuse=False):
        with tf.variable_scope(self.layer_name, reuse=reuse):
            g1 = tf.layers.conv2d(
                h, self.num_features, self.filter_size, padding='same', activation=tf.sigmoid,
                kernel_initializer=tf.contrib.layers.xavier_initializer(), name='g1')
            g2 = tf.layers.conv2d(
                h, self.num_features, self.filter_size, padding='same', activation=tf.sigmoid,
                kernel_initializer=tf.contrib.layers.xavier_initializer(), name='g2')
            g3 = tf.layers.conv2d(
                h, self.num_features, self.filter_size, padding='same', activation=tf.sigmoid,
                kernel_initializer=tf.contrib.layers.xavier_initializer(), name='g3')
            u = tf.layers.conv2d(
                h, self.num_features, self.filter_size, padding='same', activation=tf.tanh,
                kernel_initializer=tf.contrib.layers.xavier_initializer(), name='u')
            g2_h = tf.multiply(g2, h)
            g3_u = tf.multiply(g3, u)
            mu = tf.multiply(g1, tf.tanh(g2_h + g3_u))
            return mu
