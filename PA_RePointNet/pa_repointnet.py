import tensorflow as tf
from tensorflow import keras
from keras import layers

tf.random.set_seed(123)

class PAM(keras.Model):
    def __init__(self, feature_dim, name='PAM'):
        super(PAM, self).__init__()

        # model para
        self.feature_dim = feature_dim

        # model layers
        self.current_input = layers.Input(shape=(feature_dim,))
        self.gamma = tf.Variable(tf.ones(1), name='pam_gamma')
        self.conv = layers.Conv1D(filters=1, kernel_size=1, strides=1, padding='same', name='pam-conv1d',
                          data_format='channels_last', use_bias=True, bias_initializer=tf.zeros_initializer())
        self.bn = layers.BatchNormalization(name='pam_bn')
        self.softmax = layers.Softmax(name=name)

    def call(self, current):

        # a = self.current_input(current)
        a = tf.expand_dims(current, axis=2)  # Add one dimension [bs, feature_dim] --> [bs, feature_dim, c]
        b = self.conv(a)
        b = self.bn(b)
        b_trans = tf.transpose(b, perm=[0, 2, 1])  # [bs, feature_dim, c] --> [bs, c, feature_dim]
        b_mutmul = tf.matmul(b, b_trans)  # [bs, feature_dim, c] @ [bs, c, feature_dim] = [bs, feature_dim, feature_dim]
        b_mutmul = self.softmax(b_mutmul)  # The normalized weights are obtained through softmax

        c_mutmul = tf.matmul(b_trans, b_mutmul)  # [bs, c, feature_dim] @ [bs, feature_dim, feature_dim] = [bs, c, feature_dim]
        c_mutmul_trans = tf.transpose(c_mutmul, perm=[0, 2, 1])  # [bs, c, feature_dim] --> [bs, feature_dim, c]
        c_mutmul_trans = c_mutmul_trans * self.gamma

        output = layers.add([c_mutmul_trans, a])  # Input and output overlay

        output = tf.reshape(output, shape=[-1, self.feature_dim])

        return output

class PA_RePointNet(keras.Model):
    def __init__(self, current_dim, philsm_dim, repointnet_pam_flag):
        super(PA_RePointNet, self).__init__()

        # model para
        self.feature_dim = current_dim
        self.philsm_dim = philsm_dim
        self.repointnet_pam_flag = repointnet_pam_flag

        # model layers
        self.pam_repointnet = PAM(self.feature_dim, name='encoder_pam')

        self.mlp1 = keras.Sequential()
        for i in range(2):
            self.mlp1.add(layers.Dense(units=1024, name='re-pointnet_mlp1_dense{}'.format(i)))
            self.mlp1.add(layers.BatchNormalization())
            self.mlp1.add(layers.ReLU())

        self.mlp2 = keras.Sequential()
        self.mlp2.add(layers.Dense(units=1024, name='re-pointnet_mlp2_dense1'))
        self.mlp2.add(layers.BatchNormalization())
        self.mlp2.add(layers.ReLU())
        self.mlp2.add(layers.Dense(units=128, name='re-pointnet_mlp2_dense2'))
        self.mlp2.add(layers.BatchNormalization())
        self.mlp2.add(layers.ReLU())
        self.mlp2.add(layers.Dense(units=64, name='re-pointnet_mlp2_dense3'))
        self.mlp2.add(layers.BatchNormalization())
        self.mlp2.add(layers.ReLU())
        self.mlp2.add(layers.Dense(units=64, name='re-pointnet_mlp2_dense4'))
        self.mlp2.add(layers.BatchNormalization())
        self.mlp2.add(layers.ReLU())
        self.mlp2.add(layers.Dense(units=64, name='re-pointnet_mlp2_dense5'))
        self.mlp2.add(layers.BatchNormalization())
        self.mlp2.add(layers.ReLU())

        self.mlp3 = keras.Sequential()
        self.mlp3.add(layers.Dense(units=self.philsm_dim, name='re-pointnet_mlp3_dense1'))
        # self.mlp3.add(layers.Softmax())
        self.mlp3.add(layers.ReLU())

    def call(self, current):

        if self.repointnet_pam_flag == True:
            h = self.pam_repointnet(current)
            h = self.mlp1(h)
        else:
            h = self.mlp1(current)

        h = self.mlp2(h)
        philsm = self.mlp3(h)

        return philsm