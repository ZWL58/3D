import tensorflow as tf
from tensorflow import keras
from keras import layers

tf.random.set_seed(123)

class PAM(keras.Model):
    def __init__(self, feature_dim, name="PAM"):
        super(PAM, self).__init__()


        self.feature_dim = feature_dim


        self.gamma = tf.Variable(tf.ones(1), name="pam_gamma")
        self.conv = layers.Conv1D(filters=1, kernel_size=1, strides=1, padding="same", name="pam-conv1d",
                          data_format="channels_last", use_bias=True, bias_initializer=tf.zeros_initializer())
        self.bn = layers.BatchNormalization(name="pam_bn")
        self.softmax = layers.Softmax(name=name)

    def call(self, current):


        a = tf.expand_dims(current, axis=2)
        b = self.conv(a)
        b = self.bn(b)
        b_trans = tf.transpose(b, perm=[0, 2, 1])
        b_mutmul = tf.matmul(b, b_trans)
        b_mutmul = self.softmax(b_mutmul)

        c_mutmul = tf.matmul(b_trans, b_mutmul)
        c_mutmul_trans = tf.transpose(c_mutmul, perm=[0, 2, 1])
        c_mutmul_trans = c_mutmul_trans * self.gamma

        output = layers.add([c_mutmul_trans, a])

        output = tf.reshape(output, shape=[-1, self.feature_dim])

        return output

class Encoder(keras.Model):
    def __init__(self, feature_dim, label_dim, hidden_layer_num, hidden_layer_dim, latent_dim, encoder_pam_flag):
        super(Encoder, self).__init__()

        self.feature_dim = feature_dim
        self.label_dim = label_dim
        self.hidden_layer_num = hidden_layer_num
        self.hidden_layer_dim = hidden_layer_dim
        self.latent_dim = latent_dim
        self.encoder_pam_flag = encoder_pam_flag

        self.pam_encoder = PAM(self.feature_dim, name="encoder_pam")

        self.blocks = keras.Sequential()
        for i in range(self.hidden_layer_num):
            self.blocks.add(layers.Dense(units=self.hidden_layer_dim, name="encoder_dense{}".format(i)))
            self.blocks.add(layers.BatchNormalization())
            self.blocks.add(layers.ReLU())

        self.mu_output = layers.Dense(self.latent_dim)
        self.sigma_output = layers.Dense(self.latent_dim)
        self.y_avr_output = layers.Dense(self.latent_dim)

    def call(self, data):
        current = data[0]
        weld_para = data[1]

        if self.encoder_pam_flag == True:
            h1 = self.pam_encoder(current)
            h = tf.concat([h1, weld_para], axis=1)
        else:
            h = tf.concat([current, weld_para], axis=1)

        h = self.blocks(h)

        mu = self.mu_output(h)
        sigma = self.sigma_output(h)
        y_avr = self.y_avr_output(h)

        return mu, sigma, y_avr


class Decoder(keras.Model):
    def __init__(self, feature_dim, label_dim, hidden_layer_num, hidden_layer_dim, latent_dim, decoder_pam_flag):
        super(Decoder, self).__init__()

        self.feature_dim = feature_dim
        self.label_dim = label_dim
        self.hidden_layer_num = hidden_layer_num
        self.hidden_layer_dim = hidden_layer_dim
        self.latent_dim = latent_dim
        self.decoder_pam_flag = decoder_pam_flag

        self.pam_decoder = PAM(self.feature_dim, name="decoder_pam")

        self.blocks = keras.Sequential()
        for i in range(self.hidden_layer_num):
            self.blocks.add(layers.Dense(units=self.hidden_layer_dim, name="decoder_dense{}".format(i)))
            self.blocks.add(layers.BatchNormalization())
            self.blocks.add(layers.ReLU())

        self.x_ = layers.Dense(self.feature_dim)
        self.relu = layers.ReLU()

    def call(self, data):

        h = self.blocks(data)
        h = self.x_(h)
        if self.decoder_pam_flag == True:
            h = self.pam_decoder(h)
        x_ = self.relu(h)

        return x_

class PACVAE(tf.keras.Model):
    def __init__(self, feature_dim, label_dim, hidden_layer_num, hidden_layer_dim, latent_dim, encoder_pam_flag, decoder_pam_flag):
        super(PACVAE, self).__init__()

        self.feature_dim = feature_dim
        self.label_dim = label_dim
        self.hidden_layer_num = hidden_layer_num
        self.hidden_layer_dim = hidden_layer_dim
        self.latent_dim = latent_dim
        self.encoder_pam_flag = encoder_pam_flag
        self.decoder_pam_flag = decoder_pam_flag

        self.encoder = Encoder(feature_dim, label_dim, hidden_layer_num, hidden_layer_dim, latent_dim, encoder_pam_flag)
        self.decoder = Decoder(feature_dim, label_dim, hidden_layer_num, hidden_layer_dim, latent_dim, decoder_pam_flag)

    def call(self, data):
        current = data[:,:self.feature_dim]
        weld_para = data[:,self.feature_dim:self.feature_dim+self.label_dim]
        epsilon = data[:,self.feature_dim+self.label_dim:]

        mu, sigma, y_avr = self.encoder([current, weld_para])


        z = mu + keras.backend.exp(sigma / 2) * epsilon

        input = tf.concat([z, weld_para], axis=1)
        x_ = self.decoder(input)

        return mu, sigma, y_avr, x_

    def get_funcbased_model(self):
        data = tf.keras.layers.Input(shape=(1022,))
        current = data[:,:self.feature_dim]
        weld_para = data[:,self.feature_dim:self.feature_dim+self.label_dim]
        epsilon = data[:,self.feature_dim+self.label_dim:]
        mu, sigma, y_avr = self.encoder([current, weld_para])

        z = mu + keras.backend.exp(sigma / 2) * epsilon

        input = tf.concat([z, weld_para], axis=1)
        x_ = self.decoder(input)

        return keras.models.Model(inputs=data, outputs=(mu, sigma, y_avr, x_))

def compute_loss(x, mu, sigma, y_avr, x_):
    recon_loss = keras.backend.sum(keras.backend.binary_crossentropy(x, x_), axis=-1)
    kl_loss = - 0.5 * keras.backend.sum(1 + mu - keras.backend.square(mu - y_avr) - keras.backend.exp(sigma),
                                        axis=-1)
    abs_loss = tf.abs(x - x_)
    return recon_loss, kl_loss, abs_loss
