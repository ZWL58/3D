import keras.optimizers
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.ops.math_ops import rsqrt, multiply, minimum
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

class CustomSchedule1(LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule1, self).__init__()
        self.d_model = tf.constant(d_model, dtype=tf.float32)
        self.warmup_steps = warmup_steps
    def get_config(self):
        return {"demodel":self.d_model, "warmup_steps":self.warmup_steps}
    def __call__(self, step):
        arg1 = rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)
        return multiply(rsqrt(self.d_model), minimum(arg1, arg2))

# test
sample_learning_rate = CustomSchedule1(d_model=256)
plt.plot(sample_learning_rate(tf.range(200000, dtype=tf.float32)))
plt.ylabel("Learning Rate")
plt.xlabel("Train Step")
plt.show()
# learning_rate = CustomSchedule(D_MODEL)

class CustomSchedule2(LearningRateSchedule):
    def __init__(self, epochs):
        super(CustomSchedule2, self).__init__()
        self.epochs = tf.constant(epochs, dtype=tf.float32)
    def get_config(self):
        return {"demodel":self.d_model}
    def __call__(self, step):
        lr_start = 0.001
        lr_end = 0.0001
        a = (lr_end-lr_start)/self.epochs
        b = lr_start

        return a * step + b

sample_learning_rate = CustomSchedule2(epochs=5000)
plt.plot(sample_learning_rate(tf.range(5000, dtype=tf.float32)))
plt.ylabel("Learning Rate")
plt.xlabel("Train Step")
plt.show()


# epochs = 5000
# learning_rate = CustomSchedule2(epochs)
# optimizer = keras.optimizers.Adam(learning_rate)
