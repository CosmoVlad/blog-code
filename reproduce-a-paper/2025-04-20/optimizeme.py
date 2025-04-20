import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 

class SlopeInitializer(keras.initializers.Initializer):
    def __init__(self, eps, **kwargs):
        super(SlopeInitializer, self).__init__(**kwargs)
        self.eps = eps

    def __call__(self, shape, dtype=tf.float32):
        output = tf.random.uniform(
            shape=shape,
            minval=-(np.pi/2-self.eps), maxval=(np.pi/2-self.eps),
            dtype=dtype
        )
        return tf.math.tan(output)
    

class LinearFunction(layers.Layer):
    def __init__(self, initializer, **kwargs):
        super(LinearFunction, self).__init__(**kwargs)
        self.initializer = initializer
        self.var_shape = (1,)

    def call(self, var, params):

        X,Y = tf.unstack(params, axis=-1)
        
        return tf.reduce_mean((Y-var*X)**2, axis=-1, keepdims=True) 

        
    def gradient(self, var, params):

        with tf.GradientTape() as optimizee_tape:
            optimizee_tape.watch(var)
            func_val = self(var, params)
        
        return optimizee_tape.gradient(func_val, var)

class QuadraticsFunction(layers.Layer):
    def __init__(self, initializer, **kwargs):
        super(QuadraticsFunction, self).__init__(**kwargs)
        self.initializer = initializer
        self.var_shape = (10,1)

    def call(self, var, params):

        W,Y = tf.split(params, [10,1], axis=-1)
        aux = tf.matmul(W, var) - Y
        aux = tf.matmul(tf.transpose(aux, perm=[0,2,1]), aux)
        
        return tf.squeeze(aux, axis=-1)

        
    def gradient(self, var, params):

        with tf.GradientTape() as optimizee_tape:
            optimizee_tape.watch(var)
            func_val = self(var, params)
        
        return optimizee_tape.gradient(func_val, var)



        
