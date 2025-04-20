import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 

def LocalOptimizer(optimizee, optimizer, params, num_iter):

    params_shape = tf.shape(params)

    var = tf.Variable(
        initial_value=optimizee.initializer((params_shape[0], *optimizee.var_shape)), 
        trainable=True
    )


    sequence = []
    
    for i in range(num_iter):
        with tf.GradientTape() as tape:
            tape.watch(var)
            func_val = optimizee(var, params)
    
        grad = tape.gradient(func_val, var)

        optimizer.apply_gradients(zip([grad],[var]))
        sequence.append(optimizee(var, params))

    return tf.transpose(sequence, perm=[1,0,2])

class RIMBlock(layers.Layer):
    def __init__(self, features, **kwargs):
        super(RIMBlock, self).__init__(**kwargs)
        self.features = features
        self.lstm1cell = layers.LSTMCell(self.features, activation='tanh')
        self.lstm2cell = layers.LSTMCell(self.features, activation='tanh')
        self.dense = layers.Dense(1)
        
    def call(self, gradient, hidden_states, training=False):
        
        state1_h,state1_c, state2_h, state2_c = hidden_states

        dx,(state1_h,state1_c) = self.lstm1cell(gradient,[state1_h,state1_c], training=training)
        dx,(state2_h,state2_c) = self.lstm2cell(dx,[state2_h,state2_c], training=training)
        dx = self.dense(dx)
        
        return dx,state1_h,state1_c, state2_h, state2_c

class GlobalOptimizer(keras.Model):
    def __init__(self, optimizee, features=20, **kwargs):
        super(GlobalOptimizer, self).__init__(**kwargs)
        self.features = features
        self.optimizee = optimizee

    def compile(self, optimizer):
        super(GlobalOptimizer, self).compile()
        self.optimizer = optimizer
        
    def build(self, input_dim):
        self.rim_step = RIMBlock(self.features)

        
    def call(self, params, training=False, num_iters=100):
        
        state1_h = tf.zeros(shape=(1,self.features))
        state2_h = tf.zeros(shape=(1,self.features))
        state1_c = tf.zeros(shape=(1,self.features))
        state2_c = tf.zeros(shape=(1,self.features))

        params_shape = tf.shape(params)
        output_shape = (params_shape[0], *self.optimizee.var_shape)
        output = self.optimizee.initializer(output_shape)

        sequence = []
        for i in range(num_iters):
            gradient = tf.stop_gradient(self.optimizee.gradient(output, params))
            gradient = tf.reshape(gradient, (-1,1))
            output = tf.reshape(output, (-1,1))
            dx,state1_h,state1_c,state2_h,state2_c = self.rim_step(
                    gradient, 
                    [state1_h,state1_c,state2_h,state2_c], 
                    training=training
                )
            output += dx
            output = tf.reshape(output, output_shape)
            func_val = self.optimizee(output, params)
            sequence.append(func_val)

        sequence = tf.transpose(sequence, perm=[1,0,2])

        if training:
            return sequence
        return sequence,output

    @tf.function
    def train_step(self,batch,num_iters):
        with tf.GradientTape() as tape:
            y_pred = self(batch, training=True, num_iters=num_iters)
            train_loss = tf.reduce_mean(y_pred)
    
        gradients = tape.gradient(train_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
    
        return train_loss

    @tf.function
    def test_step(self,batch,num_iters):

        y_pred,var_pred = self(batch, training=False, num_iters=num_iters)
        val_loss = tf.reduce_mean(y_pred)
    
        return val_loss,var_pred

