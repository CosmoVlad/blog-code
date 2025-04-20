# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import scipy
import matplotlib.pyplot as plt


import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

from tqdm.notebook import tqdm
import time as tm


# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# tf.config.run_functions_eagerly(False)


MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rcdefaults()

# plt.rc('font',**{'family':'serif','serif':['Times']})
# plt.rc('text', usetex=True)

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title


# %matplotlib inline

# %% [markdown]
# ### Generate data for a one-parameter line $y=kx$

# %%
# our model for data (with one parameter k)

def f1(x,k):
    
    return k*x


# %%
# generate data 

rng = np.random.default_rng(seed=13)   # a random seed to have the same data every time we run this code cell

k0 = 2  # a "true" value for k
x_upper = 4

n_pts = 10
sigma = 1   # sigma for a Gaussian noise

x_data = x_upper*rng.random(n_pts)   # x values for the "measurement"
y_true = f1(x_data,k0)   # true y values
noise = sigma*rng.normal(scale=sigma, size=n_pts)
y_data = y_true + noise   # the data = true values + the noise
xx = np.linspace(0,4,500)
yy = f1(xx, k0)

# let us record the data for future use below

data_one_par = [x_data.copy(), y_data.copy(), k0, sigma]

fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))

ax.plot(xx,yy, c='darkorange', lw=2, label='truth')
ax.errorbar(x_data,y_data, yerr=sigma, fmt='b.', ms=10, label='data/measurement')

ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

ax.legend()

# %% [markdown]
# ### Find $k$ by minimizing the deviations
#
# We are to minimize
#
# $$
# \chi^2\equiv\frac 1N\sum\limits_{i=1}^N\left[y_i - k x_i\right]^2\,,
# $$
# with its gradient w.r.t. being
#
# $$
# \frac{\partial\chi^2}{\partial k} = -\frac 2N\sum\limits_{i=1}^N x_i\left[y_i - k x_i\right] = 0 \qquad \Rightarrow \qquad k = \frac{\sum y_i x_i}{\sum x_i^2}\,.
# $$

# %%
k_opt = np.sum(y_data*x_data)/np.sum(x_data**2)

print('Optimal value of k: {:.5f}'.format(k_opt))

# %%
## Sanity check using a built-in scipy function

from scipy.optimize import curve_fit

coefs,pcov = curve_fit(f1, x_data, y_data)   # pcov 

# sigmas = np.full_like(y_data, sigma)
# coefs,pcov = curve_fit(f, x_data, y_data, sigma=sigmas)
k_sp = coefs[0]

print('Scipy value of k: {:.5f}'.format(k_sp))

# %%
## Gradient descent from scratch

num_iters = 30
lr = 0.01
k_calc = -5 + 10*rng.random()

with tqdm(range(num_iters)) as tepoch:
    for i in tepoch:

        tepoch.set_description(f"Iteration {i+1}/{num_iters}")
        k_calc += lr*np.sum(x_data*(y_data - k_calc*x_data))
        tepoch.set_postfix_str("k={:.4f}".format(k_calc))
        tm.sleep(0.05)

# %% [markdown]
# ### Find $k$ with a RIM

# %%
dataset_size = 1024*10
batch_size = 64

eps = 0.1

x_mock_data = x_upper*rng.random(size=(dataset_size,n_pts))   # x values for the "measurement"
phi_mock = -(np.pi/2-eps) + 2*(np.pi/2-eps)*rng.random(size=(dataset_size,1))
k_mock = np.tan(phi_mock)
y_mock_true = f1(x_mock_data,k_mock)   # true y values
mock_noise = sigma*rng.normal(scale=sigma, size=(dataset_size,n_pts))
y_mock_data = y_mock_true + mock_noise   # the data = true values + the noise

print(x_mock_data.shape, y_mock_data.shape)

data = tf.concat((x_mock_data[...,np.newaxis], y_mock_data[...,np.newaxis]), axis=-1)
data = tf.cast(data, dtype=tf.float32)

print(data.shape)

dataset = tf.data.Dataset.from_tensor_slices(data)
train_dataset,val_dataset = tf.keras.utils.split_dataset(dataset, left_size=1024*8, shuffle=True)


# # Prepare the training dataset.
train_dataset = train_dataset.shuffle(buffer_size=batch_size).batch(batch_size)

# # Prepare the validation dataset.
val_dataset = val_dataset.shuffle(buffer_size=batch_size).batch(batch_size)



# %%
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

    def call(self, var, params):

        X,Y = tf.unstack(params, axis=-1)
        
        return tf.reduce_mean((Y-var*X)**2, axis=-1, keepdims=True) 

        
    def gradient(self, var, params):

        with tf.GradientTape() as optimizee_tape:
            optimizee_tape.watch(var)
            func_val = self(var, params)
        
        return optimizee_tape.gradient(func_val, var)

def LocalOptimizer(optimizee, optimizer, params, num_iter):

    params_shape = tf.shape(params)

    var = tf.Variable(
        initial_value=optimizee.initializer((params_shape[0],1)), 
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


# %%


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
        output = self.optimizee.initializer((params_shape[0],1))

        sequence = []
        for i in range(num_iters):
            gradient = tf.stop_gradient(self.optimizee.gradient(output, params))
            dx,state1_h,state1_c,state2_h,state2_c = self.rim_step(gradient, [state1_h,state1_c,state2_h,state2_c], training=training)
            output += dx
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




# %%
num_epochs = 10

model = GlobalOptimizer(optimizee=LinearFunction(SlopeInitializer(eps=eps)))
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.1))


for epoch in range(num_epochs):
    
    train_loss = 0.
    val_loss = 0.

    tepoch = tqdm(
            train_dataset,
            desc=f"Epoch {epoch+1}/{num_epochs}"
    )
        
    for batch_idx,data_batch in enumerate(tepoch):

        loss = model.train_step(data_batch, num_iters=100)
        train_loss += loss.numpy()
        tepoch.set_postfix_str("batch={:d}, train loss={:.4f}".format(batch_idx+1, train_loss/(batch_idx+1)))

    for batch_idx,val_batch in enumerate(val_dataset):

        if batch_idx == 4:
            break

        loss,var_pred = model.test_step(val_batch, num_iters=20)
        val_loss += loss.numpy()
    val_loss /= batch_idx + 1
    print("val loss={:.4f}".format(val_loss))
    
    



# %%
test_steps = 30

# compute LSTM losses on test batches and record the same batches to test the LocalOptimizer

losses = []
params_test = []

for batch in val_dataset.take(4):
    loss_on_batch,_ = model(batch, training=False, num_iters=test_steps)
    losses.append(loss_on_batch)
    params_test.append(batch)

losses = tf.concat(losses, axis=0).numpy()
params_test = tf.concat(params_test, axis=0)



test_losses = {}


optimizers = ['nag','adam', 'rmsprop', 'sgd']
optimizee = LinearFunction(
    SlopeInitializer(eps=eps)
)
lrates = np.logspace(-0.5,-2, 10)

for optimizer in optimizers:
    
    print(f'Optimizer: {optimizer}\n')

    trial_loss = []

    for lr in lrates:

        if optimizer == 'nag':
            model_optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=0.8, nesterov=True)
        elif optimizer == 'adam':
            model_optimizer = keras.optimizers.Adam(learning_rate=lr)
        elif optimizer == 'rmsprop':
            model_optimizer = keras.optimizers.RMSprop(learning_rate=lr)
        elif optimizer == 'sgd':
            model_optimizer = keras.optimizers.SGD(learning_rate=lr)

        trial_loss.append(
            LocalOptimizer(
                optimizee, model_optimizer, params_test, test_steps
            ).numpy()
        )
    trial_median = np.percentile(trial_loss, q=50, axis=1)
    best_idx = np.argmin(trial_median[:,-1,0]) 
    print(best_idx)

    test_losses[optimizer] = trial_loss[best_idx]



# %%
fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,5))

for optimizer,color in zip(optimizers,['violet','red','blue','green']):
    
    low,median,high = np.percentile(test_losses[optimizer], q=[16,50,84], axis=0)
    
    xx = 1 + np.arange(low.shape[0])
    ax.semilogy(xx, median.flatten(), color=color, lw=1, ls='dashed',label=optimizer)
    ax.fill_between(xx, low.flatten(), high.flatten(), color=color, alpha=0.3)

low,median,high = np.percentile(losses, q=[16,50,84], axis=0)

xx = 1 + np.arange(low.shape[0])
ax.semilogy(xx, median.flatten(), color='darkorange', lw=2, label='LSTM')
ax.fill_between(xx, low.flatten(), high.flatten(), color='darkorange', alpha=0.3)

ax.legend(loc='upper right')
ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

ax.set_ylim(1e-1, 2e+2)

ax.set_xlabel('step')
ax.set_ylabel('loss')

fig.tight_layout()
#fig.savefig('linear_optimizees.png')

# %%
data_original = tf.concat((x_data[...,np.newaxis], y_data[...,np.newaxis]), axis=-1)
data_original = tf.cast(data_original[tf.newaxis,...], dtype=tf.float32)

_,var_pred = model(data_original, training=False, num_iters=num_iters)

print(var_pred.numpy())

# %%
