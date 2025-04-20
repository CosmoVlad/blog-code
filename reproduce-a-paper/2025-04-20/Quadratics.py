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
from keras import layers

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

# %%
from models import LocalOptimizer, GlobalOptimizer
from optimizeme import QuadraticsFunction
from data_generators import QuadraticsGenerator
from config import Params




# %% [markdown]
# ### Compare a neural network (LSTM) optimizer with conventional optimizers

# %%
test_steps = Params.TEST_ITERS
num_test_batches = 4

## define an initializer and an optimizee

initializer = keras.initializers.RandomNormal(mean=0., stddev=1.)
optimizee = QuadraticsFunction(
    initializer=initializer
)

# ## load a trained GlobalOptimizer model

model = GlobalOptimizer(
    optimizee=optimizee
)

_ = model(
    keras.Input(shape=(10, 11))
)

model.load_weights('./quadratics_model.weights.h5')

## initialize a data generator for test data

test_generator = QuadraticsGenerator(initializer, Params.BATCH_SIZE, Params.STEPS_PER_EPOCH)

# %%
# data_original = tf.concat((x_data[...,np.newaxis], y_data[...,np.newaxis]), axis=-1)
# data_original = tf.cast(data_original[tf.newaxis,...], dtype=tf.float32)

# _,var_pred = model(data_original, training=False, num_iters=test_steps)

# print('LSTM optimizer after {:d} iterations value of k: {:.5f}'.format(test_steps, var_pred.numpy().flatten()[0]))

# %%

# compute LSTM losses on test batches and record the same batches to test the LocalOptimizer

losses = []
params_test = []

for idx,batch in enumerate(test_generator):
    if idx == num_test_batches:
        break
    loss_on_batch,_ = model(batch, training=False, num_iters=test_steps)
    losses.append(loss_on_batch)
    params_test.append(batch)

losses = tf.concat(losses, axis=0).numpy()
params_test = tf.concat(params_test, axis=0)



test_losses = {}


optimizers = ['nag','adam', 'rmsprop', 'sgd']
lrates = np.logspace(-1.5,-3, 10)

for optimizer in optimizers:
    
    print(f'Optimizer: {optimizer}\n')

    trial_loss = []

    for lr in lrates:

        if optimizer == 'nag':
            model_optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
        else:
            model_optimizer = keras.optimizers.get(optimizer).from_config({'learning_rate':lr})

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
    mean = np.mean(test_losses[optimizer], axis=0)
    
    xx = 1 + np.arange(low.shape[0])
    ax.semilogy(xx, median.flatten(), color=color, lw=1, ls='dashed',label=optimizer)
    #ax.semilogy(xx, mean.flatten(), color=color, lw=1, ls='dashed',label=optimizer)
    #ax.fill_between(xx, low.flatten(), high.flatten(), color=color, alpha=0.3)

low,median,high = np.percentile(losses, q=[16,50,84], axis=0)
mean = np.mean(losses, axis=0)

xx = 1 + np.arange(low.shape[0])
#ax.semilogy(xx, mean.flatten(), color='darkorange', lw=2, label='LSTM')
ax.semilogy(xx, median.flatten(), color='darkorange', lw=2, label='LSTM')
#ax.fill_between(xx, low.flatten(), high.flatten(), color='darkorange', alpha=0.3)

ax.legend(loc='upper right')
ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

ax.set_ylim(2e-2, 2e+2)

ax.set_xlabel('step')
ax.set_ylabel('loss')

fig.tight_layout()
#fig.savefig('linear_optimizees.png')

# %%
