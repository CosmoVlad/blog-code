import numpy as np
import tensorflow as tf
from tensorflow import keras

from config import Params


class LinearGenerator(keras.utils.Sequence):

    def __init__(self, initializer, batch_size, steps_per_epoch, seed=False, shuffle=True, **kwargs):
        super(LinearGenerator, self).__init__(**kwargs)
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.initializer = initializer
        self.shuffle = shuffle

        if seed:
            self.rng = np.random.default_rng(seed=Params.SEED)
        else:
            self.rng = np.random.default_rng()

         # x values for the "measurement"
        x_mock_data = Params.X_UPPER*self.rng.random(
                size=(Params.DATASET_SIZE,Params.NUM_PTS),
            )  
            
        k_mock_data = self.initializer(
                shape=(Params.DATASET_SIZE,1), 
                dtype=Params.DTYPE
            )
        y_mock_true = LinearGenerator.f1(x_mock_data,k_mock_data)   # true y values

        mock_noise = self.rng.normal(
                size=(Params.DATASET_SIZE,Params.NUM_PTS),
                scale = 1.
            )
        y_mock_data = y_mock_true + mock_noise   # the data = true values + the noise

        self.data = np.concatenate((x_mock_data[...,np.newaxis], y_mock_data[...,np.newaxis]), axis=-1)

        self.indices = np.arange(Params.DATASET_SIZE)
        if self.shuffle:
            self.rng.shuffle(self.indices)

    @staticmethod
    def f1(x,k):
        return k*x

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, index):
        
        sys_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        return tf.cast(self.data[sys_indices], dtype=Params.DTYPE)

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)

class QuadraticsGenerator(keras.utils.Sequence):

    def __init__(self, initializer, batch_size, steps_per_epoch, seed=False, shuffle=True, **kwargs):
        super(QuadraticsGenerator, self).__init__(**kwargs)
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.initializer = initializer
        self.shuffle = shuffle

        if seed:
            self.rng = np.random.default_rng(seed=Params.SEED)
        else:
            self.rng = np.random.default_rng()

            
        W_data = self.initializer(
                shape=(Params.DATASET_SIZE,10,10), 
                dtype=Params.DTYPE
            )
        y_data = self.initializer(
                shape=(Params.DATASET_SIZE,10,1), 
                dtype=Params.DTYPE
            )


        self.data = np.concatenate([W_data,y_data], axis=-1)

        self.indices = np.arange(Params.DATASET_SIZE)
        if self.shuffle:
            self.rng.shuffle(self.indices)



    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, index):
        
        sys_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        return tf.cast(self.data[sys_indices], dtype=Params.DTYPE)

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)






