import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 

from optimizeme import LinearFunction, SlopeInitializer
from models import LocalOptimizer, GlobalOptimizer
from data_generators import LinearGenerator

from config import Params
from tqdm import tqdm


initializer = SlopeInitializer(eps=Params.eps)
optimizer = keras.optimizers.get(Params.OPTIMIZER).from_config(Params.OPTIMIZER_PARAMS)
train_generator = LinearGenerator(initializer, Params.BATCH_SIZE, Params.STEPS_PER_EPOCH)
val_generator = LinearGenerator(initializer, Params.BATCH_SIZE, Params.STEPS_PER_EPOCH)


if __name__ == '__main__':
    

    model = GlobalOptimizer(
            optimizee=LinearFunction(initializer=initializer)
        )
    model.compile(optimizer=optimizer)

    num_epochs = Params.NUM_EPOCHS


    for epoch in range(num_epochs):
        
        train_loss = 0.
        val_loss = 0.

        tepoch = tqdm(
                train_generator,
                desc=f"Epoch {epoch+1}/{num_epochs}"
        )
            
        for batch_idx,data_batch in enumerate(tepoch):

            if batch_idx == len(train_generator):
                break

            loss = model.train_step(data_batch, num_iters=Params.TRAIN_ITERS)
            train_loss += loss.numpy()
            tepoch.set_postfix_str("batch={:d}, train loss={:.4f}".format(batch_idx+1, train_loss/(batch_idx+1)))


        for batch_idx,val_batch in enumerate(val_generator):

            if batch_idx == len(val_generator):
                break

            loss,var_pred = model.test_step(val_batch, num_iters=Params.VAL_ITERS)
            val_loss += loss.numpy()
        val_loss /= batch_idx + 1
        print("val loss={:.4f}".format(val_loss))

        train_generator.on_epoch_end()
        val_generator.on_epoch_end()

    #model.save_weights('linear_model.weights.h5')


