import tensorflow as tf

class Params:

	eps = 0.1


	DTYPE = tf.float32

	TRAIN_ITERS = 100
	VAL_ITERS = 20
	TEST_ITERS = 30
	OPTIMIZER ='Adam'
	OPTIMIZER_PARAMS = {'learning_rate': 0.1}
	BATCH_SIZE = 64
	NUM_EPOCHS = 20
	STEPS_PER_EPOCH = 100
	
	DATASET_SIZE = 1024*10
	NUM_PTS = 10
	X_UPPER = 4.
	SIGMA = 1.
	SEED = 10
	