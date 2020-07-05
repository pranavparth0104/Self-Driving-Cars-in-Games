
import numpy as np
from ppap import ppap
WIDTH = 160
HEIGHT = 120
LR = 1e-3
EPOCHS = 2
MODEL_NAME = 'pynfs-car-fast-{}-{}-{}-epochs-300K-data.model'.format(LR, 'alexnetv2',EPOCHS)

model = ppap(WIDTH, HEIGHT, LR)

hm_data = 2
for i in range(EPOCHS):
    for i in range(1,hm_data+1):
        train_data = np.load('training_data_balanced.npy'.format(i))

        train = train_data[:-100]
        test = train_data[-100:]

        X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
        Y = [i[1] for i in train]

        test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
        test_y = [i[1] for i in test]

        model.fit({'input': X}, {'targets': Y}, n_epoch=1, validation_set=({'input': test_x}, {'targets': test_y}),
            snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

        model.save(MODEL_NAME)
