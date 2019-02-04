import cv2, argparse, sys
import numpy as np
import matplotlib.pyplot as plt
from fdream import Config

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=123456, help="Random seed for test images")
parser.add_argument('-e', '--epochs', default=2000, help="Number of epochs to run")
parser.add_argument('-b', '--batchsize', default=8, help="Batch size")
args = vars(parser.parse_args(sys.argv[1:]))


cfg = Config()

def convert_toimg(data):
    return cv2.cvtColor((data * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

train_data = np.load('train_data.npy').astype(np.float32) / 255
test_set = [train_data[idx] for idx in [2,5,6,7,41,76,137,180,5031,4858,4681]]
test_imgs = np.hstack([convert_toimg(test_set[idx]) for idx in range(len(test_set))])

from fdream.autoencoder import AutoEncoder
from keras.optimizers import Adam

ae = AutoEncoder()
ae.load(cfg.base_dir)
ae_model = ae.encoder_decoder()
ae_model.compile(optimizer=Adam(lr=0.0008), loss='mse')

np.random.seed(int(args['seed']))
rand_vecs = np.random.normal(0.0, 1.0, (10, AutoEncoder.PARAM_SIZE))


for epoch in range(122, int(args['epochs'])):
    print("Epoch: "+str(epoch))
    history = ae_model.fit(train_data, train_data, epochs=1, batch_size=int(args['batchsize']), shuffle=True)
    loss = history.history['loss'][-1]
    print("Loss: " + str(loss))

    ae.encoder.save(cfg.base_dir+'encoder.h5')
    ae.decoder.save(cfg.base_dir+'decoder.h5')

    ret  = ae_model.predict([test_set])
    ret_imgs = np.hstack([convert_toimg(ret[idx]) for idx in range(len(test_set))])
    ret_imgs = np.vstack((test_imgs, ret_imgs))
    cv2.imwrite(cfg.base_dir+'t_'+str(epoch)+'.png', ret_imgs)






