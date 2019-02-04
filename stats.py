import sys,argparse
import numpy as np
import cv2
from fdream import Config
from fdream.autoencoder import AutoEncoder
from matplotlib import pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default="./weights/", help="Path with de/encode.h5 files")
args = vars(parser.parse_args(sys.argv[1:]))

ae = AutoEncoder()
ae.load(args["model"])
ae_model = ae.encoder_decoder()
ae_model.compile(optimizer='adam', loss='mse')


train_data = np.load('train_data.npy').astype(np.float32) / 255

x_enc = ae.encoder.predict([train_data])

x_mean = np.mean(x_enc, axis=0)
x_stds = np.std(x_enc, axis=0)
x_cov = np.cov((x_enc - x_mean).T)
u, s, v = np.linalg.svd(x_cov)
e = np.sqrt(s)

np.save(args["model"] + 'means.npy', x_mean)
np.save(args["model"] + 'stds.npy', x_stds)
np.save(args["model"] + 'evals.npy', e)
np.save(args["model"] + 'evecs.npy', v)