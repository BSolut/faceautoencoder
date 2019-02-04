import os
from keras.layers import Input, Dense, Flatten, Reshape, Conv2D, Conv2DTranspose
from keras.models import Model, load_model 

class AutoEncoder(object):
	PARAM_SIZE = 80
	INPUT_SHAPE = (166, 166, 3,)

	def __init__(self):
		self.encoder = None
		self.decoder = None
		self.model = None

	def _encoder(self):
		if self.encoder:
			return self.encoder		
		inputs = Input(shape=AutoEncoder.INPUT_SHAPE)

		x = Conv2D(3,  6, strides=2, activation="relu")(inputs)    #(81,81,3)
		x = Conv2D(128, 5, strides=2, activation="relu")(x) 	   #(39,39,128)
		x = Conv2D(128, 5, strides=2, activation="relu")(x)        #(18,18,128)
		x = Conv2D(256, 4, strides=2, activation="relu")(x)        #(8,8,256)
		x = Conv2D(256, 4, activation="relu")(x)                   #(5,5,)
		x = Conv2D(256, 3, activation="relu")(x)                   #(3,3,)
		x = Conv2D(256, 3, activation="relu")(x)                   #(1,1,256)
		x = Flatten(data_format = 'channels_last')(x)
		encoded = Dense(AutoEncoder.PARAM_SIZE)(x)

		model = Model(inputs, encoded)
		self.encoder = model
		return model

	def _decoder(self):
		if self.decoder:
			return self.decoder
		inputs = Input(shape=(AutoEncoder.PARAM_SIZE, ))

		x = Dense(256)(inputs)
		x = Reshape((1,1,256))(x)
		x = Conv2DTranspose(256, 3, activation="relu")(x)
		x = Conv2DTranspose(256, 3, activation="relu")(x)
		x = Conv2DTranspose(256, 4, activation="relu")(x)
		x = Conv2DTranspose(256, 4, strides=2, activation="relu")(x)
		x = Conv2DTranspose(128, 5, strides=2, activation="relu")(x)
		x = Conv2DTranspose(128, 5, strides=2, activation="relu")(x)
		x = Conv2DTranspose(3, 6, strides=2, activation="sigmoid")(x)

		model = Model(inputs, x)
		self.decoder = model
		return model

	def load(self, basedir):
		if os.path.isfile(basedir+'encoder.h5'):
			self.encoder = load_model(basedir+'encoder.h5')
		if os.path.isfile(basedir+'decoder.h5'):
			self.decoder = load_model(basedir+'decoder.h5')

	def encoder_decoder(self):
		enc = self._encoder()
		dec = self._decoder()

		inputs = Input(shape=AutoEncoder.INPUT_SHAPE)
		enc_out = enc(inputs)
		dec_out = dec(enc_out)
		model = Model(inputs, dec_out)

		self.model = model
		return model
