from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from preprocess import *

class KeyPointGen(Dataset):
	def __init__(self, train=True):
		self.train = train
		if self.train:
			df = pd.read_csv('./data/training.csv')
			df = preprocess(df)
			self.data = df['Image'].values/255.0
			self.y = df.drop(['Image'], axis=1).values
		else:
			df = pd.read_csv('./data/testing.csv')
			df = preprocess(df)
			self.data = df['Image'].values/255.0
			self.y = df['ImageId'].values

		self.samples = []
		for i in range(self.data.shape[0]):
			self.samples.append((self.data[i].reshape(1, 96, 96), self.y[i]))

	def __len__(self):
		return self.data.shape[0]

	def __getitem__(self, idx):
		return self.samples[idx]