import pandas as pd
import numpy as np

def preprocess(data):
	data['Image'] = data['Image'].apply(lambda x: np.array(list(map(float, x.split()))).reshape(96, 96))
	for i in data.columns[:-1]:
		data[i].fillna(data[i].mean(), inplace=True)
	return data

if __name__ == "__main__":
	train = pd.read_csv('./data/training.csv')
	test = pd.read_csv('./data/testing.csv')

	train = preprocess(train)
	test = preprocess(test)

	train.to_csv('./data/train.csv', index=False)
	test.to_csv('./data/test.csv', index=False)