from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GSNPCell
from math import sqrt
from matplotlib import pyplot
import numpy as np


# Data structure processing
def parser(x):
	return datetime.strptime(x, '%Y/%m/%d')#%Y %Y-%m %Y%m%d

# Convert the data to supervised learning data, and add 0 to the NaN value
def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]# The original data time window moves backward by lag step
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df

# Construct difference sequence
def difference(dataset, interval=1):
	diff = np.zeros(shape=(dataset.shape[0],dataset.shape[1]))
	for i in range(interval, dataset.shape[0]):
		for j in range(0,dataset.shape[1]):
			diff[i][j] = dataset[i][j] - dataset[i - interval][j]
	return diff

# Differential inverse conversion
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# Scale the data to a number between [-1, 1]
def scale(train, test):
    # Create a scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    print(train)
   # Convert train from the format of a two-dimensional array to a 23*2 tensor
     #train = train.reshape(train.shape[0], train.shape[1])
     # Use the scaler to scale the data to between [-1, 1]
    train_scaled = scaler.transform(train)
    print(train_scaled)
  
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

# Data inverse scaling, scaler is the scaler generated before, X is a one-dimensional array, and y is a value
def invert_scale(scaler, X, y):
	# Convert x, y into a list list [x,y]->[0.26733207, -0.025524002]
    # [y] Can convert a value into a single element list
	new_row = [x for x in X] + [y]
	
	# Convert the list into a one-dimensional array containing two elements, the shape is (2,)->[0.26733207 -0.025524002]
	array = np.array(new_row)
	# Reshape the one-dimensional array into a shape (1,2), 1 row, 2 elements in each row, a 2-dimensional array ->[[ 0.26733207 -0.025524002]]
	array = array.reshape(1, len(array))
	# Inverse scaling input shape is (1,2), output shape is (1,2) -> [[ 73 15]]
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

# Build an LSTM network model and train it
# batch_size How many numbers are taken from the training set each time nb_epoch: training times
def fit_lstm(train, batch_size, nb_epoch, neurons):
    # Split X, y in the data pair, the shape is [23*1]
    X, y = train[:, 0:-1], train[:, -1]
    # Join 2D data into 3D data, the shape is [23*1*1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    # neurons is the number of neurons, batch_size is the number of samples, batch_input_shape is the input shape,
     # stateful is state retention
     # 1. The same batch of data is repeatedly trained many times, and each training state can be reserved for next use
     # 2. There are sequential associations between different batches of data, and each training state can be retained
     # 3. Different batches of data, there is no correlation between the data
    model.add(GSNPCell(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))#Peephole
    model.add(Dense(1)) #Output a predicted value, activation='tanh'
    # Define loss function and optimizer
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        # shuffle=False is not to confuse the data order
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
        # After training a cycle, reset the network once
        model.reset_states()
    return model

# Start single-step prediction, model is the trained model, batch_size is the time step, and X is a one-dimensional array
def forecast_lstm(model, batch_size, X):
	# Construct a one-dimensional array X of shape (1,) containing one element into a 3D tensor of shape (1,1,1)
	X = X.reshape(1, 1, len(X))
	# Output a two-dimensional array of yhat shape (1,1)
	yhat = model.predict(X, batch_size=batch_size)
	# Return the value of yhat in the first row and one column of a two-dimensional array
	return yhat[0,0]

def forecast_lstm2(model, batch_size, X):
	# Construct a one-dimensional array X of shape (1,) containing one element into a 3D tensor of shape (1,1,1)
	X = X.reshape(1,1,len(X))
	# Output a two-dimensional array of yhat shape (1,1)
	yhat = model.predict(X, batch_size=batch_size)
	# Return the value of yhat in the first row and one column of a two-dimensional array
	return yhat[0,0]


for i in range(10):
	dataset_name="AIR"
	dataset_name1= "air"
	# 加载数据 , date_parser=parser
	# series = read_csv('AirQualityUCI.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
	# series = read_csv('03Beijing_PM2.5.csv', header=0, parse_dates=[0], index_col=0, squeeze=True,date_parser=parser)
	
	raw_values = series.values#
	diff_values = difference(raw_values, 1)

	# 将数据转换为监督学习型数据,此时输出的supervised_values是一个二维数组
	supervised = timeseries_to_supervised(diff_values, 1)
	supervised_values = supervised.values

	# Split the data into a training set and a test set. At this time, the split data set is a two-dimensional array (take the last 60 pieces of data as the test data)
	train, test = supervised_values[:-1872,:], supervised_values[-1872:,:]
	print(train)
	# Scale both the training set and the test set to between [-1, 1]
	scaler, train_scaled, test_scaled = scale(train, test)

	# Build an LSTM network model, and train it, the number of samples: 1, the number of cyclic training: 100, the number of neurons in the LSTM layer is 8

	lstm_model = fit_lstm(train_scaled, 1, 100, 8)
	# Reconstruct the shape of the input data,
	print(train_scaled)
	train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
	print(train_reshaped)
	# Use the constructed network model for prediction training
# lstm_model.predict(train_reshaped, batch_size=1)
# print(lstm_model.predict(train_reshaped, batch_size=1))
# Traverse the test data and make a single-step prediction on the data
	predictions = np.zeros(shape=(test.shape[0],raw_values.shape[1]))
	for i in range(test_scaled.shape[0]):
		# Split the 2D training set test_scaled of (12, 2) into X, y;
# Where X is the ith row from 0 to -1, and the shape is (1,) a one-dimensional array containing one element; y is the ith row and the penultimate column, which is a number;
		X, y = test_scaled[i, 0:-1], test_scaled[i,0:-1]
		# Pass the trained model lstm_model, X variable, into the prediction function, and define the step size as 1.
		yhat = forecast_lstm2(lstm_model, 1, X)
		print(yhat.shape)
		# Inverse scaling of the predicted y value
		yhat = invert_scale(scaler, X, yhat)
		# Inverse difference conversion of the predicted y value
		yhat = inverse_difference(raw_values, yhat, test_scaled.shape[0] + 1 - i)
		# Store the predicted y value
		predictions[i,:]=yhat
		

	
	

	np.savetxt('./GSNPCell/'+dataset_name+'/prediction/' + str(count) + '_'+dataset_name1+'.csv', predictions[-1872:,:], delimiter=',')
	np.savetxt('./GSNPCell/'+dataset_name+'/origin/' + str(count) + '_'+dataset_name1+'.csv', raw_values[-1872:,;], delimiter=',')
