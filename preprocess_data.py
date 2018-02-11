import pandas as pd

# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

def remove_data(data):

# Define columns of data to keep from historical stock data
	
	item= []
	open= []
	close = []
	volume = []

# Loop through the stock data objects backwards and store factors we want to keep
	i_counter=0
	for i in range(len(data)-1,-1,-1):
		item.append(i_counter)
		open.append(data['Open'][i])
		close.append(data['Close'][i])
		volume.append(data['Volume'][i])
		i_counter += 1	
	
	# Create a data frame for stock data
		
	stocks = pd.DataFrame()
	
	 # Add factors to data frame
    
	stocks['Index']=item
	stocks['Open']=open
	stocks['Close']=pd.to_numeric(close)
	stocks['Volume']=pd.to_numeric(volume)
    
   
	# return new formatted data
	
	return stocks
	
def get_normalised_data(data):
    """
    Normalises the data values using MinMaxScaler from sklearn
    :param data: a DataFrame with columns as  ['index','Open','Close','Volume']
    :return: a DataFrame with normalised value for all the columns except index
    """
    # Initialize a scaler, then apply it to the features
    scaler = MinMaxScaler()
    numerical = ['Open', 'Close', 'Volume']
    data[numerical] = scaler.fit_transform(data[numerical])

    return data