import pandas as pd

def preprocess_purchase_dummy_data(data):
    """This function add column purchase_dummy and assign value 1 to the the customers who bought product

    Args:
        data (_dataframe_): _columns['reseller_id','sku','quantity']_

    Returns:
        _dataframe_: _columns['reseller_id','sku','quantity','purchase_dummy']_
    """
    data_dummy = data.copy()
    data_dummy['purchase_dummy'] = 1
    return data_dummy

def preprocess_scaled_purchase_freq_data(data):
    """This function normalizes the product quantity and added to the column 'scaled_purchase_freq'

    Args:
        data (_dataframe_): __columns['reseller_id','sku','quantity']__

    Returns:
        _dataframe_: _columns['reseller_id','sku','quantity','scaled_purchase_freq']_
    """
    data_matrix = pd.pivot_table(data, values='quantity', index='reseller_id', columns='sku')
    data_matrix_norm = (data_matrix-data_matrix.min())/(data_matrix.max()-data_matrix.min())
    d = data_matrix_norm.reset_index()
    d.index.names = ['scaled_purchase_freq']
    return pd.melt(d, id_vars=['reseller_id'], value_name='scaled_purchase_freq').dropna()

def read_csv(file_path):
    """This function reads csv file and imputes null value with None

    Args:
        file_path (_.csv_): _csvFile_

    Returns:
        _dataframe_: _columns['reseller_id','sku','quantity']_
    """
    data = pd.read_csv(file_path,usecols=['reseller_id','sku','quantity'])
    data = data.fillna('None')
    return data
