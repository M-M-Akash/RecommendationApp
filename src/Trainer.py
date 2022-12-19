from DataPreprocessing import read_csv, preprocess_purchase_dummy_data, preprocess_scaled_purchase_freq_data
from Model import Model
from sklearn.model_selection import train_test_split
from datetime import datetime


csv_data = read_csv(
    file_path='Data/raw_data_recommendation  - raw_data_recommendation  - raw_data_recommendation  - raw_data_recommendation .csv')
purchase_dummy_data = preprocess_purchase_dummy_data(data=csv_data)
scaled_purchase_freq_data = preprocess_scaled_purchase_freq_data(data=csv_data)

timestamp = datetime.now().strftime("%Y_%m_%d-%I_%M_%p")

# cosine_model = Model(model_type='cosine', user_id_column='reseller_id',
#                                           item_id_column='sku', target_column='purchase_dummy')
# cosine_trained_model = cosine_model.train(data=purchase_dummy_data)
# cosine_trained_model.save(
#     location='Models/cosine_model'+timestamp)

    
# popularity_model = Model(model_type='popularity', user_id_column='reseller_id',
#                                                   item_id_column='sku', target_column='scaled_purchase_freq')
# popularity_trained_model = popularity_model.train(data=scaled_purchase_freq_data)
# popularity_trained_model.save(
#     location='Models/popularity_model'+timestamp)


def split_data(data):
    '''
    Splits dataset into training and test set.

    Args:
        data (pandas.DataFrame)

    Returns
        train_data
        test_data
    '''
    train_data, test_data = train_test_split(data, test_size=.2)
    return train_data, test_data


train_data_quantity, test_data_quantity = split_data(csv_data)
train_data_dummy, test_data_dummy = split_data(purchase_dummy_data)
train_data_norm, test_data_norm = split_data(scaled_purchase_freq_data)


#when train and test models
# cosine_model = Model(model_type='cosine', user_id_column='reseller_id',
#                      item_id_column='sku', target_column='purchase_dummy')
# cosine_trained_model = cosine_model.train(data=train_data_dummy)
# cosine_model.test(model=cosine_trained_model, data=test_data_dummy)

cosine_model = Model(model_type='cosine', user_id_column='reseller_id',
                     item_id_column='sku', target_column='quantity')
cosine_trained_model = cosine_model.train(data=train_data_quantity)
cosine_model.test(model=cosine_trained_model, data=test_data_quantity)

# popularity_model = Model(model_type='popularity', user_id_column='reseller_id',
#                                                   item_id_column='sku', target_column='scaled_purchase_freq')
# popularity_trained_model = popularity_model.train(data=train_data_norm)
# popularity_model.test(model=popularity_trained_model,data=test_data_norm)
