from src.DataPreprocessing import read_csv
from src.Model import Model
import pandas as pd
import turicreate as tc


csv_data = read_csv(
    file_path='Data/raw_data_recommendation  - raw_data_recommendation  - raw_data_recommendation  - raw_data_recommendation .csv')
customers_id = csv_data["reseller_id"].values


cosine_model = Model(model_type='cosine')
cosine_similarity_model = cosine_model.load()
popularity_model = Model(model_type='popularity')
popularity_model = popularity_model.load()


def customer_recommendations(customer_id):
    """This function generate the recommended products

    Args:
        customer_id (_string_): _CustomerId_

    Returns:
        _DataFrame_: _dataframe_
    """
    n_rec = 20  # number of recommended products
    if customer_id not in customers_id:
        new_user_recomm = popularity_model.recommend(k=n_rec)
        recommendations = new_user_recomm.to_dataframe(
        ).drop_duplicates(subset=['rank'])
    else:
        recommendations = cosine_similarity_model.recommend(
            users=[customer_id], k=n_rec).to_dataframe()
    return recommendations
def show_recommendations(customer_id):
    """This function returns the recommendations of customers

    Args:
        customer_id (_string): _CustomerId_

    Returns:
        _dict_: _key:userId, value:productlist_
    """
    recommendations_df = customer_recommendations(customer_id=customer_id)
    return {
        customer_id:recommendations_df["sku"].values.tolist()
    }

def delete_recommendations(customer_id,product_list):
    """This function remove the products that are passed as product_list

    Args:
        customer_id (_string_): _CustomerId_
        product_list (_list_): _The products which user wants to be removed_
    Returns:
        _dict_: _key:userId, value:productlist_
    """
    recommendations = show_recommendations(customer_id=customer_id)
    customer_product_list = recommendations[customer_id]
    for product in product_list:
        if product in customer_product_list:
            customer_product_list.remove(product)
    return{
        customer_id:customer_product_list
    }
def update_recommendations(customer_id,product_list):
    """_This function sort the products on their score and update the recommendation

    Args:
        customer_id (_str_): _userId_
        product_list (_List[Tuple[str, float]]_): _the products which will be added to the recommendations list_

    Returns:
        _dict_: _key:userId, value:productlist_
    """
    sku,score = zip(*product_list)
    df_dict = {
        'sku':sku,
        'score':score
    }
    recommendations_df = customer_recommendations(customer_id=customer_id)
    recommendations_df = recommendations_df.append(pd.DataFrame(df_dict))
    recommendations_df = recommendations_df.sort_values(by=['score'], ascending=False)
    return {
        customer_id:recommendations_df["sku"].values.tolist()
    }

