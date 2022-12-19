from fastapi import FastAPI
from main import show_recommendations, delete_recommendations, update_recommendations
from typing import List, Tuple
from fastapi import FastAPI


app = FastAPI()


@app.get("/recommend/{customer_id}")
async def recommend_user(customer_id: str):
    """This api takes customer_id as argument shows the recommendations for that customer

    Args:
        customer_id (str): _userId_

    Returns:
        _dict_: _key:userId, value:productlist_
    """
    return show_recommendations(customer_id=customer_id)

@app.post("/delete/{customer_id}/")
async def delete_items(customer_id: str,product_list: List[str]):
    """This api removes the products which are given in product_list

    Args:
        customer_id (str): _userId_
        product_list (List[str]): _the items which will be removed from the recommended products_

    Returns:
        _dict_: _key:userId, value:productlist_
    """
    return delete_recommendations(customer_id=customer_id,product_list=product_list)


@app.post("/update/{customer_id}/")
async def test_items(customer_id: str, product_list: List[Tuple[str, float]]):
    """This api update the recommended list using the weight value on products are given 

    Args:
        customer_id (str): _userId_
        product_list (List[Tuple[str, float]]): _[["productId",weight_value],["productId",weight_value]]_

    Returns:
        _dict_: _key:userId, value:productlist_
    """

    return update_recommendations(customer_id=customer_id,product_list=product_list)
    