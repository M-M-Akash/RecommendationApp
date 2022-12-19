import turicreate as tc
import os
import glob
import pandas as pd
from datetime import datetime
timestamp = datetime.now().strftime("%Y_%m_%d-%I_%M_%p")

class Model:
    """This class trains, tests and load the model
    """
    def __init__(self,model_type=None, user_id_column=None, item_id_column=None, target_column=None):
        """This constructor takes the model arguments

        Args:
            model_type (_str_, optional): _the model type could be 'cosine', 'popularity', 'pearson'_. Defaults to None.
            user_id_column (_str_, optional): _userId column name_. Defaults to None.
            item_id_column (_str_, optional): _products column name_. Defaults to None.
            target_column (_str_, optional): _the feature column name_. Defaults to None.
        """
        self.model_type = model_type
        self.user_id_column = user_id_column
        self.item_id_column = item_id_column
        self.target_column = target_column
    
    def train(self,data):
        """This function trains the model on the model type

        Args:
            data (_dataframe_): _train_data_

        Returns:
            _object_: _the trained model_
        """

        self.data = data

        if self.model_type == 'popularity':
            model = tc.popularity_recommender.create(tc.SFrame(self.data), 
                                                    user_id=self.user_id_column, 
                                                    item_id=self.item_id_column, 
                                                    target=self.target_column)
        elif self.model_type == 'cosine':
            model = tc.item_similarity_recommender.create(tc.SFrame(self.data), 
                                                        user_id=self.user_id_column, 
                                                        item_id=self.item_id_column, 
                                                        target=self.target_column, 
                                                        similarity_type='cosine')
        elif self.model_type == 'pearson':
            model = tc.item_similarity_recommender.create(tc.SFrame(self.data), 
                                                        user_id=self.user_id_column, 
                                                        item_id=self.item_id_column, 
                                                        target=self.target_column, 
                                                        similarity_type='pearson')
        return model

    def test(self,model,data):
        """This function evaluates the model and saves the model results on Output/Evaluate.csv file

        Args:
            model (_object_): _the trained model_
            data (_dataframe_): _test_data_
        """
        self.model = model
        self.data = data
        evaluation_data = self.model.evaluate(tc.SFrame(self.data))
        precision_recall_df = evaluation_data['precision_recall_overall'].to_dataframe()
        
        evaluate_df = pd.read_csv(filepath_or_buffer='Output/Evaluate.csv')
        evaluate_dict = evaluate_df.to_dict('list')
        evaluate_dict['timestamp'].append(timestamp)
        evaluate_dict['model'].append(self.model_type+" "+self.target_column)
        evaluate_dict['precision'].append(precision_recall_df['precision'].iloc[0])
        evaluate_dict['recall'].append(precision_recall_df['recall'].iloc[0])
        #print(evaluate_dict)
        evaluate_df = pd.DataFrame.from_dict(evaluate_dict)
        evaluate_df.to_csv('Output/Evaluate.csv',index=False)



    def load(self):
        """This function loads the most recent model

        Returns:
            _object_: _this model will be used to recommend products _
        """
        models = list(filter(os.path.isdir, glob.glob("Models/"+str(self.model_type)+"*")))
        model = max(models, key=os.path.getctime)
        return tc.load_model(location=str(model))
        
        
        