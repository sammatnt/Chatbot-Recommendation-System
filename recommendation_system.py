import os
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from lightfm import LightFM
from lightfm.evaluation import auc_score
from sklearn.model_selection import train_test_split

class RecommendationSystem:
    def __init__(self, df, model_path='lightfm_model.pkl'):
        """
        Initialize the recommendation system with a DataFrame.
        Sets default URLs if not already present in the data,
        prepares the interaction matrix, and loads or trains the LightFM model.
        """
        self.df = df

        # Ensure required URL columns exist
        if 'url' not in self.df.columns:
            self.df['url'] = 'https://www.fashom.com/'
        if 'product_url' not in self.df.columns:
            self.df['product_url'] = 'https://www.fashom.com/'

        self.model_path = model_path
        self.old_users = set(self.df['requester_id'].unique())

        # Prepare user-item interaction matrix and mappings
        self.train_matrix, self.user_mapping, self.item_mapping = self.prepare_interaction_matrix()

        # Load or train the LightFM model
        if os.path.exists(self.model_path):
            self.model = self.load_model()
            print("Loaded pre-trained LightFM model.")
        else:
            self.model = self.train_model()
            self.save_model()

    def prepare_interaction_matrix(self):
        """
        Converts interaction data into a sparse matrix format
        required by LightFM and also creates user and item mappings.
        """
        interaction_data = self.df[['requester_id', 'Variant ID', 'Quantity']].copy()

        # Create mappings for users and items
        user_mapping = {user_id: i for i, user_id in enumerate(interaction_data['requester_id'].unique())}
        item_mapping = {item_id: i for i, item_id in enumerate(interaction_data['Variant ID'].unique())}

        # Map each interaction to matrix indices
        row = interaction_data['requester_id'].map(user_mapping)
        col = interaction_data['Variant ID'].map(item_mapping)
        data = interaction_data['Quantity'].astype(float)

        shape = (len(user_mapping), len(item_mapping))
        interaction_matrix = coo_matrix((data, (row, col)), shape=shape)

        return interaction_matrix, user_mapping, item_mapping

    def train_model(self):
        """
        Trains the LightFM model with a logistic loss function,
        and reports the training AUC.
        """
        model = LightFM(no_components=30, loss='logistic')
        model.fit(self.train_matrix, epochs=30, verbose=True)
        train_auc = auc_score(model, self.train_matrix).mean()
        print(f"Trained LightFM model with Train AUC: {train_auc}")
        return model

    def save_model(self):
        """
        Saves the trained LightFM model along with the user and item mappings.
        """
        joblib.dump((self.model, self.user_mapping, self.item_mapping), self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        """
        Loads the LightFM model and the mappings from file.
        """
        model, self.user_mapping, self.item_mapping = joblib.load(self.model_path)
        return model

    def get_recommendations(self, requester_id, top_n=10):
        """
        For an existing user, generates recommendations by predicting scores for all items.
        Returns the top_n recommended Variant IDs and corresponding scores.
        """
        if requester_id not in self.user_mapping:
            return [], []

        user_index = self.user_mapping[requester_id]
        n_items = self.train_matrix.shape[1]

        # Get predicted scores for all items
        scores = self.model.predict(user_index, np.arange(n_items))
        top_items = np.argsort(-scores)[:top_n]

        # Reverse the mapping to get original Variant IDs
        reverse_item_mapping = {v: k for k, v in self.item_mapping.items()}
        recommended_ids = [reverse_item_mapping[i] for i in top_items]
        top_scores = [scores[i] for i in top_items]

        return recommended_ids, top_scores

    def recommend_for_old_user(self, user_id):
        """
        Generate recommendations for an existing user based on collaborative filtering.
        Retrieves product information and returns a list of recommendations.
        """
        top_variant_ids, top_scores = self.get_recommendations(user_id)
        recommendations = []
        for variant_id, score in zip(top_variant_ids, top_scores):
            product_info = self.df[self.df['Variant ID'] == variant_id].iloc[0]
            recommendations.append({
                "product_id": variant_id,
                "score": score,
                "url": product_info['url'],
                "product_url": product_info['product_url']
            })
        return {
            "user_id": user_id,
            "recommendations": recommendations
        }

    def recommend_for_new_user(self, user_filters, top_n=10):
        """
        Generate recommendations for a new user based on filter criteria.
        Filters the DataFrame based on user_filters and returns product info.
        """
        filtered_products = self.df
        if user_filters:
            for key, value in user_filters.items():
                if key in filtered_products.columns:
                    if isinstance(value, list):
                        filtered_products = filtered_products[filtered_products[key].isin(value)]
                    else:
                        filtered_products = filtered_products[filtered_products[key] == value]

        # Drop duplicate products based on Variant ID
        filtered_products = filtered_products.drop_duplicates(subset='Variant ID')

        product_columns = ['Variant ID', 'url', 'product_url']
        if 'Product ID' in filtered_products.columns:
            product_columns.append('Product ID')
        if 'Color' in filtered_products.columns:
            product_columns.append('Color')
        if 'Product Type' in filtered_products.columns:
            product_columns.append('Product Type')

        recommendations = filtered_products[product_columns].head(top_n).to_dict('records')
        return {
            "filters": user_filters,
            "recommendations": recommendations
        }

    def is_old_user(self, user_id):
        """
        Checks if a user exists in the historical data.
        """
        return user_id in self.old_users

    def recommend(self, user_id=None, user_filters=None):
        """
        Main method that orchestrates recommendation generation:
          - For old users, use collaborative filtering.
          - For new users, use filtered product recommendations.
        """
        if user_id and self.is_old_user(user_id):
            return self.recommend_for_old_user(user_id)
        return self.recommend_for_new_user(user_filters)
