from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import pairwise_distances
import rdflib
from graph import *
import json
import re
import rdflib




class Recommender():
    """
    A class to recommend movies and shared features based on similarity scores.
    """
    def __init__(self,graph):
        """
        Initialize the recommender with embeddings and a knowledge graph handler.
        """
        self.entity_emb = np.load('data/entity_embeds.npy')
        with open('data/ent2id.pkl', 'rb') as f:
            self.ent2id = pickle.load(f)
        with open('data/id2ent.pkl', 'rb') as f:
            self.id2ent = pickle.load(f)
        with open('data/ent2lbl.pkl', 'rb') as f:
            self.ent2lbl = pickle.load(f)
        with open('data/lbl2ent.pkl', 'rb') as f:
            self.lbl2ent = pickle.load(f)

        self.rel_emb = np.load('data/relation_embeds.npy')
        self.graph = graph
        self.rel_name_list = [
            'film editor', 'genre', 'nominated for', 'published in', 'award received', 'part of the series',
            'production designer', 'production company', 'art director', 'musical conductor',
            'original film format', 'platform', 'derivative work', 'sound designer', 'screenwriter',
            'character designer', 'inspired by', 'main subject', 'distributed by', 'followed by', 'genre'
            ]
        
                # Load relation data from data.json file
        
        self.rel_list = []

# Load the relations from the JSON file
        with open('data.json', 'r', encoding='utf-8') as json_file:
            self.loaded_relation_data = json.load(json_file)

# Iterate through each relation name in rel_name_list
        for rel in self.rel_name_list:
    # Remove any newlines or unwanted spaces in the relation name
            cleaned_rel = rel.replace('\n', '').strip()

    # Check if the cleaned relation name exists in the loaded relation data
            if cleaned_rel in self.loaded_relation_data.values():
        # Find the relation ID corresponding to the cleaned relation la   
               relation_id = [key for key, value in self.loaded_relation_data.items() if value == cleaned_rel][0]

        # Add the corresponding relation ID to the rel_list
               self.rel_list.append(relation_id)

        print(self.rel_list)


    def labels_to_indices(self, labels: set) -> list:
        """
        Convert a set of labels to corresponding entity indices.
        :param labels: A set of labels
        :return: A list of indices
        """
        input_indices = []
        for label in labels:
            entity_uri = self.lbl2ent.get(label)  # Map label to entity URI
            if entity_uri and entity_uri in self.ent2id:
                input_indices.append(self.ent2id[entity_uri])
        return input_indices

    def recommend_movies(self, labels: set) -> list:
        """
        Recommend the top three similar movies based on input labels.
        :param labels: A set or list of input labels
        :return: A list of top three recommended movie labels
        """
        # Convert labels to indices
        input_indices = self.labels_to_indices(labels)

        if not input_indices:
            raise ValueError("Input labels could not be mapped to any valid entities.")

        # Compute recommendations
        distances = []
        for ent_id in input_indices:
            dist = pairwise_distances(self.entity_emb[ent_id].reshape(1, -1), self.entity_emb).reshape(-1)
            distances.append(dist)

        # Average distances
        combined_dist = sum(distances) / len(distances)

        # Sort and exclude input labels
        input_labels = {self.ent2lbl[self.id2ent[ent_id]] for ent_id in input_indices}
        most_likely = [
            idx for idx in combined_dist.argsort()
            if self.ent2lbl.get(self.id2ent[idx], "Unknown") not in input_labels  # Exclude input labels
        ]

        # Get top three recommended movie labels
        similar_movies = [
            self.ent2lbl.get(self.id2ent[idx], "Unknown")
            for idx in most_likely[:3]
        ]
        return similar_movies
    


    def _common_feature(self, movies: set, K=3) -> list:
        '''
        Params:
            sentence: 
            K: top-K common features
        Return:
            movies: a list of movie Named entities in the sentence
            closest_rel_lbl: a list of common feature given movie list (currently only entities, but we can attach relation later)
        '''      
        query_compo = []
        features_emb = []
        for rel in self.rel_list:
            query_compo.append("{?movie <%s> ?obj . }" % rel)
            # union_parts.append("{?obj <%s> ?movie . }" % rel)
        

        base_url = "http://www.wikidata.org/prop/direct/"

            # 拼接完整 URL
        updated_list = [item.replace('<P', f'<{base_url}P') for item in query_compo]
        union_query = " UNION ".join(updated_list)

        # print(union_query)
        for movie_name in movies:
            query = '''
                        SELECT ?obj
                        WHERE {
                            
                            SERVICE <https://query.wikidata.org/sparql>{
                            
                                ?movie rdfs:label "%s"@en. 
                                                
                                %s                              

                            }
                        }'''%(movie_name, union_query)
            query = query.strip()

            res = []
            for row in self.graph.query(query):
                res.append([str(i) for i in row]) 

            # print(f"res: {res}")
            emb_list = [self.entity_emb[self.ent2id[rdflib.term.URIRef(ent[0])]] 
                            for ent in res if rdflib.term.URIRef(ent[0]) in self.ent2id.keys()]
            features_emb += emb_list
        
        # print(f"features_emb: {features_emb}")
        cluster = KMeans(n_clusters=np.max(( int(len(features_emb)/2), 1) ),
                         n_init='auto')
        cluster.fit(np.array(features_emb))
        labels_cnt = np.bincount(cluster.labels_)

        top_label_idx = np.argsort(labels_cnt)[::-1][:K]
        centroids = cluster.cluster_centers_[top_label_idx]
        dist = pairwise_distances(centroids, self.entity_emb)
        
        
        # select features based on the most K-th significant centroids 
        closest_rel_idx = dist.argsort()[:,0]
        # closest_rel_lbl = [id2ent[i] for i in closest_rel_idx]
        closest_rel_uri = [self.id2ent[i] for i in closest_rel_idx]
        closest_rel_lbl = [self.ent2lbl[i] for i in closest_rel_uri]
        
        return closest_rel_lbl

   


#
# def main():
#     # Initialize the Recommender class
#     recommender = Recommender(graph)
#
#     # Input label set
#     labels = {"The Lion King", "Beauty and the Beast", "Pocahontas"}
#
#     # Get recommended movies
#     try:
#         similar_movies = recommender.recommend_movies(labels)
#         print("Recommended Movies:")
#         print(similar_movies)
#     except ValueError as e:
#         print(f"Error: {e}")
#
#     # Get common features
#     try:
#         common_features = recommender._common_feature(labels, K=3)
#         print("Common Features:")
#         print(common_features)
#     except ValueError as e:
#         print(f"Error: {e}")
#
# if __name__ == "__main__":
#     main()
