
import pickle
import os
import numpy as np
from datetime import datetime

import rdflib


# class KnowledgeGraph:
#     def __init__(self):
#         self.graph = rdflib.Graph()

#     def _get_graph_cache(self, graph_path, serialized_path):
#         """Cache the RDF graph into a binary file."""
#         if os.path.exists(serialized_path):
#             print("Loading serialized graph...")
#             with open(serialized_path, 'rb') as f:
#                 self.graph = pickle.load(f)
#         else:
#             print("Parsing KG file...")
#             self.graph.parse(graph_path, format='turtle')  # 或 'xml'、'n3' 等

#             with open(serialized_path, 'wb') as f:
#                 pickle.dump(self.graph, f)
#             print(f"Serialized graph saved to {serialized_path}")

#     def load_graph(self, graph_path, cache_path=None):
#       """Loads the graph, using cache if available."""
#       if cache_path is None:
#           cache_path = graph_path + ".pickle" # 默认缓存文件名

#       self._get_graph_cache(graph_path, cache_path)


# # 使用示例：
# kg = KnowledgeGraph()

# graph_file = "data/14_graph.nt"  # 替换为你的 RDF 文件路径和正确的扩展名 (例如 .ttl, .rdf, .nt)
# cache_file = "data/14_graph.pickle"  # 缓存文件路径
# pickle_file = "data/14_graph.pickle"
# # kg.load_graph(graph_file, cache_file) #  加载图，并使用缓存


# # 直接加载 pickle 文件
# with open("data/14_graph.pickle", "rb") as f:
#     graph = pickle.load(f)



# 现在可以使用 kg.graph 进行操作了
# 例如，打印图中所有三元组：
# for s, p, o in kg.graph:
#     print(s, p, o)

# # 将 JSON 文件转换为 Pickle 文件
# def json_to_pickle(json_path, pickle_path):
#     # 读取 JSON 数据
#     with open(json_path, 'r') as f:
#         data = json.load(f)
    
#     # 将数据写入 Pickle 文件
#     with open(pickle_path, 'wb') as f:
#         pickle.dump(data, f)

# # 示例
# json_to_pickle('data/images.json', 'data/images.pkl')


# # 加载 Pickle 数据
# def load_pickle(pickle_path):
#     with open(pickle_path, 'rb') as f:
#         data = pickle.load(f)
#     return data

# # 示例
# data = load_pickle('data/images.pkl')



class multimedia_handler():
    def __init__(self, KG, imagenet ):
        self.KG = KG  # Directly an instance of rdflib.Graph

        self.image_net = imagenet
        self.PANELTY = 0.3
        self.imgs = [i['img'] for i in self.image_net]
        self.ids = [set(i['movie'] + i['cast']) for i in self.image_net]
        # Initialize cache from file if it exists
        self.cache_path = r"data/cache.pkl"       #cache_path
        self.cache = self.load_cache()  # Load cache from file

    def ent_to_id(self, entities):
        """Map entity names to IDs"""
        ent_dic = {}
        for ent_name in entities:
            query = f'''
            PREFIX ddis: <http://ddis.ch/atai/>
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX schema: <http://schema.org/>

            SELECT ?obj WHERE {{
                SERVICE <https://query.wikidata.org/sparql> {{
                    ?sub rdfs:label "{ent_name}"@en.
                    ?sub wdt:P345 ?obj.
                }}
            }} LIMIT 1
            '''.strip()

            ent_dic[ent_name] = []
            # Modify to self.KG.query
            for row, in self.KG.query(query):
                ent_dic[ent_name].append(str(row))

        # Collect all IDs
        tmp = []
        for ent_list in ent_dic.values():
            tmp += ent_list
        
        return tmp

    def load_cache(self):
        """Load cache from file"""
        if os.path.exists(self.cache_path):  # Check if the cache file exists
            with open(self.cache_path, 'rb') as f:
                return pickle.load(f)
        return {}  # Return an empty cache if no cache file exists

    def save_cache(self):
        """Save cache to file"""
        with open(self.cache_path, 'wb') as f:
            pickle.dump(self.cache, f)




    def show_img(self, entities):

        # Check cache first
        entity_key = tuple(sorted(entities))  # Convert list to a sorted tuple for consistent key
        if entity_key in self.cache:
            print("load the cache")
            return self.cache[entity_key]  # Return cached result

        """Display the most relevant image based on the input entity list"""
        print('=====================', entities)
        id_lst = self.ent_to_id(entities)
        print('=====================', id_lst)
        
        # Calculate the score for each image
        score_lst = [
            len(set(id_lst) & single_img) - self.PANELTY * len(single_img)
            for single_img in self.ids
        ]
        
        if not score_lst:  # Return a default value if no scores exist
            return "No matching image found."

        idx = np.argmax(score_lst)  # Find the index of the highest score
        print('=====================', len(score_lst), idx)

        result = f'image:{str(self.imgs[idx].split(".")[0])}'
        # Store result in cache and save to file
        self.cache[entity_key] = result
        self.save_cache()  # Save the updated cache to file
        return result

    def multimedia_question(self,entities):
        entities = str(entities[0])
        entity = entities.strip().lower()
        RDFS_LABEL = rdflib.URIRef("http://www.w3.org/2000/01/rdf-schema#label")
        uri = None  # 确保变量 `uri` 在没有匹配时定义
        for subj in self.KG.subjects(predicate=RDFS_LABEL):
            label = self.KG.value(subject=subj, predicate=RDFS_LABEL)
            if label and label.toPython().strip().lower() == entity:
                uri = subj
                break

        if uri is not None:
            P345 = rdflib.URIRef("http://www.wikidata.org/prop/direct/P345")
            p345_value = self.KG.value(subject=uri, predicate=P345)
            if p345_value:
                p345_value = p345_value.toPython()
                for item in self.image_net:
                    if isinstance(item["cast"], list):
                        for subitem in item["cast"]:
                            if subitem == str(p345_value):
                                return "image:" + item["img"]
                    else:
                        if item["cast"] == str(p345_value):
                            return "image:" + item["img"]
        return "No picture of this film can be afforded! Please change a movie!"

# with open("data/14_graph.pickle", "rb") as f:
#             graph = pickle.load(f)


# # entities = ["Julia Roberts","Pretty Woman"]
# entities = ["Denzel Washington"]
# entities = ["Sandra Bullock"]       #Let me know what Sandra Bullock looks like. 
# multimedia = multimedia_handler(graph)
# print(multimedia.show_img(entities))