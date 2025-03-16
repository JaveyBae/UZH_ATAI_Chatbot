import json
import spacy
import csv
import numpy as np
import os
import rdflib
import pandas as pd
from sklearn.metrics import pairwise_distances

# 定义命名空间
WD = rdflib.Namespace('http://www.wikidata.org/entity/')
WDT = rdflib.Namespace('http://www.wikidata.org/prop/direct/')
DDIS = rdflib.Namespace('http://ddis.ch/atai/')
RDFS = rdflib.namespace.RDFS
SCHEMA = rdflib.Namespace('http://schema.org/')

def get_closest_entity(graph,entity_label, relation_label):
    try:
        # 加载嵌入向量
        entity_emb = np.load(r'data/entity_embeds.npy')
        relation_emb = np.load(r'data/relation_embeds.npy')

        # 加载实体和关系字典
        with open(r'data/entity_ids.del', 'r') as ifile:
            ent2id = {rdflib.term.URIRef(ent): int(idx) for idx, ent in csv.reader(ifile, delimiter='\t')}
            id2ent = {v: k for k, v in ent2id.items()}

        with open(r'data/relation_ids.del', 'r') as ifile:
            rel2id = {rdflib.term.URIRef(rel): int(idx) for idx, rel in csv.reader(ifile, delimiter='\t')}
            id2rel = {v: k for k, v in rel2id.items()}

        # 通过标签查找实体
        ent2lbl = {ent: str(lbl) for ent, lbl in graph.subject_objects(RDFS.label)}
        lbl2ent = {lbl: ent for ent, lbl in ent2lbl.items()}
        entity = lbl2ent.get(entity_label)

        if entity is None:
            print(f"Entity '{entity_label}' not found in lbl2ent mapping.")
            return None

        # 加载关系数据
        with open('data.json', 'r') as json_file:
            loaded_relation_data = json.load(json_file)

        # 获取关系的键
        relation_key = next((key for key, value in loaded_relation_data.items() if value == relation_label), None)
        if relation_key is None:
            print(f"Relation '{relation_label}' not found in loaded relation data.")
            return None

        # 获取关系的 URI
        relation_uri = WDT[relation_key]

        # 获取实体和关系的向量
        head = entity_emb[ent2id[entity]]
        pred = relation_emb[rel2id[relation_uri]]

        # 根据 TransE 评分函数添加向量
        lhs = head + pred
        dist = pairwise_distances(lhs.reshape(1, -1), entity_emb).reshape(-1)
        most_likely = dist.argsort()

        # 获取最接近的实体标签
        closest_entity_id = most_likely[0]
        closest_entity_uri = id2ent[closest_entity_id]
        closest_entity_label = ent2lbl.get(closest_entity_uri, "Unknown Label")

        return closest_entity_label  # 返回 label

    except KeyError as e:
        print(f"KeyError: {e} - check entity or relation existence.")
        return None
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e} - check if the files are accessible.")
        return None
    except Exception as e:
        print(f"An error occurred in get_closest_entity: {e}")
        return None
