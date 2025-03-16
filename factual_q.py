from transformers import AutoTokenizer
from transformers import pipeline
import spacy
import json
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import json
import rdflib
import csv
import re
from rdflib import URIRef
# step1: extract entity and relation
class Query_Processer:
    def __init__(self, graph):
        device = 0
        self.nlp = spacy.load("en_core_web_sm")
        self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
        self.ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", 
                        tokenizer=self.tokenizer,
                        aggregation_strategy = "simple" ,
                                     device = device )

        with open('data_entity.json', 'r', encoding='utf-8') as f:    ##change
            self.data_entities = json.load(f)
        self.graph = graph
        self.RDFS = rdflib.namespace.RDFS

    def query_knowledge_graph(self, entity: list, relation: dict) -> str:
        # # Step 1: Prepare the entity and relation by resolving their URIs
        # self.ent2lbl = {ent: str(lbl) for ent, lbl in self.graph.subject_objects(self.RDFS.label)}
        # self.lbl2ent = {lbl: ent for ent, lbl in self.ent2lbl.items()}

        # entity_uri = self.lbl2ent.get(entity)  # Example: Resolve 'Good Neighbors' to URI
        # relation_uri = self.lbl2ent.get(relation)  # Example: Resolve 'genre' to URI

        # if not entity_uri or not relation_uri:
        #     return "Entity or relation not found in the knowledge graph."

        # # Step 2: Query the knowledge graph to retrieve objects related to the entity and relation
        # results = []
        # for obj in self.graph.objects(subject=entity_uri, predicate=relation_uri):
        #     results.append(str(obj))  # Convert the object to a string

        # # Return the results in a readable format
        # if results:
        #     return f"Results for '{entity}' with relation '{relation}': {', '.join(results)}"
        # else:
        #     return f"No results found for '{entity}' with relation '{relation}'."

        # query = f'''
        #     PREFIX ddis: <http://ddis.ch/atai/>
        #     PREFIX wd: <http://www.wikidata.org/entity/>
        #     PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        #     PREFIX schema: <http://schema.org/>

        #     SELECT ?result WHERE {{
        #         SERVICE <https://query.wikidata.org/sparql> {{
        #             wd:{entity_key} wdt:{relation_key} ?result .
        #         }}
        #     }} LIMIT 1
        # '''.strip()

        # result = self.graph.query(query)

        entity_value = list(entity[0].values())[0]
        relation_key = next(iter(relation.keys()))
        # Construct the SPARQL query dynamically

        query = f'''
                        PREFIX ddis: <http://ddis.ch/atai/>
                        PREFIX wd: <http://www.wikidata.org/entity/>
                        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
                        PREFIX schema: <http://schema.org/>

                        SELECT ?result WHERE {{
                            SERVICE <https://query.wikidata.org/sparql> {{
                                ?film rdfs:label "{entity_value}"@en .
                                ?film wdt:{relation_key} ?result .
                            }}
                        }} LIMIT 1
                    '''.strip()

        # Return the results in a readable format
        # Step 2: Execute the SPARQL query
        # 执行查询并提取结果
        try:
            result = self.graph.query(query)
            if result:
                result_bindings = list(result)
                if result_bindings:
                    # 提取结果的值
                    result_value = result_bindings[0].get("result")
                    print(f"result_value {result_value}")
                    type1 = type(result_value)
                    print(f"type:{type1}")
                    if isinstance(result_value, URIRef):
                        # with open(r'data/entity_ids.del', 'r') as ifile:
                        #     ent2id = {rdflib.term.URIRef(ent): int(idx) for idx, ent in
                        #               csv.reader(ifile, delimiter='\t')}
                        #     id2ent = {v: k for k, v in ent2id.items()}
                        # result_value = id2ent.get(result_value)
                        result_value = self.data_entities.get(result_value.toPython(), None)
                        print(f"result_value {result_value}")
                        return result_value
                    else:
                        return result_value.toPython() if result_value else ""

            return ""
        except Exception as e:
            return ""  # 出错时返回空字符串

        # Step 2: Construct the SPARQL query string
        # sparql_query = f"""
        # SELECT ?object WHERE {{
        #     <{entity_uri}> <{relation_uri}> ?object .
        # }}
        # """
        
        # # Return the SPARQL query string
        # return sparql_query


    def _entity2id(self, entity: str)-> dict:
        threshold = 88

        matching_results = {}

        best_match = process.extractOne(entity, self.data_entities.values(), scorer=fuzz.ratio, score_cutoff=threshold)

        if best_match:
            matched_title, best_score = best_match[0], best_match[1]  
            entity_uri = next((uri for uri, title in self.data_entities.items() if title == matched_title), None)
            entity_id = entity_uri.split('/')[-1:][0]
            if entity_uri:
                matching_results[entity_id] = matched_title
        return matching_results
    
    def entity_extractor_recommender(self, query)->list:
        '''
        input query
        return matching result list[]
        '''
        entities = []
        results = self.ner_pipeline(query)
        # print('ner results:',results)
        for i, result in enumerate(results):
            word = result['word'].replace("##", "")
            entities.append(word)
        # print(f'entities: {entities}')

        matching_results = []
        # fuzzy match
        if entities:


            for entity in entities:
                matching = self._entity2id(str(entity))
                print(f'matching results:{matching}')
                if matching:
                    matching_results.append(matching)
        return matching_results
    
    def entity_extractor(self, query)->list:
        '''extract entity and return a list(or dictionary) of the entity'''
        # results = self.ner_pipeline(query)
        # entities = {result['word']: result['entity_group'] for result in results}        
        # return entities
        entities = []
        results = self.ner_pipeline(query)
        
        current_entity = ""
        current_label = None
        current_start = None

        for i, result in enumerate(results):
            word = result['word'].replace("##", "")  # Remove '##' from subword tokens
            if word.startswith("##"):
                word = word[2:]  # Remove '##' prefix if still present after replacement

            if i > 0 and result['entity_group'] == results[i - 1]['entity_group'] and result['start'] == current_start + 1:
                # If the current entity group is the same as the previous one and they are consecutive, merge them
                current_entity += word
                current_start = result['end']
            else:
                # Append the previous entity to the list
                if current_entity:
                    entities.append({"word": current_entity.strip(), "entity": current_label})
                # Start a new entity group
                current_entity = word
                current_label = result['entity_group']
                current_start = result['end']

        # Append the last entity
        if current_entity:
            entities.append({"word": current_entity.strip(), "entity": current_label})

        # Convert list of entities to dictionary format for return
        merged_entities = {}
        for entity in entities:
            if entity['entity'] in merged_entities:
                merged_entities[entity['entity']] += " " + entity['word']
            else:
                merged_entities[entity['entity']] = entity['word']
        
        print(f'merged entities:{merged_entities}')


        results = []
        # 匹配现有的entity
        if merged_entities:


            for entity_value in merged_entities.values():
                matching_results = self._entity2id(str(entity_value))
                print(f'matching results:{matching_results}')
                if matching_results:
                    results.append(matching_results)
                else:
                    pass

        return results

    def relation_extractor(self, query, entity) -> dict:
        '''Extract the relation/predicate and return a dictionary with relation_id as keys'''

        #替换对应relation
        synonyms = {"released": "publication date", "directed": "director" , "born":"place of birth" , "MPAA rating": "MPAA film rating"}
        for key, value in synonyms.items():
            if key in query:
                query = query.replace(key, value)
        # Load relation data from data.json file
        with open('data.json', 'r', encoding='utf-8') as json_file:
            self.loaded_relation_data = json.load(json_file)

        # Create a dictionary to store matching results
        extracted_relations = {}

        # Ensure query is in list format (convert to list even if it's a single string)
        if isinstance(query, str):
            query = [query]

        # Convert the entity list of dictionaries to a single string list
        entity_text = " ".join([list(ent.values())[0].lower() for ent in entity if isinstance(ent, dict)])

        # Iterate through each sentence in the query
        for sentence in query:
            print(f"\nProcessing sentence: '{sentence}'")

            # Iterate over each relation label in data.json
            for relation_id, relation_label in self.loaded_relation_data.items():
                # Check if the relation label is in the sentence but not in the entity_text
                if relation_label.lower() in sentence.lower() and relation_label.lower() not in entity_text:
                    # If match criteria are met, add relation_id and relation_label to the results dictionary
                    extracted_relations[relation_id] = relation_label

        # Return the matched relations dictionary (keys are relation_id, values are relation_label)
        return extracted_relations

        # relations = {}
        # doc = self.nlp(query)
        # # for ent in doc.ents
        # #     print(f'entity text:{ent.text}, label: {ent.label_}')
        # for token in doc:
        #     # print(token.dep_, token.pos_)
        #     if token.dep_ in ("ROOT", "acl", "advcl", "relcl") and token.pos_ == "VERB":
        #         subject = [w for w in token.children if w.dep_ in ("nsubj", "nsubjpass")]
        #         obj = [w for w in token.children if w.dep_ in ("dobj", "attr", "prep", "pobj")]
        #         if subject and obj:
        #             relations["relation"]= token.lemma_
        #     if token.dep_ in ("attr", "nsubj") and token.pos_ == "NOUN":
        #         obj = [w for w in token.children if w.dep_ in ("prep", "pobj")]
        #         if obj:
        #             relations["relation"]= token.lemma_
        # return relations

