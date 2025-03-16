import sys
from pydantic import Extra
import rdflib
from rdflib import Graph
from speakeasypy import Speakeasy, Chatroom
from typing import List
import time3


import re
import unicodedata
from factual_q import Query_Processer
from Embedding_q import get_closest_entity
from Intent_recognizer import IntentRecognizer
from new_recommendor import Recommender
from MultiMedia import multimedia_handler
import pickle
import pandas as pd
import openai
import os



DEFAULT_HOST_URL = 'https://speakeasy.ifi.uzh.ch'

# nt_file_path = r'data/14_graph.nt'
nt_file_path = r'./data/14_graph_updated_triples.nt'
CROWD_SOURCING_CSV = './data/crowd_data/crowd-sourcing-output.csv'
pickle_file = "./data/14_graph.pickle"

listen_freq = 3


WD = rdflib.Namespace('http://www.wikidata.org/entity/')
WDT = rdflib.Namespace('http://www.wikidata.org/prop/direct/')
DDIS = rdflib.Namespace('http://ddis.ch/atai/')
RDFS = rdflib.namespace.RDFS
SCHEMA = rdflib.Namespace('http://schema.org/')

with open("./data/14_graph_updated_triples.pickle", "rb") as f:
    graph = pickle.load(f)


class Agent:
    def __init__(self, username, password):
        self.username = username
        self.speakeasy = Speakeasy(host=DEFAULT_HOST_URL, username=username, password=password)
        self.speakeasy.login()
        # self.graph = Graph()
        # self.graph.parse(nt_file_path, format='turtle')
        # 直接加载 pickle 文件
        self.graph = graph
        pickle_path = r"data/images.pkl"

        self.my_intent_recognizer = IntentRecognizer()
        self.my_query_processer = Query_Processer(self.graph)
        self.my_recommender = Recommender(self.graph)
        self.crowd_source = pd.read_csv(CROWD_SOURCING_CSV,
                                        index_col=0)
        with open(pickle_path, 'rb') as f:
            self.image_net = pickle.load(f)
    def listen(self):
        while True:
            # only check active chatrooms (i.e., remaining_time > 0) if active=True.
            try:
                rooms: List[Chatroom] = self.speakeasy.get_rooms(active=True)
            except Exception as e:
                print(f"Error fetching rooms: {e}")
                continue  # Skip this iteration on failure

            for room in rooms:
                try:
                    if not room.initiated:
                        # send a welcome message if room is not initiated
                        room.post_messages(f'Hello! This is a welcome message from {room.my_alias}.')
                        room.initiated = True
                    # Retrieve messages from this chat room.
                    # If only_partner=True, it filters out messages sent by the current bot.
                    # If only_new=True, it filters out messages that have already been marked as processed.
                    for message in room.get_messages(only_partner=True, only_new=True):
                        print(
                            f"\t- Chatroom {room.room_id} "
                            f"- new message #{message.ordinal}: '{message.message}' "
                            f"- {self.get_time()}")

                        # Implement your agent here #
                        query = message.message

                        # distinguish intent
                        intent = self.my_intent_recognizer.recognize_intent(query)

                        if intent == 'Multimedia':
                            # eg response_message ="image:2889/rm1919332864"
                            print(f'processing Multimedia...')
                            room.post_messages("Processing Multimedia.... It may takes a longer time (up to 20s) than other questions.")
                            Extracted_entities = self.my_query_processer.entity_extractor_recommender(query)
                            entities = [list(item.values())[0] for item in Extracted_entities]
                            print(entities)
                            multimedia = multimedia_handler(self.graph,self.image_net)
                            response_message = multimedia.show_img(entities)
                            # Extracted_entities = self.my_query_processer.entity_extractor_recommender(query)
                            # entities = [list(item.values())[0] for item in Extracted_entities]
                            # print(entities)
                            # multimedia = multimedia_handler(self.graph)
                            # response_message = multimedia.multimedia_question(entities)


                            # Crowd
                        else:
                            ''' 根据query中的entity匹配subject， 查询相应的FleissKappa和votes'''
                            response_message = ""
                            if intent == 'RECOMMEND':
                                entities = self.my_query_processer.entity_extractor_recommender(query)
                            else:
                                entities = self.my_query_processer.entity_extractor(query)

                            if not entities:
                                response_message = "Sorry ,I dont find answer in my database"
                            if entities:
                                entity = list(entities[0].keys())[0]
                                crowd_disclaimer = ""
                                crowd_match = self.crowd_source[
                                    self.crowd_source['Input1ID'].str.contains(entity, na=False)]
                                if not crowd_match.empty:
                                    crowd_disclaimer = (
                                        f'[Crowd, inter-rater agreement {crowd_match["FleissKappa"].iloc[0]}, '
                                        f'The answer distribution for this specific task was {crowd_match["CORRECT"].iloc[0]} support votes, '
                                        f'{crowd_match["INCORRECT"].iloc[0]} reject votes]'
                                    )
                                # Ensure crowd_disclaimer has a value even if empty
                                crowd_disclaimer = crowd_disclaimer or ""
                                # Handling various intents
                                if intent == "SPARSQL":
                                        # parse the query
                                        _, sparql_part = self._parse_query(query)
                                        # excute the query
                                        print(f'executing the query...')
                                        query_result = self._query_sparql(sparql_part)
                                        if query_result == "NO_RESULTS":
                                            response_message = "sorry no matching answer"
                                        elif query_result == "ERROR":
                                            response_message = "something went wrong."
                                        else:
                                            response_message = f"here is the searching result: {query_result}"

                                elif intent == "FACTUAL_OR_EMBEDDING":
                                        print(f'processing the factual or embedding query...')
                                        # entity = self.my_query_processer.entity_extractor(query)
                                        relation = self.my_query_processer.relation_extractor(query, entities)

                                        if relation:
                                                result = self.my_query_processer.query_knowledge_graph(entities, relation)
                                                print(f'result... {result}')
                                                if result:
                                                    response_message = "Factual Answer:" +self.generate_humanized_response(query,result)
                                                    # response_message = "Factual Answer"
                                                    #     f"Closest entity for {entity_value} with relation {relation_value}: {response_message}")

                                                else:
                                                    for entity_item, relation_item in zip(entities, relation.values()):
                                                        entity_value = list(entity_item.values())[0]
                                                        relation_value = relation_item
                                                        # 调用 get_closest_entity 函数并存储结果
                                                        response_message = get_closest_entity(self.graph,
                                                                                              entity_label=entity_value,
                                                                                              relation_label=relation_value)
                                                        print(f'result... {response_message}')
                                                        if response_message == None:
                                                            response_message = "Sorry ,I dont find answer in my database"
                                                        else:
                                                            response_message = "Embedding Answer:" + self.generate_humanized_response(
                                                                query, response_message)
                                                            #without HUman response
                                                            # response_message = (
                                                            #     f"Factual answer for {entity_value} with relation {relation_value}:  {response_message}")


                                        else:
                                            response_message = "Sorry ,I dont find answer in my database"

                                elif intent == 'RECOMMEND':
                                    print(f'processing recommend query...')
                                    # print(f'Recommend Query is  {query}')
                                    # entities = self.my_query_processer.entity_extractor_recommender(query)
                                    # print(f'Recommend entities is  {entities}')
                                    movies = [list(item.values())[0] for item in entities]
                                    print(f'Recommend movies is  {movies}')
                                    # if movies = none , give a template answer
                                    recommend_movies = self.my_recommender.recommend_movies(movies)
                                    common_features = self.my_recommender._common_feature(movies)
                                    # 如果列表有多个元素，进行格式化处理
                                    if len(common_features) > 1:
                                        feature_str = ', '.join(
                                            common_features[:-1]) + f' and {common_features[-1]}'
                                    else:
                                        feature_str = common_features[0] if common_features else "no feature"
                                    if len(recommend_movies) > 1:
                                        movies_str = ', '.join(
                                            recommend_movies[:-1]) + f' and {recommend_movies[-1]}'
                                    else:
                                        # 如果只有一个电影
                                        movies_str = recommend_movies[0] if recommend_movies else "no movies"

                                    response_message = f"The movies you mentioned share common features like {feature_str}. Based on these, I recommend similar movies such as {movies_str}."

                                    print(response_message)
                                    # shared_attributes = self.my_recommender.get_shared_attributes(entities)
                                    # recommend_movies = self.my_recommender.recommend_movies(entities)
                                    # response_message = f'Adequate recommendations will be {shared_attributes}, such as the movies {recommend_movies}'


                                elif intent == 'RANDOM':
                                        print(f'processing random query...')
                                        response_message = f"Hello, do you have any questions?"

                                response_message = response_message + crowd_disclaimer

                        # add crowd augemented info to the response message
                        # if intent != 'Multimedia' and intent != 'RECOMMEND':


                        # Default fallback for response_message
                        if not response_message.strip():  # Ensure it's not empty or just whitespace
                            response_message = "Sorry, I couldn't process your query. Please try again!"

                        # Postprocessing special characters
                        response_message = unicodedata.normalize('NFKD', response_message)
                        response_message = re.sub(r'[^\x00-\x7F]+', '', response_message).encode('utf-8').decode(
                            'utf-8')

                        # Postprocessing date
                        response_message = re.sub(
                            r'\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}:\d{2}(?:[+-]\d{2}:\d{2}|Z))?',
                            lambda x: x.group()[:10],
                            response_message
                        )

                        room.post_messages(response_message)
                        room.mark_as_processed(message)
                        print(f'response for the query:{response_message}')


                except Exception as e:
                    print(f"Error in processing room {room.room_id}: {e}")
                    continue  # Skip this room on error

                # Retrieve reactions from this chat room.
                # If only_new=True, it filters out reactions that have already been marked as processed.
                for reaction in room.get_reactions(only_new=True):
                    print(
                        f"\t- Chatroom {room.room_id} "
                        f"- new reaction #{reaction.message_ordinal}: '{reaction.type}' "
                        f"- {self.get_time()}")

                    # Implement your agent here #
                    room.post_messages(f"Received your reaction: '{reaction.type}' ")
                    room.mark_as_processed(reaction)

            time.sleep(listen_freq)

    @staticmethod
    def get_time():
        return time.strftime("%H:%M:%S, %d-%m-%Y", time.localtime())

    def _parse_query(self, query: str):
        instruction_part = ""
        sparql_part = ""
        sparql_match = re.search(r'(\bSELECT\b|\bPREFIX\b)', query, re.IGNORECASE)

        instruction_part = query[:sparql_match.start()].strip()
        sparql_part = query[sparql_match.start():].replace("'''", '').strip()

        return instruction_part, sparql_part

    def _query_sparql(self, query: str) -> list:
        """ 执行 SPARQL 查询并返回结果 """
        try:
            results = self.graph.query(query)
            # 提取字符串结果
            extracted_results = [str(row[0]) for row in results]
            if not extracted_results:  # Check if extracted_results is empty
                return "NO_RESULTS"
                # Join the extracted results into a single string
            extracted_results = ", ".join(extracted_results)
            # remove non-ASCII characters
            extracted_results = unicodedata.normalize('NFKD', extracted_results)
            extracted_results = re.sub(r'[^\x00-\x7F]+', '', extracted_results)
            return extracted_results.encode('utf-8').decode('utf-8')  # Return as a formatted string
        except Exception as e:
            return "ERROR"

    def generate_humanized_response(self, question: str, answer: str) -> str:
        """
        Generates a humanized response based on the given question and answer.

        Parameters:
            question (str): The question to be answered.
            answer (str): The answer to the question.

        Returns:
            str: A detailed, full sentence response.
            :param qustion:
        """
        # Define the prompt
        prompt = f"""Generate a natural, conversational response based on the provided question and answer. If the answer seems incorrect, do not attempt to correct it; instead, create a response that aligns with the provided answer. Ensure the response is in full sentences and sounds human-like.

        Question: {question}
        Answer: {answer}

        Example responses:
        1. Question: "Who is the director of Star Wars: Episode VI - Return of the Jedi?"
           Response: "I believe the director is Richard Marquand."

        2. Question: "Who is the screenwriter of The Masked Gang: Cyprus?"
           Response: "Based on my knowledge, it is Cengiz Küçükayvaz."

        3. Question: "When was 'The Godfather' released?"
           Response: "It was released in 1972."

        Please generate a response in a similar tone and style. Do not include the question"""
        # Generate the response using GPT-3.5-turbo
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100
        )

        # Extract and return the generated reply
        reply = response['choices'][0]['message']['content'].strip()
        return reply


if __name__ == '__main__':
    demo_bot = Agent("ancient-flame", "A3fcD1X4")
    demo_bot.listen()
