from llama_index.core import VectorStoreIndex, Document, ServiceContext, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.memory import ChatMemoryBuffer

import os
import copy
import json

key = open('api_key.json')
data = json.load(key)
os.environ["OPENAI_API_KEY"] = data['OPENAI_API_KEY']

merge_node_prompt = """
You will be given two documents. Your job is to determine if they are about the exact same topic and, if they are, merge them into a single coherent document. Otherwise, you should output "NO".

For example:
- If Document1 is "Sam loves to perform outdoor activities such as biking and hiking" and Document2 is "Sam loves to play video games", the merged document could be "Sam loves to perform outdoor activities such as biking and hiking and also enjoys playing video games." Since both documents are about Sam's interests, they can be merged. Therefore, the output should be:
  -------------------------
  YES
  Sam loves to perform outdoor activities such as biking and hiking and also enjoys playing video games.
  -------------------------
- If Document1 is "Buenos Aires, city and capital of Argentina. The city is situated on the shore of the Río de la Plata, 150 miles (240 km) from the Atlantic Ocean." and Document2 is "Santiago, capital of Chile. It lies on the canalized Mapocho River, with views of high Andean peaks to the east.", the output should be:
  -------------------------
  NO
  -------------------------
  These documents should not be merged because they are about two different cities.

Your input is provided below:
-------------------------
Document1: {doc1}
Document2: {doc2}
-------------------------

The output format should be:
-------------------------
{{"YES" if the documents should be merged. "NO" if the documents shouldn't be merged}}
{{the merged document text if the text above was "YES"}}
-------------------------

Follow this output format strictly. Do not merge documents unless they are about the same topic. Ensure that the merged document makes sense and flows naturally. Default to using the information in Document1 over Document2 if there are contradictions.
"""

# merge_node_prompt = """
# You will be given two documents that are closely related with one other. Your job will be to either output "NO" in the case that the two documents aren't about the exact same topic, or output "YES" and merge the two documents into a single document. 
#
# For example, if I have two documents such as "Sam loves to perform outdoor activities such as biking and hiking" and "Sam loves to play video games", then an acceptable merge would be "Sam loves to perform outdoor activities such as biking and hiking and video games". In this case, you can merge the documents because they are both about Sam's interests.
#
# However, if you have two documents such as "Buenos Aires, city and capital of Argentina. The city is situated on the shore of the Río de la Plata, 150 miles (240 km) from the Atlantic Ocean." and "Santiago, capital of Chile. It lies on the canalized Mapocho River, with views of high Andean peaks to the east.", then DO NOT MERGE these two documents because they are about two different things, namely two different cities. These are two different cities, and their information should NOT be merged. 
#
# The goal of this operation is too have a more concise document containing information from both documents if both documents share overlap. Remember, ONLY MERGE DOCUMENTS IF THEY HAVE SUBSTANTIAL OVERLAP (i.e. they talk about the same subject). When you merge documents, make sure that the merged document makes sense and flows naturally. It is very important that no information is lost in the merge process, because then information about the user will be lost. It is possible that the two documents can contradict each other, and in that case you should default to using the information in Document1 over the information in Document2, because it is the more recent document.
#
#
# Your input is provided below:
# -------------------------
# Document1: {doc1}
# Document2: {doc2}
# -------------------------
#
# The output format should be:
# -------------------------
# {{"YES" if the documents should be merged. "NO" if the documents shouldn't be merged}}
# {{the merged document text if the text above was "YES"}}
# -------------------------
#
# It is extremely important that you follow this output format and do not deviate from it. Do not output any extra new lines or other extraneous characters. As a final reminder DO NOT MERGE DOCUMENTS IF THEY AREN'T ABOUT THE SAME TOPIC.
# """

user_profile_prompt = """
Generate a user profile based on all of the stored information about the user. Be sure to give a detailed description based on the available history, 
inluding things such as what they like, what their hobbies are, what their aspirations are, etc. if that information is available.
"""

class HistoryNode:
  def __init__(self, content, prev=None, next=None):
    self.content = content
    self.prev = prev if prev is not None else []
    self.next = next if next is not None else []

class HistoryGraph:
  def __init__(self):
    self.nodes = {}

  def add_node(self, node):
    self.nodes[node.content.doc_id] = node

  def remove_node(self, node):
    del self.nodes[node.content.doc_id]
class MemoryBot:
    def __init__(self, chunk_size=256, chunk_overlap=20):
        self.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
        self.llm = OpenAI(model="gpt-4o")
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        self.index = VectorStoreIndex([])
        self.text_splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.history = HistoryGraph()
        self.chat_engine = self.index.as_chat_engine(chat_mode="best")
        self.original_documents = {}

    def add_document(self, content):
        retriever = self.index.as_retriever(similarity_top_k=1)
        initial_doc = Document(text=content)
        self.history.add_node(HistoryNode(content=initial_doc))
        self.original_documents[content] = initial_doc.doc_id
        nodes = retriever.retrieve(content)
        if(len(nodes) == 0):
          self.index.insert_nodes(self.text_splitter.get_nodes_from_documents([initial_doc]))
          return initial_doc.doc_id

        top_node_document_id = nodes[0].node.ref_doc_id
        
        all_nodes_for_document = self.index.docstore.get_nodes(self.index.ref_doc_info[top_node_document_id].node_ids)
        full_text = "".join([node.text for node in all_nodes_for_document])

        print(merge_node_prompt.format(doc1=content, doc2=full_text))
        resp = OpenAI().complete(merge_node_prompt.format(doc1=content, doc2=full_text))

        if("YES" in resp.text):
          print("------------")
          print(resp.text)
          print("------------")
          lines = resp.text.split('\n', 1)
          doc = Document(text=lines[1])
          self.index.delete_ref_doc(top_node_document_id, delete_from_docstore=True)

          print(initial_doc.doc_id)
          print(top_node_document_id)
          print(self.history.nodes[initial_doc.doc_id].next)
          print(self.history.nodes[top_node_document_id].next)
          self.history.nodes[initial_doc.doc_id].next.append(doc.doc_id)
          self.history.nodes[top_node_document_id].next.append(doc.doc_id)

          print(self.history.nodes[initial_doc.doc_id].next)
          print(self.history.nodes[top_node_document_id].next)
          self.history.add_node(HistoryNode(content=doc, prev=[initial_doc.doc_id, top_node_document_id]))
          self.index.insert_nodes(self.text_splitter.get_nodes_from_documents([doc]))
        else:
          self.index.insert_nodes(self.text_splitter.get_nodes_from_documents([initial_doc]))
        self.chat_engine = self.index.as_chat_engine(chat_mode="best", memory=self.chat_engine.memory)
        return initial_doc.doc_id

    def send_query(self, query):
        response = self.chat_engine.chat(query)
        return response.response

    def add_conversation(self, conversation):
        # print("YOOOOOOOOOO")
        # print(len(memory_to_conversation(conversation)))
        return self.add_document(memory_to_conversation(conversation))

    def search(self, query, top_k=5):
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)
        print([node.text for node in nodes])
        return [node.text for node in nodes]

    def start_new_conversation(self):
        doc_id = self.add_conversation(self.chat_engine.memory)
        print("THE DOC_ID: ", doc_id)
        self.chat_engine = self.index.as_chat_engine(chat_mode="best", memory=self.chat_engine.memory)
        return doc_id

    def delete_document_id(self, document_id):
        next_nodes = self.history.nodes[document_id].next
        self.index.delete_ref_doc(document_id, delete_from_docstore=True)
        if(len(next_nodes) == 0):
          return

        self.index.delete_ref_doc(self.history.nodes[next_nodes[0]].content.doc_id, delete_from_docstore=True)

        if(self.history.nodes[self.history.nodes[next_nodes[0]].prev[0]].content.doc_id != document_id):
          sibling_node = self.history.nodes[self.history.nodes[next_nodes[0]].prev[0]]
        else:
          sibling_node = self.history.nodes[self.history.nodes[next_nodes[0]].prev[1]]
        sibling_node.next = copy.deepcopy(self.history.nodes[next_nodes[0]].next)
        self.index.insert_nodes(self.text_splitter.get_nodes_from_documents([sibling_node.content]))
                              
        self.history.remove_node(self.history.nodes[document_id])
        self.history.remove_node(self.history.nodes[next_nodes[0]])

        while(len(sibling_node.next) > 0):
          if(self.history.nodes[self.history.nodes[sibling_node.next[0]].prev[0]].content.doc_id != document_id):
            cur_sibling = self.history.nodes[self.history.nodes[sibling_node.next[0]].prev[0]]
          else:
            cur_sibling = self.history.nodes[self.history.nodes[sibling_node.next[0]].prev[1]]
          resp = OpenAI().complete(merge_node_prompt.format(doc1=sibling_node.content, doc2=cur_sibling.content))
          if("YES" in resp.text):
            lines = resp.text.split('\n', 1)
            self.history.nodes[sibling_node.next[0]].content.text = lines[1]
            self.index.update_ref_doc(
                self.history.nodes[sibling_node.next[0]].content,
            )
            sibling_node = self.history.nodes[sibling_node.next[0]]
          else:
            self.index.delete_ref_doc(self.history.nodes[sibling_node.next[0]].content.doc_id, delete_from_docstore=True)
            sibling_node.next = []
            cur_sibling.next = copy.deepcopy(self.history.nodes[cur_sibling.next[0]].next)
            self.history.remove_node(self.history.nodes[sibling_node.next[0]])
            sibling_node = self.history.nodes[cur_sibling.next[0]]
        self.chat_engine = self.index.as_chat_engine(chat_mode="best", memory=self.chat_engine.memory)

    def delete_document(self, document_text):
        document_id = self.original_documents[document_text]
        self.delete_document_id(document_id)

    def generate_user_profile(self):                   
        qe = self.index.as_query_engine(response_mode="tree_summarize")
        response = qe.query(user_profile_prompt)
        print(response)
        print(response.response)
        print(self.index.docstore)
        return response.response

def memory_to_conversation(memory: ChatMemoryBuffer) -> str:
    conversation = ""
    for message in memory.get_all():
        role = message.role.capitalize()
        content = message.content
        conversation += f"{role}: {content}\n\n"
    return conversation.strip()

