from llama_index.core import VectorStoreIndex, Document, ServiceContext, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.memory import ChatMemoryBuffer

import os
import copy
import json

# get openai key from json
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

user_profile_prompt = """
Generate a user profile based on all of the stored information about the user. Be sure to give a detailed description based on the available history, 
inluding things such as what they like, what their hobbies are, what their aspirations are, etc. if that information is available.
"""

# class to hold nodes in the document graph
class HistoryNode:
  def __init__(self, content, prev=None, next=None):
    self.content = content
    self.prev = prev if prev is not None else []
    self.next = next if next is not None else []

# document graph data structure
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
        # get the closest document for comparison
        retriever = self.index.as_retriever(similarity_top_k=1)
        initial_doc = Document(text=content)
        self.history.add_node(HistoryNode(content=initial_doc))

        # add to a list of documents that the user added (doesn't include merged docs)
        self.original_documents[content] = initial_doc.doc_id
        nodes = retriever.retrieve(content)
        if(len(nodes) == 0):
          self.index.insert_nodes(self.text_splitter.get_nodes_from_documents([initial_doc]))
          return initial_doc.doc_id

        top_node_document_id = nodes[0].node.ref_doc_id
       
        # get the full text of the closest document
        all_nodes_for_document = self.index.docstore.get_nodes(self.index.ref_doc_info[top_node_document_id].node_ids)
        full_text = "".join([node.text for node in all_nodes_for_document])
        print(merge_node_prompt.format(doc1=content, doc2=full_text))

        # ask LLM if the documents should be merged
        resp = OpenAI().complete(merge_node_prompt.format(doc1=content, doc2=full_text))

        if("YES" in resp.text):
          lines = resp.text.split('\n', 1)
          doc = Document(text=lines[1])

          # Delete the other document which will be merged from the index
          self.index.delete_ref_doc(top_node_document_id, delete_from_docstore=True)

        # Define the successor node relationships to previous
          self.history.nodes[initial_doc.doc_id].next.append(doc.doc_id)
          self.history.nodes[top_node_document_id].next.append(doc.doc_id)

          self.history.add_node(HistoryNode(content=doc, prev=[initial_doc.doc_id, top_node_document_id]))
          self.index.insert_nodes(self.text_splitter.get_nodes_from_documents([doc]))
        else:
          self.index.insert_nodes(self.text_splitter.get_nodes_from_documents([initial_doc]))
          # update chat engine with new memory
        self.chat_engine = self.index.as_chat_engine(chat_mode="best", memory=self.chat_engine.memory)
        return initial_doc.doc_id

    def send_query(self, query):
        response = self.chat_engine.chat(query)
        return response.response

    def add_conversation(self, conversation):
        return self.add_document(memory_to_conversation(conversation))

    def search(self, query, top_k=5):
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)
        print([node.text for node in nodes])
        return [node.text for node in nodes]

    def start_new_conversation(self):
        doc_id = self.add_conversation(self.chat_engine.memory)
        self.chat_engine = self.index.as_chat_engine(chat_mode="best", memory=self.chat_engine.memory)
        return doc_id

    def delete_document_id(self, document_id):
        next_nodes = self.history.nodes[document_id].next
        self.index.delete_ref_doc(document_id, delete_from_docstore=True)

        # if document is standalone (hasn't been merged) we are done
        if(len(next_nodes) == 0):
          return

        # delete successor merged document
        self.index.delete_ref_doc(self.history.nodes[next_nodes[0]].content.doc_id, delete_from_docstore=True)

        # get sibling document which was merged with original document
        if(self.history.nodes[self.history.nodes[next_nodes[0]].prev[0]].content.doc_id != document_id):
          sibling_node = self.history.nodes[self.history.nodes[next_nodes[0]].prev[0]]
        else:
          sibling_node = self.history.nodes[self.history.nodes[next_nodes[0]].prev[1]]

        # sibling node moves up and takes place of merged node
        sibling_node.next = copy.deepcopy(self.history.nodes[next_nodes[0]].next)

        # reinsert sibling node since it was taken out of index when merge occurred
        self.index.insert_nodes(self.text_splitter.get_nodes_from_documents([sibling_node.content]))
        
        # remove document and merged document from graph data structure
        self.history.remove_node(self.history.nodes[document_id])
        self.history.remove_node(self.history.nodes[next_nodes[0]])

        # deal with 2nd degree merges (nodes that were made from merging the original merged document)
        while(len(sibling_node.next) > 0):
          if(self.history.nodes[self.history.nodes[sibling_node.next[0]].prev[0]].content.doc_id != document_id):
            cur_sibling = self.history.nodes[self.history.nodes[sibling_node.next[0]].prev[0]]
          else:
            cur_sibling = self.history.nodes[self.history.nodes[sibling_node.next[0]].prev[1]]
        # redo the merge with the sibling node
          resp = OpenAI().complete(merge_node_prompt.format(doc1=sibling_node.content, doc2=cur_sibling.content))
         # If nodes should be merged, then update the merged node that exists in the graph with the new content 
          if("YES" in resp.text):
            lines = resp.text.split('\n', 1)
            self.history.nodes[sibling_node.next[0]].content.text = lines[1]
            self.index.update_ref_doc(
                self.history.nodes[sibling_node.next[0]].content,
            )
            sibling_node = self.history.nodes[sibling_node.next[0]]
          else:
              # If now the two documents shouldn't be merged, split them up
            self.index.delete_ref_doc(self.history.nodes[sibling_node.next[0]].content.doc_id, delete_from_docstore=True)
            # move the sibling node into it's own branch
            sibling_node.next = []
            cur_sibling.next = copy.deepcopy(self.history.nodes[cur_sibling.next[0]].next)
            self.history.remove_node(self.history.nodes[sibling_node.next[0]])
            sibling_node = self.history.nodes[cur_sibling.next[0]]
        self.chat_engine = self.index.as_chat_engine(chat_mode="best", memory=self.chat_engine.memory)

    def delete_document(self, document_text):
        document_id = self.original_documents[document_text]
        self.delete_document_id(document_id)

    def generate_user_profile(self):
        # set query engine to mode that collapses all nodes
        qe = self.index.as_query_engine(response_mode="tree_summarize")
        # ask to generate user profile
        response = qe.query(user_profile_prompt)
        return response.response

# turn llama_index ChatMemoryBuffer object into string
def memory_to_conversation(memory: ChatMemoryBuffer) -> str:
    conversation = ""
    for message in memory.get_all():
        role = message.role.capitalize()
        content = message.content
        conversation += f"{role}: {content}\n\n"
    return conversation.strip()

