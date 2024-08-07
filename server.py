from flask import Flask, render_template, request 
from flask_cors import CORS
from memory import *
app = Flask(__name__)
CORS(app)

bot = MemoryBot()
@app.route('/add_document', methods=['POST'])
def add_document():
    req = request.json
    doc_id = bot.add_document(req['document'])
    return doc_id, 200

@app.route('/chat', methods=['POST'])
def chat():
    req = request.json
    return bot.send_query(req['query']), 200

@app.route('/search', methods=['POST'])
def search():
    req = request.json
    return bot.search(req['query'], 1);

@app.route('/new_conversation', methods=['POST'])
def new_conversation():
    doc_id = bot.start_new_conversation()
    return doc_id, 200

@app.route('/delete_document', methods=['POST'])
def delete_document():
    req = request.json
    print(req['id'])
    bot.delete_document_id(req['id'])
    print(bot.index.docstore.docs)
    return "success", 200

@app.route('/get_profile', methods=['POST'])
def get_profile():
    return bot.generate_user_profile(), 200




if __name__ == '__main__':
    app.run(debug=True)
