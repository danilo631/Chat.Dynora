from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from neural_network import AdvancedChatbot
import os
import nltk

# Diretório onde os dados serão baixados
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)

# Faz o download dos recursos necessários
nltk.download('rslp', download_dir=nltk_data_dir)
nltk.download('wordnet', download_dir=nltk_data_dir)
nltk.download('omw-1.4', download_dir=nltk_data_dir)

# Adiciona o caminho dos dados para o NLTK
nltk.data.path.append(nltk_data_dir)

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

chatbot = AdvancedChatbot()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/admin')
def admin():
    return render_template('admin.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()

        if not user_message:
            return jsonify({'error': 'Mensagem vazia!'}), 400

        response = chatbot.predict(user_message)
        return jsonify({'response': response, 'status': 'success'})

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/admin/list', methods=['GET'])
def list_database():
    return jsonify(chatbot.database)

@app.route('/admin/add', methods=['POST'])
def add_entry():
    data = request.get_json()
    categoria = data.get('categoria', 'geral')
    pergunta = data.get('pergunta')
    resposta = data.get('resposta')

    if not pergunta or not resposta:
        return jsonify({'error': 'Campos obrigatórios!'}), 400

    if categoria not in chatbot.database['categories']:
        chatbot.database['categories'][categoria] = []

    chatbot.database['categories'][categoria].append({
        'pergunta': pergunta,
        'resposta': resposta
    })

    chatbot._save_database()
    chatbot.tfidf_matrix = chatbot._build_knowledge_base()

    return jsonify({'status': 'success', 'message': 'Entrada adicionada'})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
