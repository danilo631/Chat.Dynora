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

# Inicializa o chatbot
chatbot = AdvancedChatbot()

@app.route('/')
def index():
    """Rota principal que renderiza a página inicial."""
    return render_template('index.html')

@app.route('/admin')
def admin():
    """Rota para a interface de administração."""
    return render_template('admin.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    """Rota para obter uma resposta do chatbot."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Nenhum dado recebido!', 'status': 'error'}), 400

        user_message = data.get('message', '').strip()
        if not user_message:
            return jsonify({'error': 'Mensagem vazia!', 'status': 'error'}), 400

        response = chatbot.predict(user_message)
        return jsonify({'response': response, 'status': 'success'})

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/admin/list', methods=['GET'])
def list_database():
    """Rota para listar todo o banco de dados."""
    try:
        return jsonify(chatbot.database)
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/admin/add', methods=['POST'])
def add_entry():
    """Rota para adicionar uma nova entrada ao banco de dados."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Nenhum dado recebido!', 'status': 'error'}), 400

        categoria = data.get('categoria', 'geral')
        pergunta = data.get('pergunta', '').strip()
        resposta = data.get('resposta', '').strip()

        if not pergunta or not resposta:
            return jsonify({'error': 'Campos obrigatórios faltando!', 'status': 'error'}), 400

        if categoria not in chatbot.database['categories']:
            chatbot.database['categories'][categoria] = []

        # Verifica se a pergunta já existe
        if any(entry['pergunta'] == pergunta for entry in chatbot.database['categories'][categoria]):
            return jsonify({'error': 'Pergunta já existe!', 'status': 'error'}), 400

        chatbot.database['categories'][categoria].append({
            'pergunta': pergunta,
            'resposta': resposta
        })

        chatbot._save_database()
        chatbot.tfidf_matrix = chatbot._build_knowledge_base()

        return jsonify({'status': 'success', 'message': 'Entrada adicionada com sucesso!'})

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/admin/remove', methods=['POST'])
def remove_entry():
    """Rota para remover uma entrada do banco de dados."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Nenhum dado recebido!', 'status': 'error'}), 400

        categoria = data.get('categoria', 'geral')
        pergunta = data.get('pergunta', '').strip()

        if not pergunta:
            return jsonify({'error': 'Pergunta não fornecida!', 'status': 'error'}), 400

        if categoria not in chatbot.database['categories']:
            return jsonify({'error': f"Categoria '{categoria}' não encontrada!", 'status': 'error'}), 404

        # Remove a entrada se existir
        entries = chatbot.database['categories'][categoria]
        initial_length = len(entries)
        chatbot.database['categories'][categoria] = [entry for entry in entries if entry['pergunta'] != pergunta]

        if len(chatbot.database['categories'][categoria]) == initial_length:
            return jsonify({'error': f"Pergunta '{pergunta}' não encontrada!", 'status': 'error'}), 404

        chatbot._save_database()
        chatbot.tfidf_matrix = chatbot._build_knowledge_base()

        return jsonify({'status': 'success', 'message': 'Entrada removida com sucesso!'})

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/admin/update', methods=['POST'])
def update_entry():
    """Rota para atualizar uma entrada no banco de dados."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Nenhum dado recebido!', 'status': 'error'}), 400

        categoria = data.get('categoria', 'geral')
        pergunta = data.get('pergunta', '').strip()
        nova_resposta = data.get('nova_resposta', '').strip()

        if not pergunta or not nova_resposta:
            return jsonify({'error': 'Campos obrigatórios faltando!', 'status': 'error'}), 400

        if categoria not in chatbot.database['categories']:
            return jsonify({'error': f"Categoria '{categoria}' não encontrada!", 'status': 'error'}), 404

        # Atualiza a entrada se existir
        for entry in chatbot.database['categories'][categoria]:
            if entry['pergunta'] == pergunta:
                entry['resposta'] = nova_resposta
                chatbot._save_database()
                chatbot.tfidf_matrix = chatbot._build_knowledge_base()
                return jsonify({'status': 'success', 'message': 'Entrada atualizada com sucesso!'})

        return jsonify({'error': f"Pergunta '{pergunta}' não encontrada!", 'status': 'error'}), 404

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
