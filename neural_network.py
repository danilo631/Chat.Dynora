import json
import os
import numpy as np
import random
import re
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import RSLPStemmer
from nltk.corpus import wordnet
import nltk
import wikipedia
import wikipedia.exceptions
import requests
from bs4 import BeautifulSoup
from googletrans import Translator
from sympy import sympify, SympifyError
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from threading import Thread

# Preparação de dados para o ambiente NLTK
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)

nltk.download('rslp', download_dir=nltk_data_dir)
nltk.download('wordnet', download_dir=nltk_data_dir)
nltk.download('omw-1.4', download_dir=nltk_data_dir)

nltk.data.path.append(nltk_data_dir)

# Configuração da Wikipedia em português
wikipedia.set_lang('pt')

# Tradutor
translator = Translator()

# Rede Neural com PyTorch
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.softmax(out)
        return out

class AdvancedChatbot:
    def __init__(self):
        print("Iniciando DYNORA AI com recursos avançados...")
        self.database = self.load_database()
        self.stemmer = RSLPStemmer()
        self.vectorizer = TfidfVectorizer()
        self.learning_rate = 0.01
        self.bias = 0.1
        self.weights = np.random.rand(max(1, len(self._get_all_questions())))
        self.tfidf_matrix = self._build_knowledge_base()
        self.cache = {}
        self.tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
        self.bert_model = BertModel.from_pretrained('neuralmind/bert-base-portuguese-cased')
        self.neural_net = NeuralNet(input_size=768, hidden_size=512, output_size=5)  # 5 classes de intenções
        self.optimizer = optim.AdamW(self.neural_net.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        print("✅ DYNORA AI pronta para interação!")

    def load_database(self):
        if not os.path.exists('knowledge_base.json'):
            print("Nenhum banco de dados encontrado. Criando um novo...")
            return {"categories": {}, "interaction_history": []}
        with open('knowledge_base.json', 'r', encoding='utf-8') as f:
            print("Base de conhecimento carregada com sucesso.")
            return json.load(f)

    def _preprocess(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()
        expanded_words = []
        for word in words:
            expanded_words.append(word)
            expanded_words.extend(self.get_synonyms(word))
        return ' '.join([self.stemmer.stem(word) for word in expanded_words])

    def get_synonyms(self, word):
        if word in self.cache:
            return self.cache[word]
        synonyms = set()
        for syn in wordnet.synsets(word, lang='por'):
            for lemma in syn.lemmas('por'):
                synonyms.add(lemma.name().replace('_', ' '))
        self.cache[word] = list(synonyms)
        return self.cache[word]

    def _build_knowledge_base(self):
        all_questions = self._get_all_questions(preprocessed=True)
        if not all_questions:
            return self.vectorizer.fit_transform(["vazio"])
        return self.vectorizer.fit_transform(all_questions)

    def _get_all_questions(self, preprocessed=False):
        questions = []
        for category in self.database['categories'].values():
            for entry in category:
                original = entry['pergunta']
                processed = self._preprocess(original)
                questions.append(processed if preprocessed else original)
        return questions

    def predict(self, input_text):
        # Verifica se a pergunta está em outro idioma e traduz
        input_text = self._translate_input(input_text)

        # Verifica se é uma operação matemática
        math_result = self._verificar_operacao_matematica(input_text)
        if math_result is not None:
            return math_result

        # Classifica a intenção da pergunta
        intent = self._classificar_intencao(input_text)
        print(f"[DYNORA] Intenção detectada: {intent}")

        # Processa a pergunta com base na intenção
        if intent == "pesquisa":
            return self._processar_pesquisa(input_text)
        elif intent == "matematica":
            return self._processar_matematica(input_text)
        elif intent == "traducao":
            return self._traduzir_texto(input_text)
        elif intent == "resumo":
            return self._resumir_texto_longo(input_text)
        elif intent == "codigo":
            return self._gerar_codigo(input_text)
        else:
            return self._fallback_response(input_text)

    def _classificar_intencao(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        outputs = self.bert_model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        prediction = self.neural_net(cls_embedding)
        intent_idx = torch.argmax(prediction, dim=1).item()
        intents = ["pesquisa", "matematica", "traducao", "resumo", "codigo"]
        return intents[intent_idx]

    def _processar_pesquisa(self, input_text):
        resposta_wiki = self.buscar_na_wikipedia_aprimorada(input_text)
        if resposta_wiki:
            print("[DYNORA] Resposta fornecida pela Wikipedia.")
            resposta_resumida = self._resumir_texto(resposta_wiki)
            self._save_interaction(input_text, resposta_resumida, confidence=0.6, fonte="Wikipedia")
            return resposta_resumida

        resposta_duckduckgo = self.buscar_duckduckgo(input_text)
        if resposta_duckduckgo:
            print("[DYNORA] Resposta fornecida pelo DuckDuckGo.")
            resposta_resumida = self._resumir_texto(resposta_duckduckgo)
            self._save_interaction(input_text, resposta_resumida, confidence=0.6, fonte="DuckDuckGo")
            return resposta_resumida

        resposta_google = self.buscar_google(input_text)
        if resposta_google:
            print("[DYNORA] Resposta fornecida pelo Google.")
            resposta_resumida = self._resumir_texto(resposta_google)
            self._save_interaction(input_text, resposta_resumida, confidence=0.6, fonte="Google")
            return resposta_resumida

        resposta_gerada = self._generate_new_answer(input_text)
        self._save_interaction(input_text, resposta_gerada, confidence=0)
        return resposta_gerada

    def _processar_matematica(self, input_text):
        try:
            if any(op in input_text for op in ['+', '-', '*', '/', '^', 'raiz', 'sqrt']):
                texto = input_text.replace('vezes', '*').replace('x', '*').replace('dividido por', '/').replace('raiz quadrada de', 'sqrt')
                resultado = sympify(texto)
                print(f"[DYNORA] Realizando operação: {texto} = {resultado}")
                return f"O resultado da operação é {resultado}."
        except SympifyError:
            pass
        return "Não consegui resolver a operação matemática."

    def _traduzir_texto(self, texto):
        try:
            partes = texto.split("traduzir")
            if len(partes) > 1:
                texto_para_traduzir = partes[1].strip()
                traduzido = translator.translate(texto_para_traduzir, dest='en').text
                return f"Tradução: {traduzido}"
            return "Por favor, especifique o texto a ser traduzido."
        except Exception as ex:
            print(f"[DYNORA] Erro ao traduzir: {ex}")
            return "Não consegui traduzir o texto."

    def _resumir_texto_longo(self, texto):
        try:
            partes = texto.split("resumir")
            if len(partes) > 1:
                texto_para_resumir = partes[1].strip()
                return self._resumir_texto(texto_para_resumir)
            return "Por favor, especifique o texto a ser resumido."
        except Exception as ex:
            print(f"[DYNORA] Erro ao resumir: {ex}")
            return "Não consegui resumir o texto."

    def _gerar_codigo(self, texto):
        try:
            if "python" in texto.lower():
                return "Aqui está um exemplo de código Python:\n\n```python\nprint('Olá, mundo!')\n```"
            elif "javascript" in texto.lower():
                return "Aqui está um exemplo de código JavaScript:\n\n```javascript\nconsole.log('Olá, mundo!');\n```"
            elif "html" in texto.lower():
                return "Aqui está um exemplo de código HTML:\n\n```html\n<h1>Olá, mundo!</h1>\n```"
            return "Posso gerar códigos em Python, JavaScript e HTML. Peça algo como 'gerar código Python'."
        except Exception as ex:
            print(f"[DYNORA] Erro ao gerar código: {ex}")
            return "Não consegui gerar o código."

    def _verificar_operacao_matematica(self, texto):
        try:
            if any(op in texto for op in ['+', '-', '*', '/', '^', 'raiz', 'sqrt']):
                texto = texto.replace('vezes', '*').replace('x', '*').replace('dividido por', '/').replace('raiz quadrada de', 'sqrt')
                resultado = sympify(texto)
                print(f"[DYNORA] Realizando operação: {texto} = {resultado}")
                return f"O resultado da operação é {resultado}."
        except SympifyError:
            pass
        return None

    def _generate_new_answer(self, input_text):
        context = self._find_best_context(input_text)
        related_answers = self._get_related_answers(context)

        if related_answers:
            resumo = self._synthesize_answer(related_answers, context)
            return resumo

        return self._fallback_response(input_text)

    def _synthesize_answer(self, related_answers, context):
        texto_completo = ' '.join(set(related_answers))
        resumo = self._resumir_texto(texto_completo)

        response_template = random.choice([
            f"Baseado no contexto '{context}', aqui está um resumo:",
            f"Considerei informações sobre '{context}':",
            "De acordo com o que encontrei:"
        ])
        return f"{response_template} {resumo}"

    def _resumir_texto(self, texto):
        sentencas = list(set(re.split(r'[.!?]', texto)))
        resumo = '. '.join([s.strip() for s in sentencas if len(s.strip()) > 0][:3])
        return resumo + '.'

    def _save_interaction(self, pergunta, resposta, confidence, fonte="generated"):
        new_entry = {
            "pergunta": pergunta,
            "resposta": resposta,
            "timestamp": datetime.now().isoformat(),
            "confidence": confidence,
            "source": fonte
        }

        if confidence >= 0.7 or fonte in ["Wikipedia", "DuckDuckGo", "Google"]:
            self._add_to_database(pergunta, resposta)

        self.database["interaction_history"].append(new_entry)
        self._save_database()
        self.tfidf_matrix = self._build_knowledge_base()

    def _add_to_database(self, pergunta, resposta):
        if "geral" not in self.database["categories"]:
            self.database["categories"]["geral"] = []

        if any(entry['pergunta'] == pergunta for entry in self.database["categories"]["geral"]):
            return

        self.database["categories"]["geral"].append({
            "pergunta": pergunta,
            "resposta": resposta
        })

        # Expandir neurônios quando a base cresce
        self._expand_neurons()

    def _save_database(self):
        with open('knowledge_base.json', 'w', encoding='utf-8') as f:
            json.dump(self.database, f, indent=2, ensure_ascii=False)

    def _find_best_context(self, input_text):
        processed_input = self._preprocess(input_text)
        input_vec = self.vectorizer.transform([processed_input])
        similarities = cosine_similarity(input_vec, self.tfidf_matrix).flatten()

        if np.max(similarities) > 0.4:
            best_match_idx = np.argmax(similarities)
            return self._get_all_questions()[best_match_idx]
        return "informações gerais"

    def _get_related_answers(self, context):
        related_answers = []
        for category in self.database['categories'].values():
            for entry in category:
                if context and context in self._preprocess(entry['pergunta']):
                    related_answers.append(entry['resposta'])
        return related_answers

    def _fallback_response(self, input_text):
        fallback_respostas = [
            "Desculpe, ainda estou aprendendo sobre isso. Pode reformular?",
            "Ótima pergunta! Vou estudar para te dar uma resposta melhor depois.",
            "Não entendi completamente. Pode explicar de outro jeito?"
        ]
        return random.choice(fallback_respostas)

    def train_on_feedback(self, pergunta, resposta, positivo=True):
        if positivo:
            self._add_to_database(pergunta, resposta)
            print(f"[DYNORA TREINO] Aprendi com a pergunta: {pergunta}")
        else:
            print(f"[DYNORA TREINO] Feedback negativo recebido. Nenhuma alteração feita.")

# Exemplo de uso
if __name__ == "__main__":
    chatbot = AdvancedChatbot()
    while True:
        user_input = input("Você: ")
        if user_input.lower() in ["sair", "exit"]:
            break
        response = chatbot.predict(user_input)
        print(f"DYNORA: {response}")
