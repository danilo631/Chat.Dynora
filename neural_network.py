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

# Preparação de dados para o ambiente NLTK
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)

nltk.download('rslp', download_dir=nltk_data_dir)
nltk.download('wordnet', download_dir=nltk_data_dir)
nltk.download('omw-1.4', download_dir=nltk_data_dir)

nltk.data.path.append(nltk_data_dir)

# Configuração da Wikipedia em português
wikipedia.set_lang('pt')

class AdvancedChatbot:
    def __init__(self):
        print("Iniciando DYNORA AI com recursos avançados...")
        self.database = self.load_database()
        self.stemmer = RSLPStemmer()
        self.vectorizer = TfidfVectorizer()
        self.learning_rate = 0.05
        self.bias = 0.1
        self.weights = np.random.rand(max(1, len(self._get_all_questions())))
        self.tfidf_matrix = self._build_knowledge_base()
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
        synonyms = set()
        for syn in wordnet.synsets(word, lang='por'):
            for lemma in syn.lemmas('por'):
                synonyms.add(lemma.name().replace('_', ' '))
        return list(synonyms)

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
        soma_resultado = self._verificar_soma(input_text)
        if soma_resultado is not None:
            return soma_resultado

        processed_input = self._preprocess(input_text)
        input_vec = self.vectorizer.transform([processed_input])
        similarities = cosine_similarity(input_vec, self.tfidf_matrix).flatten()

        best_match_idx = similarities.argmax()
        confidence = similarities[best_match_idx] if similarities.size > 0 else 0

        print(f"[DYNORA] Confiança na resposta: {confidence:.2f}")

        if confidence > 0.7:
            self._update_weights(best_match_idx, confidence)
            resposta = self._get_answer(best_match_idx)
            self._save_interaction(input_text, resposta, confidence)
            return resposta
        else:
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

            resposta_gerada = self._generate_new_answer(input_text)
            self._save_interaction(input_text, resposta_gerada, confidence=0)
            return resposta_gerada

    def _verificar_soma(self, texto):
        numeros = re.findall(r'\d+', texto)
        if len(numeros) >= 2 and any(op in texto for op in ['+', 'soma', 'somar', 'adicionar', 'adição']):
            numeros = list(map(int, numeros))
            resultado = sum(numeros)
            print(f"[DYNORA] Realizando soma: {numeros} = {resultado}")
            return f"O resultado da soma é {resultado}."
        return None

    def buscar_na_wikipedia_aprimorada(self, consulta):
        try:
            print(f"[DYNORA WIKI] Buscando: {consulta}")
            resultado = wikipedia.summary(consulta, sentences=5, auto_suggest=False)
            return resultado
        except wikipedia.DisambiguationError as e:
            print(f"[DYNORA WIKI] Desambiguação, tentando outras opções...")
            for option in e.options[:3]:
                try:
                    resultado = wikipedia.summary(option, sentences=5)
                    return resultado
                except:
                    continue
            return "Encontrei várias opções na Wikipédia. Pode ser mais específico?"
        except wikipedia.PageError:
            print("[DYNORA WIKI] Página não encontrada.")
            return None
        except Exception as ex:
            print(f"[DYNORA WIKI] Erro inesperado: {ex}")
            return None

    def buscar_duckduckgo(self, consulta):
        try:
            print(f"[DYNORA DUCKDUCKGO] Buscando: {consulta}")
            url = f"https://duckduckgo.com/html/?q={consulta}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            results = soup.find_all('div', class_='result__snippet')
            if results:
                return ' '.join([result.get_text() for result in results[:3]])
            return None
        except Exception as ex:
            print(f"[DYNORA DUCKDUCKGO] Erro inesperado: {ex}")
            return None

    def _get_answer(self, idx):
        perguntas = self._get_all_questions()
        pergunta = perguntas[idx] if idx < len(perguntas) else None

        for category in self.database['categories'].values():
            for entry in category:
                if entry['pergunta'] == pergunta:
                    return entry['resposta']
        return "Desculpe, não encontrei uma resposta precisa."

    def _update_weights(self, idx, confidence):
        adjustment = self.learning_rate * (1 - confidence)
        if idx >= len(self.weights):
            self._expand_neurons()
        self.weights[idx] += adjustment
        print(f"[DYNORA] Peso ajustado [{idx}]: {self.weights[idx]:.4f}")
        self.bias += self.learning_rate * (confidence - 0.5)

    def _expand_neurons(self):
        print("[DYNORA] Expandindo rede de neurônios...")
        self.weights = np.append(self.weights, np.random.rand())

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

        if confidence >= 0.7 or fonte == "Wikipedia" or fonte == "DuckDuckGo":
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
