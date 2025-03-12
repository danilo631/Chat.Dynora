import json
import os
import numpy as np
import random
import re
import requests
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import RSLPStemmer
from nltk.corpus import wordnet
import nltk
import wikipedia

# Configuração inicial
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)

nltk.download('rslp', download_dir=nltk_data_dir)
nltk.download('wordnet', download_dir=nltk_data_dir)
nltk.download('omw-1.4', download_dir=nltk_data_dir)

nltk.data.path.append(nltk_data_dir)

wikipedia.set_lang("pt")


class AdvancedChatbot:
    def __init__(self):
        print("Iniciando DYNORA AI 2.0...")
        self.database = self.load_database()
        self.stemmer = RSLPStemmer()
        self.vectorizer = TfidfVectorizer()
        self.learning_rate = 0.01
        self.bias = 0.1

        # Inicializando pesos e camadas ocultas
        self.hidden_layer_size = 8
        self.input_size = len(self._get_all_questions())
        self.output_size = 1
        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_layer_size)
        self.weights_hidden_output = np.random.rand(self.hidden_layer_size, self.output_size)

        self.tfidf_matrix = self._build_knowledge_base()

        print("✅ DYNORA 2.0 pronta para uso!")

    def load_database(self):
        if not os.path.exists('knowledge_base.json'):
            print("Nenhum banco de dados encontrado. Criando um novo...")
            return {"categories": {}, "interaction_history": []}
        with open('knowledge_base.json', 'r', encoding='utf-8') as f:
            print("Base de conhecimento carregada.")
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
        # Verificar operações matemáticas
        if self._is_math_expression(input_text):
            resultado = self._evaluate_math_expression(input_text)
            return f"O resultado da sua conta é: {resultado}"

        # Verificar busca no dicionário
        if "defina" in input_text or "o que significa" in input_text:
            termo = self._extrair_termo_definicao(input_text)
            definicao = self._buscar_definicao_dicionario(termo)
            if definicao:
                return f"{termo}: {definicao}"

        # Busca de informações na Wikipédia
        if "explique" in input_text or "o que é" in input_text:
            termo = self._extrair_termo_definicao(input_text)
            resumo_wiki = self._buscar_wikipedia(termo)
            if resumo_wiki:
                return resumo_wiki

        processed_input = self._preprocess(input_text)
        input_vec = self.vectorizer.transform([processed_input])
        similarities = cosine_similarity(input_vec, self.tfidf_matrix).flatten()

        best_match_idx = similarities.argmax()
        confidence = similarities[best_match_idx]

        if confidence > 0.7:
            resposta = self._get_answer(best_match_idx)
            self._update_weights(best_match_idx, confidence)
            self._save_interaction(input_text, resposta, confidence)
            return resposta
        else:
            resposta_gerada = self._generate_new_answer(input_text)
            self._save_interaction(input_text, resposta_gerada, confidence=0)
            return resposta_gerada

    # ===== Funções Adicionais =====

    def _is_math_expression(self, text):
        return bool(re.search(r'\d+\s*[\+\-\*\/]\s*\d+', text))

    def _evaluate_math_expression(self, expression):
        try:
            # Remove tudo que não for número ou operador
            expression = re.sub(r'[^\d\+\-\*\/\.\(\)\s]', '', expression)
            return eval(expression)
        except Exception:
            return "Não consegui calcular isso."

    def _extrair_termo_definicao(self, frase):
        palavras = frase.split()
        if "defina" in palavras:
            index = palavras.index("defina") + 1
        elif "explique" in palavras:
            index = palavras.index("explique") + 1
        elif "significa" in palavras:
            index = palavras.index("significa") + 1
        else:
            return frase
        return ' '.join(palavras[index:])

    def _buscar_definicao_dicionario(self, termo):
        try:
            url = f"https://api.dictionaryapi.dev/api/v2/entries/pt/{termo}"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                definicao = data[0]['meanings'][0]['definitions'][0]['definition']
                return definicao
            else:
                return "Não encontrei definição no dicionário."
        except Exception:
            return "Erro ao buscar definição."

    def _buscar_wikipedia(self, termo):
        try:
            summary = wikipedia.summary(termo, sentences=2)
            return f"De acordo com a Wikipédia: {summary}"
        except wikipedia.exceptions.DisambiguationError as e:
            return f"O termo é ambíguo, seja mais específico: {e.options[:5]}"
        except wikipedia.exceptions.PageError:
            return "Não encontrei informações na Wikipédia."
        except Exception:
            return "Erro na busca da Wikipédia."

    # ===== Aprendizado e Memória =====

    def _get_answer(self, idx):
        perguntas = self._get_all_questions()
        pergunta = perguntas[idx]

        for category in self.database['categories'].values():
            for entry in category:
                if entry['pergunta'] == pergunta:
                    return entry['resposta']
        return "Desculpe, não encontrei uma resposta precisa."

    def _update_weights(self, idx, confidence):
        adjustment = self.learning_rate * (1 - confidence)
        self.weights_input_hidden[idx % self.input_size] += adjustment
        self.weights_hidden_output += adjustment
        self.bias += self.learning_rate * (confidence - 0.5)

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

    def _save_interaction(self, pergunta, resposta, confidence):
        new_entry = {
            "pergunta": pergunta,
            "resposta": resposta,
            "timestamp": datetime.now().isoformat(),
            "confidence": confidence,
            "source": "generated" if confidence < 0.7 else "known"
        }

        if confidence >= 0.7:
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
