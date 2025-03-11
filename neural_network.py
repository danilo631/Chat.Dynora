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

nltk.download('wordnet')
nltk.download('omw-1.4')

class AdvancedChatbot:
    def __init__(self):
        self.database = self.load_database()
        self.stemmer = RSLPStemmer()
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self._build_knowledge_base()
        self.learning_rate = 0.1
        self.weights = np.random.rand(len(self._get_all_questions()))
        print("✅ DYNORA iniciada com sucesso!")

    def load_database(self):
        if not os.path.exists('knowledge_base.json'):
            return {"categories": {}, "interaction_history": []}
        with open('knowledge_base.json', 'r', encoding='utf-8') as f:
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
        processed_input = self._preprocess(input_text)
        input_vec = self.vectorizer.transform([processed_input])
        similarities = cosine_similarity(input_vec, self.tfidf_matrix).flatten()

        best_match_idx = similarities.argmax()
        confidence = similarities[best_match_idx]

        print(f"[DYNORA] Confiança: {confidence:.2f}")

        if confidence > 0.7:
            self._update_weights(best_match_idx, confidence)
            resposta = self._get_answer(best_match_idx)
            self._save_interaction(input_text, resposta, confidence)
            return resposta

        generated_response = self._generate_new_answer(input_text)
        self._save_interaction(input_text, generated_response, confidence=0)
        return generated_response

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
        self.weights[idx] += adjustment

    def _generate_new_answer(self, input_text):
        context = self._find_best_context(input_text)
        related_answers = self._get_related_answers(context)

        if related_answers:
            return self._synthesize_answer(input_text, related_answers, context)

        return self._fallback_response(input_text)

    def _synthesize_answer(self, input_text, related_answers, context):
        response_template = random.choice([
            f"Com base no contexto '{context}':",
            "Relatando informações similares:",
            "Considerando o que sei:"
        ])
        best_answer = max(related_answers, key=lambda x: len(x))
        return f"{response_template} {best_answer} (pergunta: {input_text})"

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
        fallback_list = [
            "Desculpe, não entendi sua pergunta. Pode reformular?",
            "Ainda estou aprendendo sobre isso, poderia dar mais detalhes?",
            "Não tenho certeza sobre isso, mas posso tentar pesquisar!"
        ]
        return random.choice(fallback_list)
