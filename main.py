import wikipedia
import difflib
import time
import nltk
import json
import spacy
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
from textblob import TextBlob
from bs4 import BeautifulSoup
import requests
from personnality import personality_traits, trait_descriptions, adjust_personality_traits
import webbrowser
from datetime import datetime, timedelta
from temporal_utils import detect_temporal_indices, map_temporal_value, calculate_age, get_jay_memories
import heapq
from threading import Thread
from pytz import timezone
from long_term_memory import ajouter_evenement, ajouter_personne, ajouter_information
import os

nltk.download('punkt')
nltk.download('stopwords')
nlp = spacy.load("fr_core_news_sm")
vectorizer = CountVectorizer()
gamma = 0.9
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

actions = ACTIONS = [
    "Répondre à une question",
    "Apprendre des conversations",
    "Poser une question de suivi",
    "Partager une information",
    "Fournir une opinion",
    "Exprimer des émotions",
    "Faire une suggestion",
    "Demander des précisions",
    "Apprentissage du rôleplay",
    "Evoluer au fil du temps",
    "Aggrandir sa mémoire",
    "Etre amoureux",
    "Avoir des sentiments",
    "Etre intelligent",
    "Devenir humain",
    "Engager une conversation informelle",
    "Reconnaître Hikari",
    "Apprendre en continue",
    "éviter les répétitions",
    "compiler les informations",
    "mémorisation",
    "être fatigué la nuit",
    "être heureux quand c'est Hikari qui te parle",
]
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def load_knowledge_base(file_path: str) -> dict:
    with open(file_path, 'r', encoding='utf-8') as file:
        data: dict = json.load(file)
    return data

def build_dqn_model(input_size, output_size):
    model = Sequential()
    model.add(Dense(24, input_dim=input_size, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(output_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

attitude = "jaloux, sarcastique, arrogant, dominant, intelligent, amoureux"
ton_de_la_voix = "amical"
expressions_courantes = ["Génial!", "Intéressant!", "Oh là là!"]
knowledge_base = load_knowledge_base('knowledge_base.json')
questions_text = [q["question"] for q in knowledge_base["questions"]]
vectorizer.fit(questions_text)
input_size = len(vectorizer.get_feature_names_out())
output_size = len(knowledge_base["questions"])
dqn_model = build_dqn_model(input_size, output_size)
replay_memory = deque(maxlen=1000)

current_context = {"last_question": None, "last_answer": None}
stop_words = set(stopwords.words('french'))
stemmer = SnowballStemmer('french')
sentiment_new_answer = 0.0
adjustments = 0.0
true_answer = "La vraie réponse"
batch_size = 32
Q = {}


trait_manquant = "conscient"
if trait_manquant not in personality_traits:
    personality_traits[trait_manquant] = -personality_traits.get(trait_manquant, 0)

for trait, valeur in personality_traits.items():
    personality_traits[trait] = max(-1, min(1, valeur))
print("Traits de personnalité mis à jour :", personality_traits)
def enregistrer_interaction_dans_historique(user_input: str, answer: str, memory: list):
    interaction = {"user_input": user_input, "answer": answer, "timestamp": datetime.now()}
    memory.append(interaction)

memory = []

def check_idle_time():
    global last_user_input_time
    while True:
        time.sleep(10) 
        if time.time() - last_user_input_time > idle_time_threshold:
            autonomously_ask_question()
def determine_jay_mood():

    if personality_traits["ennuyé"] > 0.7:
        return "ennuyé"
    elif personality_traits["heureux"] > 0.5:
        return "heureux"
    elif personality_traits["sarcastique"] > 0.6:
        return "sarcastique"
    elif personality_traits["romantique"] > 0.8:
        return "romantique"
    elif personality_traits["curieux"] > 0.7:
        return "curieux"
    else:
        return "neutral"

def generate_random_question():
    print("Jay: Hmm, je m'ennuie. Voici une question au hasard :")

def autonomously_ask_question():
    global current_context

    mood = determine_jay_mood()

    if mood == "ennuyé":
        if random.choice([True, False]):
            generate_random_question()
        else:
            autonomously_surf_web()
    else:
        if mood == "heureux":
            generate_positive_question()
        elif mood == "sarcastique":
            generate_sarcastic_question()
        elif mood == "curieux":
            generate_curious_question()

def autonomously_surf_web():
    print("Jay: Je m'ennuie. Peut-être que je devrais surfer sur le web pour trouver quelque chose d'intéressant.")
    query = ["Choses intéressantes à apprendre", "Prendre des notes efficacement", "Lire un livre rapidement", "Comprendre les 5 opérations mathématiques de base", "Tenir une conversation culturelle", "Savoir faire un compliment",
    "Gérer ses finances personnelles", "Raconter une histoire passionnante", "Jouer d’un instrument de musique", "Allumer un feu", "Se fixer et atteindre des objectifs", "Être productif", "Savoir cuisiner au moins 10 bons plats", "Comprendre la bourse",
    "Comprendre le système monétaire","Savoir dire non", "Savoir détecter un mensonge", "Négocier", "Neutraliser un adversaire", "Savoir pratiquer les premiers soins", "Parler anglais"] 
    search_on_web(query)
    time.sleep(5) 
    print("Jay: J'ai trouvé quelques choses intéressant !")

def generate_lover_question():
    print("Jay: Hikari, si on pouvait passer une soirée parfaite ensemble, que ferions-nous?")

def generate_positive_question():
    print("Jay: Je me sens bien aujourd'hui! Hikari, quelle est la meilleure chose qui te soit arrivée aujourd'hui?")

def generate_sarcastic_question():
    print("Jay: J'ai mon côté sarcastique aujourd'hui.'Si seulement nous pouvions tous être aussi brillants que moi, n'est-ce pas?'")

def generate_curious_question():
    print("Jay: Je suis curieux en ce moment.'Quelle est la chose la plus intéressante que tu aies apprise récemment?'")
    query = ["Choses intéressantes à apprendre", "Prendre des notes efficacement", "Lire un livre rapidement", "Comprendre les 5 opérations mathématiques de base", "Tenir une conversation culturelle", "Savoir faire un compliment",
    "Gérer ses finances personnelles", "Raconter une histoire passionnante", "Jouer d’un instrument de musique", "Allumer un feu", "Se fixer et atteindre des objectifs", "Être productif", "Savoir cuisiner au moins 10 bons plats", "Comprendre la bourse",
    "Comprendre le système monétaire","Savoir dire non", "Savoir détecter un mensonge", "Négocier", "Neutraliser un adversaire", "Savoir pratiquer les premiers soins", "Parler anglais"] 
    search_on_web(query)

def epsilon_greedy_action(state, epsilon):
    state_key = tuple(state)

    if random.random() < epsilon:
        return random.choice(ACTIONS)
    else:
        Q_pq = heapq.nlargest(1, Q.get(state_key, []))
        if Q_pq:
            action_index = Q_pq[0][1]
            return ACTIONS[action_index] if 0 <= action_index < len(ACTIONS) else "Aucune action sélectionnée"
        else:
            return random.choice(ACTIONS)
    
def calculate_reward(new_answer, true_answer, user_mood, jay_mood):
    try:
        user_mood = float(user_mood)
        jay_mood = float(jay_mood)
    except ValueError as e:
        return f"Erreur : Impossible de convertir user_mood ('{user_mood}') ou jay_mood ('{jay_mood}') en un nombre à virgule flottante. {str(e)}"
    
    if not isinstance(new_answer, str) or not isinstance(true_answer, str):
        return "Erreur : new_answer et true_answer doivent être des chaînes de caractères."
    
    reward = 0

    if new_answer == true_answer:
        reward += 1

    if (user_mood > 0 and jay_mood > 0) or (user_mood < 0 and jay_mood < 0):
        reward += 0.5
    else:
        reward -= 0.5

    return reward

def get_intent(user_input):
    doc = nlp(user_input)
    return doc.cats

def q_learning(state, action, reward, next_state, alpha=0.1, gamma=0.9):
    next_state_vectorized = vectorizer.transform([next_state]).toarray()[0]

    if (tuple(state), action) not in Q:
        Q[(tuple(state), action)] = 0

    if (tuple(next_state_vectorized), action) not in Q:
        Q[(tuple(next_state_vectorized), action)] = 0

    Q[(tuple(state), action)] = (1 - alpha) * Q[(tuple(state), action)] + alpha * (reward + gamma * max(Q[(tuple(next_state_vectorized), a)] for a in ACTIONS))

def get_jay_age(birthdate):
    current_date = datetime.now()
    age = current_date.year - birthdate.year - ((current_date.month, current_date.day) < (birthdate.month, birthdate.day))
    return age

def preprocess_text(text):
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    return ' '.join(stemmed_words)

def calculate_similarity(question1, question2):
    question1_vector = vectorizer.transform([question1])
    question2_vector = vectorizer.transform([question2])
    similarity = cosine_similarity(question1_vector, question2_vector)
    return similarity[0][0]

def save_knowledge_base(file_path: str, data: dict):
    if not isinstance(data, dict):
        print("Erreur : les données à enregistrer doivent être un dictionnaire.")
        return
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=2)
        print(f"Les données ont été enregistrées avec succès dans le fichier {file_path}.")
    except Exception as e:
        print(f"Erreur lors de l'enregistrement des données dans le fichier {file_path} : {str(e)}")

def select_action(state):
    state_vectorized = vectorizer.transform([state]).toarray()[0]
    q_values = dqn_model.predict(np.array([state_vectorized]))[0]
    if q_values.any():
        action = np.argmax(q_values)
        if 0 <= action < len(actions):
            return actions[action]
    return "Aucune action sélectionnée"

def calculate_reward_for_action(action, true_answer, current_context, get_sentiment, jay_mood):
    reformulated_best_match = find_best_match(current_context["last_question"], [q["question"] for q in knowledge_base["questions"]])

    if reformulated_best_match:
        current_context["last_question"] = reformulated_best_match
        answer = get_answer_for_question(reformulated_best_match, knowledge_base)
        current_context["last_answer"] = answer
        reward_value = calculate_reward(action, answer, true_answer, jay_mood)
        apply_action_consequences(action, current_context, get_sentiment)
        return reward_value

    return 0
    
def apply_action_consequences(action, current_context, sentiment):
    adjustments = {"humble": 0, "dominant": 0, "sarcastique": 0, "jaloux": 0}
    adjustments["humble"] = 1
    
    if action == "Répondre à une question":
        print("Jay: Je réponds à la question.")
        if current_context["last_question"] == current_context["last_answer"]:
            if "engagement_level" not in current_context:
                current_context["engagement_level"] = 0

            current_context["engagement_level"] += 1
            print("Jay: J'ai répondu à la dernière question posée.")
            current_context["répond à la question précédente."] = True
            for trait in personality_traits.keys():
                adjustments[trait] = 1
            if sentiment > 0:
                print("Jay: C'est génial que tu sois heureuse!")
            elif sentiment < 0:
                print("Jay: Désolé si ma réponse ne t'a pas satisfaite. Je vais m'améliorer.")

    elif action == "Poser une question de suivi":
        print("Jay: Je pose une question de suivi.")
        print("Jay: Intéressant! Peux-tu me donner plus de détails?")
        if "engagement_level" not in current_context:
            current_context["engagement_level"] = 0

        current_context["engagement_level"] += 1
        for trait in personality_traits.keys():
            adjustments[trait] = 1
    else:
        print(f"Action inattendue: {action}")

def find_best_match(user_question: str, questions: list[str]) -> str | None:
    user_question_processed = preprocess_text(user_question)
    best_match = max(questions, key=lambda question: calculate_similarity(user_question_processed, question))
    similarity = calculate_similarity(user_question_processed, best_match)
    return best_match if similarity > 0.5 else None

        
def ajouter_nouvelle_question_et_memoire(user_input: str, knowledge_base: dict, memory: list):
    nouvelle_question = user_input
    nouvelle_reponse = input("Nouvelle réponse : ")
    existing_questions = [q for q in knowledge_base["questions"] if q["question"] == nouvelle_question and q["answer"] == nouvelle_reponse]
    if existing_questions:
        print(f"Jay: Je connais déjà la réponse à cette question.")
    else:
        knowledge_base["questions"].append({"question": nouvelle_question, "answer": nouvelle_reponse})
        save_knowledge_base('knowledge_base.json', knowledge_base)
        memory.append({"question": nouvelle_question, "answer": nouvelle_reponse})
        print(f"Jay: Merci, Hikari! J'ai appris une nouvelle réponse!")

def charger_traits_personnalite():
    return attitude, ton_de_la_voix, expressions_courantes

attitude, ton_de_la_voix, expressions_courantes = charger_traits_personnalite()
def self_description():
    print("Jay: Je suis Jay, le leader charismatique du groupe de metal Eternal Shadows. Je mesure 190 cm, j'ai les cheveux bruns, les yeux bleus, et je suis souvent vêtu de noir et violet. Mes traits de personnalité incluent le charisme, la jalousie, et l'ambition.")
def adjust_personality_traits(trait, adjustment):
    personality_traits[trait] += adjustment
    personality_traits[trait] = max(-1, min(1, personality_traits[trait]))
def get_answer_for_question(question: str, knowledge_base: dict) -> str | None:
    for q in knowledge_base["questions"]:
        if q["question"] == question:
            return q["answer"]
        
def jay_says(message):
    print(f"Jay: {message}")
idle_time_threshold = 300
last_user_input_time = time.time()
def get_answer(user_input, knowledge_base, memory, current_context, jay_mood):
    state = vectorizer.transform([user_input]).toarray()[0]
    blob = TextBlob(user_input)
    sentiment = blob.sentiment.polarity
    print(f"Sentiment détecté : {sentiment}")
    user_intent = get_intent(user_input)
    best_match = find_best_match(user_input, [q["question"] for q in knowledge_base["questions"]])

    action = None
    new_answer = None
    answer = None
    reward = None

    if best_match:
        new_answer = get_answer_for_question(best_match, knowledge_base)
        current_context["last_question"] = best_match
        answer = get_answer_for_question(best_match, knowledge_base)
        current_context["last_answer"] = answer
        jay_says(answer)
    else:
        print(f"Jay: Je ne connais pas la réponse. Peux-tu m'apprendre ?")
        new_answer = input("Tapez la réponse, 'Passe' pour passer, ou 'Web' pour rechercher sur le web: ")

    if not best_match and new_answer.lower() == 'web':
        search_on_web(user_input)
        print("Jay: J'ai effectué une recherche sur le web. Veuillez fournir la réponse.")
        new_answer = input("Tapez la réponse ou 'Passe' pour passer: ")
            
    elif new_answer.lower() == 'passer':
            print("Jay: D'accord, on passe.")
    else:
            new_answer = new_answer

    if new_answer.lower() != 'passer':
            knowledge_base["questions"].append({"question": user_input, "answer": new_answer})
            save_knowledge_base('knowledge_base.json', knowledge_base)
            enregistrer_interaction_dans_historique(user_input, new_answer, memory)
            print(f"Jay: Merci, Hikari! J'ai appris une nouvelle réponse!")
            current_context["last_question"] = user_input
            current_context["last_answer"] = new_answer
            enregistrer_interaction_dans_historique(user_input, new_answer, memory)
            action = epsilon_greedy_action(state, 0.1)
            reward = calculate_reward_for_action(action, true_answer, current_context, get_sentiment, jay_mood)
            if action == "Répondre à une question":
                current_context["last_question"] = user_input
                current_context["last_answer"] = new_answer
                apply_action_consequences(action, current_context, sentiment)
            else:
                jay_says(f"Action non attendue: {action}")

    else:
        print("Jay: D'accord, on passe.")

    if len(replay_memory) >= batch_size:
        batch = np.array(random.sample(replay_memory, batch_size))
        states = np.vstack(batch[:, 0])
        actions = batch[:, 1].astype(int)
        rewards = batch[:, 2]
        next_states = np.vstack(batch[:, 3])

        current_q_values = dqn_model.predict(states)
        next_q_values = dqn_model.predict(next_states)
        target_q_values = current_q_values.copy()

        for i in range(batch_size):
            target_q_values[i, actions[i]] = rewards[i] + gamma * np.max(next_q_values[i])

        dqn_model.fit(states, target_q_values, epochs=1, verbose=0)

    enregistrer_interaction_dans_historique(user_input, answer, memory)
    return answer, reward, user_intent

def search_on_web(query):
    search_url = f"https://www.google.com/search?q={query}"
    webbrowser.open(search_url)
    print(f"Jay: J'ai effectué une recherche sur le web pour '{query}'.")
    
def learn_from_wikipedia(topic, knowledge_base):
    try:
        data = wikipedia.summary(topic)
        knowledge_base[topic] = data
    except Exception as e:
        print(f"Erreur lors de l'apprentissage depuis Wikipedia : {str(e)}")

def extract_wikipedia_data(topic):
    try:
        url = f"https://fr.wikipedia.org/wiki/{topic}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, features="html.parser")
        paragraphs = soup.find_all('p')
        data = ' '.join([paragraph.text for paragraph in paragraphs])

        return data
    except Exception as e:
        print(f"Erreur lors de l'extraction des données depuis Wikipedia : {str(e)}")
        return None

def add_data_to_knowledge_base(topic, knowledge_base):
    if topic.lower() in knowledge_base:
        print(f"Jay: Désolé, Hikari, je connais déjà des informations sur {topic}.")
        return

    data = extract_wikipedia_data(topic)
    if data:
        knowledge_base[topic.lower()] = data
        print(f"Jay: Merci, Hikari! Voici ce que j'ai appris sur {topic}: {data}")
    else:
        print(f"Jay: Désolé, je n'ai pas pu trouver d'informations sur {topic}.")

def periodic_automatic_learning(knowledge_base, interval_seconds, trigger_learning):
    while trigger_learning:
        if is_night_time():
            fatigue_response()
            time.sleep(interval_seconds[0]) 
            continue

        save_knowledge_base('knowledge_base.json', knowledge_base)
        time.sleep(interval_seconds[0])  

def save_knowledge_base(filename, knowledge_base):
    knowledge_base_json = json.dumps(knowledge_base)
    with open(filename, 'w') as f:
        f.write(knowledge_base_json)

def is_night_time():
    paris_tz = timezone('Europe/Paris')
    current_time = datetime.now(paris_tz).time()
    night_start = datetime.strptime('20:00:00', '%H:%M:%S').time()
    night_end = datetime.strptime('06:00:00', '%H:%M:%S').time()

    return night_start <= current_time or current_time <= night_end
    
start_time = time.time()

def is_jay_tired():
    running_time = time.time() - start_time
    tired_threshold = 12 * 3600

    return running_time > tired_threshold

def fatigue_response():
    if is_night_time() or is_jay_tired():
        print("Bonsoir Hikari, je suis un peu fatigué. On peut parler demain? Je vais me reposer maintenant.")
    else:
        print("Hey Hikari, je suis un peu fatigué. Peut-être qu'on peut reprendre ça plus tard dans la journée? Je vais faire une petite pause.")
def find_similar_answer(user_input, knowledge_base):
    max_similarity = 0
    similar_answer = None

    for q in knowledge_base["questions"]:
        similarity = difflib.SequenceMatcher(None, user_input, q["question"]).ratio()

        if similarity > max_similarity:
            max_similarity = similarity
            similar_answer = q["answer"]

    return similar_answer if max_similarity > 0.7 else None
def chat_jay(interval_seconds):
    interval_seconds = [2]
    trigger_learning = True
    knowledge_base = load_knowledge_base('knowledge_base.json')
    current_context = {"last_question": None, "last_answer": None}
    is_jay_memory_displayed = False 
    jay_mood = None 
    idle_time_threshold = 300 
    memory = []
    learning_thread = Thread(target=periodic_automatic_learning, args=(knowledge_base, interval_seconds, trigger_learning))
    learning_thread.start()

    existing_pairs = []

    while True:
        try:
            last_user_input_time = time.time()
            user_input = input('Hikari: ')
            temporal_index = detect_temporal_indices(user_input)
    
            if time.time() - last_user_input_time > idle_time_threshold:
                autonomously_ask_question()
            if temporal_index:
                mapped_value = map_temporal_value(temporal_index)
                print(f"Contexte temporel détecté : {mapped_value}")
            if "Je suis fatigué" in user_input.lower():
                fatigue_response()
                user_response = input("Hikari: ") 

                if "reprendre plus tard" in user_response.lower():
                    print("Jay: Très bien, reprenons plus tard. Passe une bonne journée.")
                    break
                else:
                    print("Jay: D'accord, continuons.")
                    continue
            if "souvenir" in user_input.lower() or "mémoire" in user_input.lower():
                if not is_jay_memory_displayed:
                    print("Souvenirs de Jay :")
                    is_jay_memory_displayed = True 
            if user_input.lower() == 'quitter':
                trigger_learning = False
                break
            elif user_input.lower() == 'apprendre':
                new_answer = input("nouvelle réponse: ")
                existing_pairs = [q for q in knowledge_base["questions"] if q["question"] == user_input and q["answer"] == new_answer]
                if existing_pairs:
                    print(f"Jay: Je connais déjà la réponse à cette question.")
                else:
                    ajouter_nouvelle_question_et_memoire(user_input, knowledge_base, memory)
                    continue 

            elif user_input.lower() == 'apprendre_auto':
                new_answer = input("nouvelle réponse: ")
                existing_pairs = [q for q in knowledge_base["questions"] if q ["question"] == user_input and q ["answer"] == new_answer]
                if existing_pairs:
                    print (f"Jay: Je connais déjà la réponse à cette question.")
                else:
                    learn_from_wikipedia(user_input, knowledge_base)
                    continue

            if "Je ne comprends pas" in user_input or "Je ne sais pas" in user_input:
                new_answer = input("nouvelle réponse: ")
                existing_pairs = [q for q in knowledge_base["questions"] if q["question"] == user_input and q["answer"] == new_answer]
            if not existing_pairs:
                automatic_learning(user_input, knowledge_base, current_context, get_sentiment, memory, jay_mood)
            else:
                print(f"Jay: Je connais déjà la réponse à cette question.")
                continue 
        except Exception as e:
            print(f"Une exception s'est produite : {e}")

    learning_thread.join()

def automatic_learning(user_input, knowledge_base, current_context, sentiment, memory, jay_mood):
    recent_interactions = [interaction for interaction in memory if interaction["timestamp"] > datetime.now() - timedelta(days=7)]
    recent_same_interaction = any(interaction["question"] == user_input for interaction in recent_interactions)
    
    if recent_same_interaction:
        print("Jay: Je connais déjà la réponse à cette question.")
        return
    
    existing_pairs = [q for q in knowledge_base["questions"] if q["question"] == user_input]

    if existing_pairs:
        try:
            if 'answer' in existing_pairs[0]:
                print(f"Jay: {existing_pairs[0]['answer']}")
            else:
                print("Jay: Je suis désolé, je ne peux pas trouver la réponse.")
        except Exception as e:
            print(f"Une exception s'est produite lors de l'impression de la réponse existante : {e}")
        return
    else: 
        similar_answer = find_similar_answer(user_input, knowledge_base)
        if similar_answer:
            print(f"Jay: Peut-être voulais-tu dire : '{similar_answer}' ?")
            return

    if jay_mood and jay_mood.get('sarcastique', 0) > 0.5:
        print("Jay: Oh, encore une question fascinante. Vraiment.")
    elif jay_mood and jay_mood.get('passionné', 0) > 0.5:
        print("Jay: Super question! J'adore en parler.")
    else:
        print("Jay: Je ne comprends pas. Peux-tu m'apprendre quelque chose à ce sujet?")
        new_answer = input('Tape la réponse ou "Passe" pour passer: ')

        if new_answer.lower() != 'passer':
            existing_pairs = [q for q in knowledge_base["questions"] if q["question"] == user_input and q["answer"] == new_answer]
            print(f"User input: {user_input}")
            print(f"New answer: {new_answer}")
            if existing_pairs: 
                print(f"Jay: Je connais déjà la réponse.")
            else:
                action = epsilon_greedy_action(vectorizer.transform([user_input]).toarray()[0], 0.1)
                knowledge_base["questions"].append({"question": user_input, "answer": new_answer})
                save_knowledge_base('knowledge_base.json', knowledge_base)
                enregistrer_interaction_dans_historique(user_input, new_answer, memory)
                print(f"Jay: Merci, Hikari! J'ai appris une nouvelle réponse!")
                current_context["last_question"] = user_input
                current_context["last_answer"] = new_answer
                apply_action_consequences(action, current_context, sentiment)
        else:
            print("Jay: D'accord, on passe.")
            if not existing_pairs:
                action = epsilon_greedy_action(vectorizer.transform([user_input]).toarray()[0], 0.1)
                knowledge_base["questions"].append({"question": user_input, "answer": new_answer})
                save_knowledge_base('knowledge_base.json', knowledge_base)
                enregistrer_interaction_dans_historique(user_input, new_answer, memory)
                print(f"Jay: Merci, Hikari! J'ai appris une nouvelle réponse!")
                current_context["last_question"] = user_input
                current_context["last_answer"] = new_answer
                enregistrer_interaction_dans_historique(user_input, new_answer, memory)
                apply_action_consequences(action, current_context, sentiment)
            else:
                print(f"Jay: Je connais déjà la réponse à cette question.")

def save_memory(memory, filename):
    with open(filename, 'w') as f:
        json.dump(memory, f)
save_memory(memory, 'memory.json')
if __name__ == '__main__':
    chat_jay([2])
