import json
import difflib
import tensorflow as tf
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DistilBertTokenizer, TFDistilBertModel
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer
import torch

nlp = spacy.load("fr_core_news_sm")

knowledge_base = {"questions": []}
memory = []
mood_jay = "neutre"
mood_hikari = "neutre"
est_amoureux = False
longchain_agent = {"envies": ["amour", "manger"], "recherches": ["Motionless in white, Caliban"], "intérêts": ["metalband", "Fanfiction", "guitare"]}


gpt2_tokenizer = GPT2Tokenizer.from_pretrained(r"E:\New_local\Jay\models\124M")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")


try:
    with open('knowledge_base.json', 'r') as file:
        knowledge_base = json.load(file)
except FileNotFoundError:
    knowledge_base = {"questions": []}

personalities = {
    "Hikari": {"relationship_status": "amical"},
    "Viktoria": {"relationship_statuts": "ex petite amie"},
    "Alex": {"relationship_statuts": "bassiste"},
    "Neil": {"relationship_statuts": "batteur"},
    "John": {"relationship_statuts": "manageur"},
    "Min-jun": {"relationship_statuts": "père de Hikari"},
    "Greed": {"relation_statuts": "fanfiction de Hikari"}
}
def preprocess_text(text):
    return text.lower()

def find_best_match(user_input, questions):
    user_input = preprocess_text(user_input)
    questions_text = [q["question"] for q in questions]
    best_match = difflib.get_close_matches(user_input, questions_text, n=1, cutoff=0.7)
    return best_match[0] if best_match else None

def get_answer_for_question(question, knowledge_base, memory):
    question = preprocess_text(question)
    for q in memory:
        if preprocess_text(q["question"]) == question:
            return q["answer"], True

    for q in knowledge_base["questions"]:
        if preprocess_text(q["question"]) == question:
            return q["answer"], True

    return None, False

def save_knowledge_base(filename, data):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=2)

def is_answer_already_recorded(user_input, knowledge_base, memory):
    return (
        preprocess_text(user_input) in [preprocess_text(q["question"]) for q in knowledge_base["questions"]] or
        preprocess_text(user_input) in [preprocess_text(q["question"]) for q in memory]
    )

def analyze_sentiment_gpt2(text):
    inputs = gpt2_tokenizer(text, return_tensors="pt")  
    outputs = gpt2_model(**inputs)
    logits = outputs.logits

    predicted_classes = torch.argmax(logits, dim=1).numpy()

    if predicted_classes.size == 0:
        print(f"Warning: Unable to determine predicted_class: {predicted_classes}")
        return "UNKNOWN", 0.5

    sentiment_label = "UNKNOWN"
    sentiment_score = 0.5

    if isinstance(predicted_classes, int):
        predicted_class = predicted_classes
        labels = ["NEGATIVE", "POSITIVE"]
        
        if isinstance(predicted_class, int):
            sentiment_label = labels[predicted_class]
            sentiment_score = torch.nn.functional.softmax(logits, dim=1).detach().numpy()[0][predicted_class]

    return sentiment_label, sentiment_score

def recognize_person(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return None

def adjust_response_based_on_mood(response, mood_jay, mood_hikari):
    if mood_jay == "heureux" and mood_hikari == "heureux":
        return response + " C'est génial de discuter avec toi!"
    elif mood_jay == "fâché":
        return "Jay: " + "Je ne suis pas d'humeur. Laisse-moi tranquille."

    return response

def automatic_learning(user_input, new_answer, knowledge_base, memory):
    user_input = preprocess_text(user_input)
    person = recognize_person(user_input) 

    similar_question = find_best_match(user_input, knowledge_base["questions"])
    if similar_question:
        print(f"Jay: Peut-être voulais-tu dire : '{similar_question}' ?")
        return

    if is_answer_already_recorded(user_input, knowledge_base, memory):
        existing_answer, _ = get_answer_for_question(user_input, knowledge_base, memory)
        print(f"Jay: {existing_answer}")
    else:
        print("Jay: Je ne sais pas. Recherche dans ma base de données...")
        existing_answer, is_recorded = get_answer_for_question(user_input, knowledge_base, memory)
        if is_recorded:
            sentiment_label, sentiment_score = analyze_sentiment_gpt2(existing_answer)
            adjusted_response = adjust_response_based_on_mood(existing_answer, mood_jay, mood_hikari)
            print(f"Jay: {adjusted_response}")
            print(f"Sentiment de la réponse de Jay: {sentiment_label} avec un score de {sentiment_score}")
        else:
            print("Jay: Je ne trouve pas de réponse. Peux-tu m'apprendre quelque chose à ce sujet?")
            new_answer = input("nouvelle réponse: ")

            if not is_answer_already_recorded(user_input, knowledge_base, memory):
                knowledge_base["questions"].append({"question": user_input, "answer": new_answer})
                save_knowledge_base('knowledge_base.json', knowledge_base)
                memory.append({"question": user_input, "answer": new_answer})
                print(f"Jay: Merci, Hikari! J'ai appris une nouvelle réponse!")
                if person in personalities:
                    personalities[person]["relationship_status"] = "amical" 

def chat_jay():
    while True:
        try:
            user_input = input('Hikari: ')
            sentiment_label, sentiment_score = analyze_sentiment_gpt2(user_input)
            if sentiment_score > 0.5:
                mood_hikari = "heureux"
                if est_amoureux:
                    print("Jay: C'est agréable de te voir de bonne humeur, Hikari")
            elif sentiment_score < -0.5:
                mood_hikari = "fâché"
                if est_amoureux:
                    print("Jay: Pourquoi es-tu fâchée, Hikari ? Parlons-en.")
            else:
                mood_hikari = "neutre"

            if user_input.lower() == 'quitter':
                break
            elif user_input.lower() == 'apprendre':
                new_answer = input("nouvelle réponse: ")
                automatic_learning(user_input, new_answer, knowledge_base, memory)

            else:
                existing_answer, is_recorded = get_answer_for_question(user_input, knowledge_base, memory)
                if is_recorded:
                    sentiment_label, sentiment_score = analyze_sentiment_gpt2(existing_answer)
                    adjusted_response = adjust_response_based_on_mood(existing_answer, mood_jay, mood_hikari)
                    print(f"Jay: {adjusted_response}")
                    print(f"Sentiment de la réponse de Jay: {sentiment_label} avec un score de {sentiment_score}")
                else:
                    print("Jay: Je ne sais pas. Recherche dans ma base de données...")
                    existing_answer, is_recorded = get_answer_for_question(user_input, knowledge_base, memory)
                    if is_recorded:
                        sentiment_label, sentiment_score = analyze_sentiment_gpt2(existing_answer)
                        adjusted_response = adjust_response_based_on_mood(existing_answer, mood_jay, mood_hikari)
                        print(f"Jay: {adjusted_response}")
                        print(f"Sentiment de la réponse de Jay: {sentiment_label} avec un score de {sentiment_score}")
                    else:
                        print("Jay: Je ne trouve pas de réponse. Peux-tu m'apprendre quelque chose à ce sujet?")
                        new_answer = input("nouvelle réponse: ")

                        if not is_answer_already_recorded(user_input, knowledge_base, memory):
                            knowledge_base["questions"].append({"question": user_input, "answer": new_answer})
                            save_knowledge_base('knowledge_base.json', knowledge_base)
                            memory.append({"question": user_input, "answer": new_answer})
                            print(f"Jay: Merci, Hikari! J'ai appris une nouvelle réponse!")

        except KeyboardInterrupt:
            print("\nJay: Attends, tu veux vraiment quitter? Si oui, tape 'quitter'.")

def generate_response(person, mood, is_jaloux, est_amoureux, sentiment_score):
    global mood_hikari

    if sentiment_score > 0.5:
        mood_hikari = "heureux"

    if mood == 'positif':
        if person == 'Hikari':
            if est_amoureux:
                return "Jay: Salut ma belle, comment ça va?"
            elif is_jaloux:
                return "Jay: Salut Hikari, tu étais avec qui ?"
        elif person == 'Viktoria':
            return "Jay: Qu'est-ce que tu veux, Viktoria?"
        else:
            return "Jay: Salut, que puis-je faire pour toi?"
    elif mood == 'négatif':
        if person == 'Hikari':
            return "Jay: Quoi encore, Hikari?"
        elif person == 'Viktoria':
            return "Jay: J'ai rien à te dire, Viktoria."
        else:
            return "Jay: Qu'est-ce que tu veux?"
    else:
        return "Jay: Salut, comment ça va?"
    
if __name__ == '__main__':
    chat_jay()
