import json
import difflib
from transformers import pipeline
import spacy

nlp = spacy.load("fr_core_news_sm")

knowledge_base = {"questions": []}
memory = []

mood_jay = "neutre"
mood_hikari = "neutre"

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

def analyze_sentiment(text):
    sentiment_analyzer = pipeline('sentiment-analysis')
    result = sentiment_analyzer(text)[0]
    return result['label'], result['score']

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
            sentiment_label, sentiment_score = analyze_sentiment(existing_answer)
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

def chat_jay():
    while True:
        try:
            user_input = input('Hikari: ')

            if user_input.lower() == 'quitter':
                break
            elif user_input.lower() == 'apprendre':
                new_answer = input("nouvelle réponse: ")
                automatic_learning(user_input, new_answer, knowledge_base, memory)

            else:
                existing_answer, is_recorded = get_answer_for_question(user_input, knowledge_base, memory)
                if is_recorded:
                    sentiment_label, sentiment_score = analyze_sentiment(existing_answer)
                    adjusted_response = adjust_response_based_on_mood(existing_answer, mood_jay, mood_hikari)
                    print(f"Jay: {adjusted_response}")
                    print(f"Sentiment de la réponse de Jay: {sentiment_label} avec un score de {sentiment_score}")
                else:
                    print("Jay: Je ne sais pas. Recherche dans ma base de données...")
                    existing_answer, is_recorded = get_answer_for_question(user_input, knowledge_base, memory)
                    if is_recorded:
                        sentiment_label, sentiment_score = analyze_sentiment(existing_answer)
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

if __name__ == '__main__':
    chat_jay()

