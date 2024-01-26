from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os
import json
import time
from difflib import SequenceMatcher
from langid import classify
import spacy



class Jay:
    def __init__(self, model_output_dir='saved_model', memory_file='jay_memory.txt', json_file='conversations.json', feedback_file='feedback.json'):
        self.personality_data = self.read_from_json(json_file)
        self.memory_file = memory_file
        self.feedback_file = feedback_file
        self.memory = self.load_memory()
        self.language_model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.last_interaction_time = None
        self.initiate_conversation_threshold = 60
        self.conversation_threshold = 10
        self.current_context = None
        self.reward_coefficient = 0.1
        self.feedback_data = self.load_feedback()  
        self.model_output_dir = model_output_dir
        self.create_model_output_dir()

    def create_model_output_dir(self):
        if not os.path.exists(self.model_output_dir):
            os.makedirs(self.model_output_dir)

    def save_model(self, output_dir='saved_model'):
        with open(self.memory_file, 'a', encoding='utf-8') as file:
            if self.current_context:
                file.write(self.current_context['text'] + '\n')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.language_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def load_model(self, input_dir='saved_model'):
        self.language_model = GPT2LMHeadModel.from_pretrained(input_dir)
        self.tokenizer = GPT2Tokenizer.from_pretrained(input_dir)

    def load_memory(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                return [{"text": line.strip()} for line in lines]
        else:
            return []

    def load_feedback(self):
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, 'r', encoding='utf-8') as file:
                    return json.load(file)
            except json.decoder.JSONDecodeError:
                return {'feedback': []}
        else:
            return {'feedback': []}

    def update_feedback(self, feedback):
        feedback_data = self.load_feedback()
        feedback_data['feedback'].append(feedback)
        with open(self.feedback_file, 'w', encoding='utf-8') as file:
            json.dump(feedback_data, file, ensure_ascii=False, indent=4)

    def update_memory_from_json(self, json_file):
        new_data = self.read_from_json(json_file)
        if new_data:
            for interaction in new_data:
                if not self.is_similar(interaction, self.memory):
                    self.current_context = {
                        'text': interaction['text'],
                        'emotion': interaction.get('emotion', ''),
                        'sentiment': interaction.get('sentiment', ''),
                        'speaker': interaction.get('speaker', '')
                    }

    def is_similar(self, new_interaction, existing_interactions, threshold=0.8):
        new_text = new_interaction.get('text', '') if isinstance(new_interaction, dict) else new_interaction
        for existing_interaction in existing_interactions:
            existing_text = existing_interaction.get('text', '') if isinstance(existing_interaction, dict) else existing_interaction
            similarity = SequenceMatcher(None, new_text, existing_text).ratio()
            if similarity > threshold:
                return True
        return False

    def process_input(self, input_data):
        input_obj = json.loads(input_data)
        text = input_obj.get("text", "")
        emotion = input_obj.get("emotion", "")
        sentiment = input_obj.get("sentiment", "")
        response = self.generate_response(text, emotion, sentiment)
        self.update_memory(response)

    def evaluate_response(self, generated_response, user_input):
        reward = 0

        if self.current_context is not None:
            emotion = self.current_context.get("emotion", "").lower()
            sentiment = self.current_context.get("sentiment", "").lower()

            if "arrogant" in generated_response.lower():
                reward += 0.3
            if "sarcastique" in generated_response.lower():
                reward += 0.2
            if "antipathique" in generated_response.lower():
                reward += 0.2
            if "leader" in generated_response.lower() and "metal" in generated_response.lower():
                reward += 0.3
            if emotion == "positif":
                reward += 0.1
            elif emotion == "négatif":
                reward -= 0.1

            if sentiment == "positif":
                reward += 0.1
            elif sentiment == "négatif":
                reward -= 0.1

            jay_traits = ["arrogant", "sarcastique", "antipathique", "jaloux", "dominant", "leader de metal"]
            for trait in jay_traits:
                if trait in generated_response.lower():
                    reward += 0.1
            if "?" in generated_response:
                reward += 0.2
            if "pourquoi" in user_input.lower() and "parce que" in generated_response.lower():
                reward += 0.3
            if self.is_french(generated_response):
                reward += 0.2
            else:
                reward -= 0.2

        return reward


    def should_initiate_conversation(self):
        if self.last_interaction_time is None:
            return True
        elapsed_time = time.time() - self.last_interaction_time
        return elapsed_time > self.initiate_conversation_threshold
    
    def is_french(self, text):
        lang, _ = classify(text)
        return lang == 'fr'
    
    def correct_grammar(self, text):
        nlp = spacy.load("fr_core_news_sm")
        doc = nlp(text)
        corrected_text = " ".join(token.text for token in doc)
        return corrected_text
    
    def respond(self, user_input):
        generated_response = ""
        generated_responses = self.generate_responses(user_input)

        if generated_responses:
            chosen_response = generated_responses[0] 
            response_text, _, _ = chosen_response
            generated_response = f"Jay : {response_text}"
        else:
            generated_response = f"Jay : {self.generate_language_model_response(user_input)}"

        print(generated_response)
        feedback = input("Donne ton feedback (positif/négatif/neutre) sur la réponse de Jay : ")
        self.update_feedback({"response": generated_response, "feedback": feedback})
        feedback_value = feedback.lower()
        self.learn_from_feedback(feedback_value)
        reward = self.evaluate_response(generated_response, user_input)
        self.update_memory({'text': user_input, 'emotion': '', 'sentiment': ''})
        self.update_model(user_input, reward)
        self.save_memory({'text': generated_response})
        return generated_response
    
    def generate_language_model_response(self, user_input):
        input_ids = self.tokenizer.encode(user_input, return_tensors="pt")
        output = self.language_model.generate(input_ids, max_length=50, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95)
        generated_response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        corrected_response = self.correct_grammar(generated_response)
        return corrected_response
    
    @staticmethod
    def has_meaningful_content(text):
        return len(text) > 5
    
    def update_memory(self, interaction_dict):
        if self.is_french(interaction_dict['text']) and self.has_meaningful_content(interaction_dict['text']):
            self.current_context = interaction_dict
            if not any(self.is_similar(interaction_dict, existing) for existing in self.memory):
                self.memory.append(interaction_dict)
                self.save_memory(interaction_dict)

    def update_model(self, input_text, reward=0):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        self.language_model.train()
        adjusted_reward = reward * (1 - 0.1)

        if self.current_context is not None:
            feedback_value = self.current_context.get("feedback", "").lower()
            feedback_value = 0.2 if "positif" in feedback_value else -0.2 if "négatif" in feedback_value else 0
            emotion_value = self.current_context.get("emotion", "").lower()
            emotion_value = 0.2 if "positif" in emotion_value else -0.2 if "négatif" in emotion_value else 0
            sentiment_value = self.current_context.get("sentiment", "").lower()
            sentiment_value = 0.2 if "positif" in sentiment_value else -0.2 if "négatif" in sentiment_value else 0

            feedback_value *= (1 - 0.1)
            emotion_value *= (1 - 0.1)
            sentiment_value *= (1 - 0.1)
            adjusted_reward += feedback_value + emotion_value + sentiment_value

        loss = self.compute_loss(input_ids, adjusted_reward)
        self.backpropagate(loss)

    def compute_loss(self, input_ids, reward):
        outputs = self.language_model(input_ids, labels=input_ids)
        loss = outputs.loss

        if reward < 0:
            loss -= reward * self.reward_coefficient
        else:
            loss += reward * self.reward_coefficient
        return loss
    
    def backpropagate(self, loss):
        loss.backward()
        optimizer = torch.optim.Adam(self.language_model.parameters(), lr=0.001)
        optimizer.step()
        optimizer.zero_grad()

    def save_memory(self, interaction_dict):
        with open(self.memory_file, 'a', encoding='utf-8') as file:
            file.write(interaction_dict['text'] + '\n')

    def read_from_json(self, json_file):
        with open(json_file, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
            if isinstance(data, dict):
                return data
            else:
                return {}
            
    def generate_responses(self, user_input):
        responses = []
        if "bonjour" in user_input.lower():
            return [("Bonjour!", "neutre", "neutre")]
        if "positif" in user_input.lower():
            positive_responses = ["C'est génial!", "Je suis content de t'entendre dire ça.", "Positivité tout le chemin!"]
            responses.extend([(response, "positif", "positif") for response in positive_responses])
        if "pourquoi" in user_input.lower():
            responses.extend([("Parce que c'est ainsi que les choses fonctionnent.", "neutre", "neutre"),
                          ("Je ne suis pas sûr, tu as une idée?", "curiosite", "positif")])

        if "apprendre" in user_input.lower():
            self.learn_from_experience()
            responses.extend([("J'apprends aussi! Comment puis-je m'améliorer?", "excitation", "positif"),
                          ("L'apprentissage est excitant, n'est-ce pas?", "excitation", "positif")])

        if "toi" in user_input.lower() or "ton" in user_input.lower():
            responses.extend([("Parlons plutôt de toi. Qu'est-ce qui t'intéresse?", "curiosite", "neutre"),
                          ("Je suis curieux, quelles sont tes passions?", "curiosite", "neutre")])

        responses.extend([("Intéressant!", "neutre", "neutre"),
                      ("Que veux-tu dire par là?", "curiosite", "neutre"),
                      ("Je ne suis pas sûr de comprendre, peux-tu expliquer davantage?", "confusion", "neutre")])

        return responses

    def learn_from_experience(self):
        if self.current_context:
            user_input = self.current_context.get("text", "")
            reward = self.evaluate_response(self.current_context.get("text", ""), user_input)
            if reward > 0:
                self.update_model(user_input, reward)

    def learn_from_feedback(self, feedback_value):
        if feedback_value == "positif":
            self.update_model("Feedback positif", 0.1)
        elif feedback_value == "négatif":
            self.update_model("Feedback négatif", -0.1)
        else:
            pass

    def adjust_model_weights(self, positive_feedback=True):
        current_weights = self.language_model.state_dict()
        positive_factor = 1.1 
        negative_factor = 0.9
        for param_tensor in current_weights:
            if positive_feedback:
                current_weights[param_tensor] *= positive_factor
            else:
                current_weights[param_tensor] *= negative_factor
        self.language_model.load_state_dict(current_weights)

bot = Jay()

try:
    while True:
        if bot.should_initiate_conversation():
            user_input = input("Hikari : ")
        else:
            user_input = input()

        if user_input.lower() == 'exit':
            print("Fin de la conversation.")
            bot.save_model()
            bot.save_memory()
            break

        print("Avant la réponse...")
        try:
            bot_response = bot.respond(user_input)
            bot.save_model()
        except KeyboardInterrupt:
            print("\nInterruption manuelle. Fin de la conversation.")
            bot.save_model()
            interaction_dict = {'text': '', 'emotion': '', 'sentiment': ''}
            bot.save_memory(interaction_dict)
            break

        print("Après la réponse...")
        print(bot_response)
        json_file = 'conversations.json'
        bot.update_memory_from_json(json_file)

except KeyboardInterrupt:
    print("\nInterruption manuelle. Fin de la conversation.")
    bot.save_model() 
    interaction_dict = {'text': '', 'emotion': '', 'sentiment': ''}
    bot.save_memory(interaction_dict)
