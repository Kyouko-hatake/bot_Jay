import numpy as np
from collections import Counter
from keras.models import Sequential, load_model
from keras.layers import Embedding, Dense, Flatten
from keras.callbacks import EarlyStopping
import re
import json
import os

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9éèêàçùûôöîïâäë\'\"\-\s.?!]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def create_vocabulary(text, vocab=None):
    words = text.split()
    word_counts = Counter(words)
    if vocab is None:
        vocab = {word: idx for idx, (word, _) in enumerate(word_counts.items(), 1)}
    else:
        for word in words:
            if word not in vocab:
                vocab[word] = len(vocab) + 1
    return vocab

def predict_next_word(model, current_word, vocab):
    word_index = vocab.get(current_word, None)
    if word_index is None:
        return None
    predicted_index = np.argmax(model.predict(np.array([word_index])), axis=-1)[0]
    for word, idx in vocab.items():
        if idx == predicted_index:
            return word
    return None

def generate_text(model, start_word, vocab, length=10):
    current_word = start_word
    generated_text = [current_word]
    for _ in range(length - 1):
        next_word = predict_next_word(model, current_word, vocab)
        if next_word is None:
            break
        generated_text.append(next_word)
        current_word = next_word
    return ' '.join(generated_text)

# Préparation des données d'entraînement
def prepare_training_data(dialogues, vocab):
    X = []
    y = []
    for user_question, jay_response in dialogues:
        user_question_words = user_question.split()
        jay_response_words = jay_response.split()
        for i in range(len(user_question_words)):
            if i < len(user_question_words) - 1:
                X.append(vocab[user_question_words[i]])
                y.append(vocab[user_question_words[i + 1]])
        if jay_response_words:
            X.append(vocab[user_question_words[-1]])
            y.append(vocab[jay_response_words[0]])
            for i in range(len(jay_response_words) - 1):
                X.append(vocab[jay_response_words[i]])
                y.append(vocab[jay_response_words[i + 1]])
    return np.array(X), np.array(y)

def initialize_model(vocab_size, embedding_dim=10):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim),
        Flatten(),
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def expand_model(model, old_vocab_size, new_vocab_size, embedding_dim=10):
    new_model = Sequential([
        Embedding(input_dim=new_vocab_size, output_dim=embedding_dim),
        Flatten(),
        Dense(new_vocab_size, activation='softmax')
    ])
    new_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    old_weights = model.get_weights()
    new_weights = new_model.get_weights()
    
    if old_weights and new_weights:
        new_weights[0][:old_vocab_size] = old_weights[0][:old_vocab_size]
        new_weights[1][:old_vocab_size, :] = old_weights[1][:old_vocab_size, :]
        new_weights[2][:old_vocab_size] = old_weights[2][:old_vocab_size]
        
        new_model.set_weights(new_weights)
    
    return new_model

def save_checkpoint(model, vocab, dialogues, checkpoint_dir='checkpoints'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    model.save(os.path.join(checkpoint_dir, 'model.keras'))
    with open(os.path.join(checkpoint_dir, 'vocab.json'), 'w') as f:
        json.dump(vocab, f)
    with open(os.path.join(checkpoint_dir, 'dialogues.json'), 'w') as f:
        json.dump(dialogues, f)

def load_checkpoint(checkpoint_dir='checkpoints'):
    model = load_model(os.path.join(checkpoint_dir, 'model.keras'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  
    with open(os.path.join(checkpoint_dir, 'vocab.json'), 'r') as f:
        vocab = json.load(f)
    with open(os.path.join(checkpoint_dir, 'dialogues.json'), 'r') as f:
        dialogues = json.load(f)
    return model, vocab, dialogues

def interactive_mode():
    global vocab, model, dialogues
    
    vocab = create_vocabulary("")
    vocab_size = len(vocab) + 1  
    model = initialize_model(vocab_size)
    dialogues = []

    if os.path.exists('checkpoints/model.keras') and os.path.exists('checkpoints/vocab.json') and os.path.exists('checkpoints/dialogues.json'):
        model, vocab, dialogues = load_checkpoint()
    
    if dialogues:
        print("Dialogues déjà entraînés :")
        for user_question, jay_response in dialogues:
            print(f"Utilisateur: {user_question}")
            print(f"Jay: {jay_response}")
    else:
        print("Aucun dialogue entraîné pour le moment.")
    
    while True:
        user_input = input("Entrez une question pour Jay ou 'exit' pour quitter: ").strip()
        if user_input.lower() == 'exit':
            print("Fin de l'interaction.")
            break
        
        user_question_clean = preprocess_text(user_input)
        
        jay_response = input("Entrez la réponse attendue de Jay: ").strip()
        jay_response_clean = preprocess_text(jay_response)
        
        # Vérifier si le dialogue est déjà entraîné
        if (user_question_clean, jay_response_clean) in dialogues:
            print("Dialogue déjà entraîné.")
            continue
        
        dialogues.append((user_question_clean, jay_response_clean))
        
        corpus = ' '.join([q + ' ' + a for q, a in dialogues])
        old_vocab_size = len(vocab) + 1
        vocab = create_vocabulary(corpus, vocab)
        new_vocab_size = len(vocab) + 1
        print("Vocabulaire mis à jour:", vocab)
        
        if new_vocab_size > old_vocab_size:
            model = expand_model(model, old_vocab_size, new_vocab_size)
            print("Modèle étendu.")
        
        X_train, y_train = prepare_training_data(dialogues, vocab)
        y_train = np.expand_dims(y_train, axis=-1)
        print("Données d'entraînement préparées:", X_train, y_train)
        
        early_stopping = EarlyStopping(monitor='loss', patience=5)
        model.fit(X_train, y_train, epochs=100, verbose=1, callbacks=[early_stopping])
        
        save_checkpoint(model, vocab, dialogues)
        
        start_word = input("Entrez un mot pour commencer la génération de texte: ").strip()
        generated_text = generate_text(model, start_word, vocab)
        print("Texte généré:", generated_text)
        print("Tokens utilisés:", [vocab[word] for word in user_question_clean.split()])

interactive_mode()
