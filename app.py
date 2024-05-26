from flask import Flask, request, render_template, session, jsonify
from flask_session import Session
import random
import sys
import optuna
import subprocess
import uuid
import re
import requests
import logging
import json
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from pandas import DataFrame
import numpy as np
from collections import deque
import signal
from textblob import TextBlob

import keras_nlp
import keras
from keras import backend as K
import jax
from lime import lime_tabular
import optuna
import nltk
from nltk.tokenize import word_tokenize
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, LSTM, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from imblearn.ensemble import BalancedBaggingClassifier
import peft
from peft import LoraConfig
import torch
from trl import SFTTrainer
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from langchain_text_splitters import TokenTextSplitter

CONVERSATION_LIMIT = 10
HISTORY_FILE = 'conversation_history.json'
TOKENIZERS_PARALLELISM= False
memory_entries = []
conversation_histories = {}
keras.mixed_precision.set_global_policy("mixed_float16")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(filename='ml_process.log', level=logging.INFO)


# Load conversation histories from file
def load_conversation_histories():
    try:
        with open(HISTORY_FILE, 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

# Save conversation histories to file
def save_conversation_histories():
    with open(HISTORY_FILE, 'w') as file:
        json.dump(conversation_histories, file)


# Generate a new conversation ID
def generate_conversation_id():
    return str(uuid.uuid4())

# Retrieve conversation history by ID
def retrieve_conversation_history(conversation_id):
    return conversation_histories.get(conversation_id, [])

# Store conversation history by ID
def store_conversation_history(conversation_id, conversation_history):
    conversation_histories[conversation_id] = conversation_history
    save_conversation_histories()

text_splitter = TokenTextSplitter(chunk_size=10, chunk_overlap=0)  

    # Tokenize conversation history
def tokenize_conversation_history(conversation_history):
    tokenized_history = []
    for message in conversation_history:
        if message["role"] == "user" or message["role"] == "assistant":
            tokens = text_splitter.split_text(message["content"])
            tokenized_history.append((message["role"], tokens))
    return tokenized_history

# List installed models using subprocess and curl

def process_model(model, question, content):
    try:
        # Generate the response using the Phi-3 model
        response = run_model_chat(question, content)
        return (model, response[model])
    except Exception as e:
        logging.error(f"Error generating response for model {model}. Error message: {str(e)}")
        return None
    
# Load the Phi-3 model and training configuration
checkpoint_path = "microsoft/phi-1_5"
phi3_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")
tokenizer.model_max_length = 2048
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = 'right'
model_kwargs = dict(
        use_cache=False,
        trust_remote_code=True,
        use_flash_attention_2=False, 
        torch_dtype=torch.bfloat16,
        device_map=None
)
model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **model_kwargs)
training_config = {
            "bf16": True,
            "do_eval": False,
            "learning_rate": 5.0e-06,
            "log_level": "info",
            "logging_steps": 20,
            "logging_strategy": "steps",
            "lr_scheduler_type": "cosine",
            "num_train_epochs": 1,
            "max_steps": -1,
            "output_dir": "./checkpoint_dir",
            "overwrite_output_dir": True,
            "per_device_eval_batch_size": 4,
            "per_device_train_batch_size": 4,
            "remove_unused_columns": True,
            "save_steps": 100,
            "save_total_limit": 1,
            "seed": 0,
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs":{"use_reentrant": False},
            "gradient_accumulation_steps": 1,
            "warmup_ratio": 0.2,
            }

peft_config = {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": "all-linear",
            "modules_to_save": None,
            }       
train_conf = TrainingArguments(**training_config)
peft_conf = LoraConfig(**peft_config)
model.half()

def run_model_chat(question, content):
    model_names = checkpoint_path()
    input_ids = tokenizer.encode(f"{question} {content}", return_tensors="pt")
    output = phi3_model.generate(input_ids, max_length=150, num_return_sequences=1, temperature=0.7)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(checkpoint_path, model, question, content) for model in model_names]
        results = [future.result() for future in futures if future.result() is not None]

    all_responses = {model: response for model, response in results}
    return all_responses
    
# Define fuzzy variables and membership functions for FS-IRDM
risk = ctrl.Antecedent(np.arange(0, 11, 1), 'risk')
uncertainty = ctrl.Antecedent(np.arange(0, 11, 1), 'uncertainty')
decision = ctrl.Consequent(np.arange(0, 11, 1), 'decision')

risk['low'] = fuzz.trimf(risk.universe, [0, 0, 5])
risk['medium'] = fuzz.trimf(risk.universe, [0, 5, 10])
risk['high'] = fuzz.trimf(risk.universe, [5, 10, 10])

uncertainty['low'] = fuzz.trimf(uncertainty.universe, [0, 0, 5])
uncertainty['medium'] = fuzz.trimf(uncertainty.universe, [0, 5, 10])
uncertainty['high'] = fuzz.trimf(uncertainty.universe, [5, 10, 10])

decision['low'] = fuzz.trimf(decision.universe, [0, 0, 5])
decision['medium'] = fuzz.trimf(decision.universe, [0, 5, 10])
decision['high'] = fuzz.trimf(decision.universe, [5, 10, 10])

# Define fuzzy rules
rules = [
    ctrl.Rule(risk['low'] & uncertainty['low'], decision['high']),
    ctrl.Rule(risk['medium'] & uncertainty['low'], decision['medium']),
    ctrl.Rule(risk['high'] & uncertainty['low'], decision['low']),
    ctrl.Rule(risk['low'] & uncertainty['medium'], decision['medium']),
    ctrl.Rule(risk['medium'] & uncertainty['medium'], decision['medium']),
    ctrl.Rule(risk['high'] & uncertainty['medium'], decision['low']),
    ctrl.Rule(risk['low'] & uncertainty['high'], decision['low']),
    ctrl.Rule(risk['medium'] & uncertainty['high'], decision['low']),
    ctrl.Rule(risk['high'] & uncertainty['high'], decision['low']),
]

# Create the fuzzy control system
decision_ctrl = ctrl.ControlSystem(rules)
decision_sim = ctrl.ControlSystemSimulation(decision_ctrl)

# Function to train the ML model at startup
def run_ml_code(conn):
    # Generate synthetic data
    num_samples = 16854
    num_features = 32
    data = generate_synthetic_data(num_samples, num_features)
    X_resampled, y_resampled = preprocess_data(data)

    # Optimize the model using Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_resampled, y_resampled), n_trials=400)

    # Get the best hyperparameters
    best_params = study.best_params
    logging.info(f"Best hyperparameters: {best_params}")

    # Train the model with the best hyperparameters
    model = create_model(input_shape=(X_resampled.shape[1], 1), num_classes=len(np.unique(y_resampled)))
    optimizer = Adam(learning_rate=best_params['learning_rate'])
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_resampled.reshape(-1, X_resampled.shape[1], 1), y_resampled, batch_size=best_params['batch_size'], epochs=10, verbose=1)

    # Save the trained model
    model.save('decision_making_model.h5')
    logging.info("Model training completed.")

    # Fine-tune the Phi-3 model using the conversation history
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    tokenizer.model_max_length = 2048
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = 'right'

    # Use the conversation history as the training dataset
    train_dataset = conversation_histories.values()
    processed_train_dataset = [{"messages": [{"role": message["role"], "content": message["content"]} for message in conversation]} for conversation in train_dataset]
    processed_train_dataset = processed_train_dataset.map(conversation_histories, fn_kwargs={"tokenizer": tokenizer}, num_proc=10, remove_columns=["messages"], desc="Applying chat template to train dataset")

    trainer = SFTTrainer(
        model=phi3_model,
        args=train_conf,
        peft_config=peft_conf,
        train_dataset=processed_train_dataset,
        eval_dataset=None,
        max_seq_length=2048,
        dataset_text_field="text",
        tokenizer=tokenizer,
        packing=True
    )
    trainer.train()

    # Send the trained models back to the main process
    conn.send((model, phi3_model))
    conn.close()

# Generate synthetic data
def generate_synthetic_data(num_samples, num_features, imbalance_ratio=0.3):
    X = np.random.rand(num_samples, num_features)
    y = np.random.choice([0, 1], size=num_samples, p=[1 - imbalance_ratio, imbalance_ratio])
    data = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(num_features)])
    data['target'] = y
    return data

def preprocess_data(data):
    X = data.drop('target', axis=1)
    y = data['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    preprocessor = keras_nlp.models.BertPreprocessor.from_preset(
        "bert_tiny_en_uncased",
        sequence_length=256,
    )

    # Convert X_scaled to a list of strings
    X_texts = [' '.join(str(x) for x in row) for row in X_scaled]

    tokenized_data = preprocessor(X_texts)

    # Convert the JAX array to a NumPy array using jax.device_get()
    X_token_ids = jax.device_get(tokenized_data['token_ids'])

    # Apply imbalanced learning techniques
    over_sampler = SMOTE(random_state=42)
    under_sampler = RandomUnderSampler(random_state=42)
    pipeline = Pipeline([('over', over_sampler), ('under', under_sampler)])
    X_resampled, y_resampled = pipeline.fit_resample(X_token_ids, y_encoded)

    return X_resampled, y_resampled



def create_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv1D(512, 64, activation='relu', padding='same')(inputs)
    x = Conv1D(512, 64, activation='relu', padding='same')(x)
    x = Conv1D(512, 64, activation='relu', padding='same')(x)
    x = LSTM(512, return_sequences=False)(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


# Define the objective function for Optuna optimization
def objective(trial, X_train, y_train):
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [256, 512])
    dropout_rate = trial.suggest_float('dropout_rate', 0.334, 0.665)

    model = create_model(input_shape=(X_train.shape[1], 1), num_classes=len(np.unique(y_train)))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    cv_scores = []
    kf = KFold(n_splits=3, shuffle=True, random_state=32)
    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        model.fit(X_train_fold.reshape(-1, X_train_fold.shape[1], 1), y_train_fold, batch_size=batch_size, epochs=10, verbose=0)
        y_pred_fold = model.predict(X_val_fold.reshape(-1, X_val_fold.shape[1], 1))
        cv_scores.append(balanced_accuracy_score(y_val_fold, np.argmax(y_pred_fold, axis=1)))

    return np.mean(cv_scores)

def custom_orthogonal(shape, dtype=None):
    return Orthogonal(gain=1.0)(shape, dtype=dtype)

# Load the trained model
def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path, custom_objects={'Orthogonal': custom_orthogonal})
        if not model.optimizer:
            logging.warning("No training configuration found in the save file, so the model was *not* compiled. Compiling it manually.")
            optimizer = Adam(learning_rate=0.000001)
            model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        elif not hasattr(model.optimizer, 'variables'):
            logging.warning("Optimizer does not have 'variables' attribute. Recreating the optimizer.")
            optimizer = Adam(learning_rate=0.000001)
            model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    except IOError:
        logging.error(f"Error loading model from path: {model_path}")
        return None

# Define the neural network model for DQN
def build_dqn_model(state_size, action_size):
    inputs = Input(shape=(state_size,))
    x = Dense(64, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(action_size, activation='linear')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.000001))
    return model

# DQNAgent class for reinforcement learning
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.998    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.000001
        self.epsilon_decay = 0.995
        self.learning_rate = 0.000001
        self.model = build_dqn_model(state_size, action_size)
        self.target_model = build_dqn_model(state_size, action_size)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
            self.model.fit(state, target, epochs=1, verbose=1)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

state_size = 2
action_size = 11
agent = DQNAgent(state_size, action_size)

def compute_decision(risk_value, uncertainty_value):
    state = np.array([risk_value, uncertainty_value]).reshape(1, -1)
    action = agent.act(state)
    decision_sim.input['risk'] = risk_value
    decision_sim.input['uncertainty'] = uncertainty_value
    decision_sim.compute()
    logging.info(f"Computing decision... Action: {action}, Decision Value: {decision_sim.output['decision']}")
    return action, decision_sim.output['decision']

# Truncate user input to a maximum length
def truncate_user_input(user_input, max_length):
    return user_input[:max_length]

# Handle user query and update conversation history
def handle_query(question, user_input, conversation_offset, risk_score, uncertainty_score, system_prompt, temperature, session, conversation_history):
    url_pattern = re.compile(r'(https?://\S+)')
    url_match = url_pattern.search(user_input)
    
    if url_match:
        url = url_match.group(1)
        try:
            response = requests.get(url)
            if response.status_code == 200:
                page_content = response.text
                user_input += f"\n\nURL Content:\n{page_content}"
            else:
                user_input += f"\n\nFailed to fetch the web page. Status code: {response.status_code}"
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching URL: {url}. Error: {e}")
            user_input += f"\n\nFailed to fetch the web page. Error: {str(e)}"

    user_input = truncate_user_input(user_input, max_length=80000)

    max_length = 80000
    if len(user_input) > max_length:
        user_input = user_input[:max_length]

    conversation_id = session.get('conversation_id')
    if conversation_id in conversation_histories:
        conversation_history = conversation_histories[conversation_id]
    else:
        conversation_history = []

    tokenized_history = tokenize_conversation_history(conversation_history)

    messages = []

    for message in conversation_history:
        if message["role"] == "user":
            messages.append({"role": "user", "content": message["content"]})
        elif message["role"] == "assistant":
            messages.append({"role": "assistant", "content": message["content"]})
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_input})

    # Run the query through the installed models
    all_responses = run_model_chat(question=question, content=user_input)

    # For simplicity, just take the first response (adjust as needed)
    assistant_response = next(iter(all_responses.values()), "Sorry, I couldn't generate a response.")

    conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append({"role": "assistant", "content": assistant_response})

    store_conversation_history(conversation_id, conversation_history)
    memory_entries.append((user_input, assistant_response))

    # Compute decision and update Q-table
    state = np.array([risk_score, uncertainty_score]).reshape(1, -1)
    action, decision_value = compute_decision(risk_score, uncertainty_score)
    reward = decision_value  # Use the decision value as a reward for simplicity

    next_risk_score = random.uniform(0, 10)
    next_uncertainty_score = random.uniform(0, 10)
    next_state = np.array([next_risk_score, next_uncertainty_score]).reshape(1, -1)
    done = False  # In a real scenario, this would be determined by the conversation context
    agent.remember(state, action, reward, next_state, done)
    agent.replay(64)  # Train with a batch size of 32
    agent.update_target_model()

    try:
        # Retrieve new data (e.g., from a database or buffer)
        new_data = retrieve_new_data(conversation_history)

        if new_data is not None:
            logging.info("Performing dynamic learning...")
            print("Dynamic Learning in progress...")
            # Perform dynamic learning with the new data
            dynamic_learning(new_data)
            logging.info("Dynamic learning completed.")
            print("Dynamic Learning Successfully completed!")

        return assistant_response, conversation_history, risk_score, uncertainty_score, system_prompt, temperature

    except Exception as e:
        logging.error(f"Error in handle_query: {e}")
        # Return a tuple with default values in case of an error
        return "Sorry, an error occurred.", [], 0.0, 0.0, "", 0.0
      
# Retrieve new data (e.g., from a database or buffer)
def retrieve_new_data(conversation_history):
    try:
        # Retrieve new data from the conversation history
        new_data = []
        for message in conversation_history[-10:]:  # Retrieve the last 5 messages
            if message["role"] == "user":
                user_input = message["content"]
                new_data.append({"user_input": user_input, "processed": False, "target": 0})  # Assign a default target value

        # Convert new_data to a pandas DataFrame
        new_data_df = pd.DataFrame(new_data)

        # Perform any necessary data preprocessing or filtering
        # For example, you can filter rows based on a specific condition
        new_data_df = new_data_df[new_data_df['processed'] == False]

        # Update the processed flag for the retrieved data
        new_data_df['processed'] = True

        return new_data_df

    except Exception as e:
        logging.error(f"Error retrieving new data: {e}")
        return None
    

# Dynamic Learning
def dynamic_learning(new_data):
    try:
        # Preprocess the new data
        logging.info("Preprocessing new data...")
        X_new = new_data.drop('target', axis=1)
        y_new = new_data['target']

        # Perform feature extraction on text data
        vectorizer = TfidfVectorizer()
        X_new = vectorizer.fit_transform(X_new.iloc[:, 0])
        X_new = X_new.toarray()

        try:
            X_new = X_new.astype(np.float32)
        except ValueError as e:
            logging.warning(f"Error converting features to float: {str(e)}")
            logging.info("Performing feature extraction on text data...")
        
        # Reshape the input data to match the expected shape
        input_shape = (X_new.shape[1], 1)
        X_new = X_new.reshape((-1, *input_shape))

        # Load the existing model
        logging.info("Loading the trained model...")
        model_path = 'decision_making_model.h5'
        model = load_model(model_path)

        if model is None:
            # If no existing model, train a new model with the new data
            logging.info("Training a new model with the new data.")
            model = create_model(input_shape=input_shape, num_classes=len(np.unique(y_new)))
            optimizer = Adam(learning_rate=0.01)
            model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
            model.fit(X_new, y_new, batch_size=256, epochs=10, callbacks=[early_stopping])
        else:
            # If an existing model is found, incrementally train it with the new data
            logging.info("Incrementally training the existing model with the new data.")

            # Check if the model's optimizer has the right variables
            if not hasattr(model.optimizer, 'variables'):
                model.optimizer = Adam(learning_rate=0.01)
                model.compile(optimizer=model.optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            # Compare the input shape of the new data with the model's expected input shape
            expected_input_shape = model.input_shape[1:]
            if input_shape != expected_input_shape:
                logging.warning(f"Input shape mismatch: Expected {expected_input_shape}, found {input_shape}")
                logging.info("Rebuilding the model architecture...")
                model = create_model(input_shape=input_shape, num_classes=len(np.unique(y_new)))
                optimizer = Adam(learning_rate=0.01)
                model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
            model.fit(X_new, y_new, batch_size=256, epochs=10, callbacks=[early_stopping])

        # Save the trained model
        logging.info("Saving the updated model...")
        model.save(model_path)
        logging.info("Model saved successfully.")

        return model
    
    except Exception as e:
        logging.error(f"Error in dynamic_learning: {str(e)}")
        return None
    
# Perform stream processing on user input
def perform_stream_processing(user_input):
    sentiment_score = analyze_sentiment(user_input)
    word_count = len(user_input.split())
    keywords = ["decision", "uncertainty", "risk"]
    keyword_count = sum(1 for keyword in keywords if keyword in user_input.lower())

    return {
        "sentiment_score": sentiment_score,
        "word_count": word_count,
        "keyword_count": keyword_count
    }

# Perform batch processing on stored data
def perform_batch_processing(batch_data):
    try:
        topics = topic_modeling(batch_data)
        user_behavior_insights = analyze_user_behavior(batch_data)
        y_true, y_pred, labels = get_true_and_predicted_labels(batch_data)

        # Convert NumPy arrays to lists for JSON serialization
        if isinstance(y_true, np.ndarray):
            y_true = y_true.tolist()
        if isinstance(y_pred, np.ndarray):
            y_pred = y_pred.tolist()
        if isinstance(labels, np.ndarray):
            labels = labels.tolist()

        return {
            "topics": topics,
            "user_behavior_insights": user_behavior_insights,
            "y_true": y_true,
            "y_pred": y_pred,
            "labels": labels
        }

    except Exception as e:
        logging.error(f"Error in perform_batch_processing: {e}")
        return {
            "topics": [],
            "user_behavior_insights": [],
            "y_true": [],
            "y_pred": [],
            "labels": []
        }

# Analyze sentiment of text input
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    return sentiment_score

# Perform topic modeling on batch data
def topic_modeling(batch_data):
    try:
        documents = []
        for row in batch_data:
            if len(row) >= 2:
                documents.append(row[0] + " " + row[1])
            else:
                logging.warning(f"Skipping row with insufficient elements: {row}")

        if not documents:
            return ["No valid data for topic modeling."]

        vectorizer = CountVectorizer(stop_words='english')
        X = vectorizer.fit_transform(documents)
        lda_model = LatentDirichletAllocation(n_components=7, random_state=42)
        lda_model.fit(X)

        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(lda_model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
            topics.append(f"Topic #{topic_idx + 1}: " + ", ".join(top_words))

        return topics

    except Exception as e:
        logging.error(f"Error in topic_modeling: {e}")
        return ["Error in topic modeling."]

# Analyze user behavior from batch data
def analyze_user_behavior(batch_data):
    user_input_count = len([row[0] for row in batch_data])
    assistant_response_count = len([row[1] for row in batch_data])

    user_input_lengths = [len(row[0]) for row in batch_data]
    avg_user_input_length = sum(user_input_lengths) / len(user_input_lengths) if user_input_lengths else 0

    return [
        f"User Input Count: {user_input_count}",
        f"Assistant Response Count: {assistant_response_count}",
        f"Average User Input Length: {avg_user_input_length}"
    ]

# Get true and predicted labels for batch data
def get_true_and_predicted_labels(batch_data):
    try:
        y_true = []
        y_pred = []

        for user_input, assistant_response in batch_data:
            user_sentiment = analyze_sentiment(user_input)
            true_label = 1 if user_sentiment > 0 else 0

            assistant_sentiment = analyze_sentiment(assistant_response)
            predicted_label = 1 if assistant_sentiment > 0 else 0

            y_true.append(true_label)
            y_pred.append(predicted_label)

        # Convert y_true and y_pred to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Get the unique classes present in both y_true and y_pred
        labels = np.unique(np.concatenate((y_true, y_pred)))

        return y_true, y_pred, labels

    except Exception as e:
        logging.error(f"Error in get_true_and_predicted_labels: {e}")
        return [], [], []
    

def signal_handler(sig, frame):
    global flask_process
    logging.info('Stopping the ML process...')
    ml_process.terminate()
    ml_process.join()
    logging.info('ML process stopped.')

    if flask_process:
        logging.info('Stopping the Flask app process...')
        flask_process.terminate()
        flask_process.join()
        logging.info('Flask app process stopped.')

    sys.exit(0)

# Retrieve batch data from memory entries
def retrieve_batch_data():
    return memory_entries

# Store batch processing results
def store_batch_processing_results(batch_processing_result):
    try:
        # Store the batch processing results in a database or file
        # Implement the storage mechanism based on your requirements
        logging.info("Storing batch processing results: %s", batch_processing_result)
    except Exception as e:
        logging.error(f"Error storing batch processing results: {e}")

if __name__ == '__main__':
    global flask_process
    TOKENIZERS_PARALLELISM=(True | False)
    nltk.download('punkt')
    
    conversation_histories = load_conversation_histories()

    # Start the Flask app
    flask_process = subprocess.Popen(['python3', 'flask_app.py'])

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create a pipe for communication between processes
    parent_conn, child_conn = multiprocessing.Pipe()

    # Create and start the ML process
    ml_process = multiprocessing.Process(target=run_ml_code, args=(child_conn,))
    ml_process.start()

    # Receive the trained model from the ML process
    model = parent_conn.recv()

    # Wait for the Flask app process to finish
    flask_process.wait()
    force_download = True
    TOKENIZERS_PARALLELISM=(True | False)

    conversation_histories = load_conversation_histories()