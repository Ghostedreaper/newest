from flask import Flask, request, render_template, session, jsonify
from flask_session import Session
import json
from app import (
    generate_conversation_id, retrieve_conversation_history, handle_query,
    perform_stream_processing, retrieve_batch_data, perform_batch_processing,
    store_batch_processing_results
)
from sklearn.metrics import balanced_accuracy_score
import logging
import numpy as np
from sklearn.metrics import accuracy_score


app = Flask(__name__)
app.secret_key = 'Rh65$@jk#!sQ1mN9'
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.config["TOKENIZERS_PARALLELISM"] = "(true | false)"
Session(app)

CONVERSATION_LIMIT = 100
HISTORY_FILE = 'conversation_history.json'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@app.route('/')
def home():
    try:
        conversation_id = generate_conversation_id()
        session['conversation_id'] = conversation_id
        conversation_history = retrieve_conversation_history(conversation_id)
        suggestions = ["Ask me for a summary.", "What should we decide on X?", "Update me on the latest events."]
        return render_template('chat.html', suggestions=suggestions, CONVERSATION_LIMIT=CONVERSATION_LIMIT, conversation_id=conversation_id, conversation_history=conversation_history)
    except Exception as e:
        logging.error(f"Error in home route: {str(e)}")
        return "An error occurred", 500

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        if not data:
            raise ValueError("Missing request data")
        user_input = data.get('user_input')
        if not user_input:
            raise ValueError("Missing 'user_input' in request data")

        question = data.get('question')  # Retrieve the question from the request data
        if not question:
            raise ValueError("Missing 'question' in request data")

        conversation_offset = data.get('conversation_offset', 0)
        risk_score = float(data.get('risk_score', 0.0))
        uncertainty_score = float(data.get('uncertainty_score', 0.0))
        system_prompt = data.get('system_prompt', '')
        temperature = float(data.get('temperature', 1.0))
        conversation_id = data.get('conversation_id')

        if not conversation_id:
            raise ValueError("Missing 'conversation_id' in request data")

        conversation_history = retrieve_conversation_history(conversation_id)

        # Ensure handle_query is a callable function and not shadowed
        assert callable(handle_query), "handle_query must be a callable function"

        response, updated_conversation_history, risk_score, uncertainty_score, system_prompt, temperature = handle_query(
            question, user_input, conversation_offset, risk_score, uncertainty_score, system_prompt, temperature, session, conversation_history
        )

        stream_processing_result = perform_stream_processing(user_input)
        batch_data = retrieve_batch_data()
        batch_processing_result = perform_batch_processing(batch_data)
        store_batch_processing_results(batch_processing_result)

        accuracy_score = None
        if 'y_true' in batch_processing_result and 'y_pred' in batch_processing_result:
            y_true = np.array(batch_processing_result['y_true'])
            y_pred = np.array(batch_processing_result['y_pred'])

            logging.info(f"Shape of y_true: {y_true.shape}")
            logging.info(f"Dimensions of y_true: {y_true.ndim}")
            logging.info(f"Shape of y_pred: {y_pred.shape}")
            logging.info(f"Dimensions of y_pred: {y_pred.ndim}")

            # Flatten the arrays if they have more than 1 dimension
            if y_true.ndim > 1:
                y_true = y_true.flatten()
            if y_pred.ndim > 1:
                y_pred = y_pred.flatten()

            accuracy_score = balanced_accuracy_score(y_true, y_pred)

        suggestions = ["Ask me for a summary.", "What should we decide on X?", "Update me on the latest events."]

        return jsonify(
            response=response,
            conversation_history=updated_conversation_history,
            suggestions=suggestions,
            conversation_offset=conversation_offset,
            CONVERSATION_LIMIT=CONVERSATION_LIMIT,
            batch_processing_result=batch_processing_result,
            conversation_id=conversation_id,
            risk_score=risk_score,
            uncertainty_score=uncertainty_score,
            system_prompt=system_prompt,
            temperature=temperature,
            accuracy_score=accuracy_score
        )
    except Exception as e:
        logging.error(f"Error in ask route: {str(e)}")
        return jsonify(error=str(e)), 500
    
if __name__ == '__main__':
    try:
        app.run(debug=True)
    except Exception as e:
        logging.error(f"Flask app failed to start: {str(e)}")