from flask import Flask, request, render_template
import pickle
from tensorflow.keras.models import load_model
import re

app = Flask(__name__)

# Load the Naive Bayes model and its TF-IDF vectorizer
nb_model = pickle.load(open('model.sav', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Load the Neural Network model and its tokenizer

nn_model = load_model('my_spam_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

stop_words = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
    "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "d",
    "ll", "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn",
    "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn",
    "weren", "won", "wouldn"
])


def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove numbers, special characters, and punctuation
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # Tokenize text (split into words)
    words = text.split()

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    # Join words back to string
    text = ' '.join(words)
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/nb_classifier', methods=['GET', 'POST'])
def nb_classifier():
    if request.method == 'POST':
        message = request.form['message']
        preprocessed_message = preprocess_text(message)  # Ensure this function is defined
        vectorized_message = tfidf_vectorizer.transform([preprocessed_message])
        prediction = nb_model.predict(vectorized_message)[0]
        return render_template('nb_classifier.html', prediction_text='Email is {}'.format('Spam' if prediction == 1 else 'Not Spam'))
    return render_template('nb_classifier.html')

@app.route('/nn_classifier', methods=['GET', 'POST'])
def nn_classifier():
    if request.method == 'POST':
        message = request.form['message']
        preprocessed_message = preprocess_text(message)  # Ensure this function is defined
        sequence = tokenizer.texts_to_sequences([preprocessed_message])
        padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)  # Define max_sequence_length
        prediction = nn_model.predict(padded_sequence)[0]
        return render_template('nn_classifier.html', prediction_text='Email is {}'.format('Spam' if prediction[0] > 0.5 else 'Not Spam'))
    return render_template('nn_classifier.html')

if __name__ == '__main__':
    app.run(debug=True)