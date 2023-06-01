import pickle
import string
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def Clean_text(text):
    lower_text = text.lower()
    word_text = nltk.word_tokenize(lower_text)
    stop = stopwords.words('english')
    punctuations = list(string.punctuation)
    stop = stop + punctuations
    ps = PorterStemmer()
    output_words = []
    for w in word_text:
        if w not in stop:
            output_words.append(ps.stem(w))
    return " ".join(output_words)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('Model.pkl', 'rb'))

st.title('EMAIL / SMS -  SPAM CLASSIFIER')

input_sms = st.text_area("Enter the message...")
if st.button('Predict'):
        # 1. preprocess
        Cleaned_sms = Clean_text(input_sms)
        # 2. vectorize
        vectorized_sms = tfidf.transform([Cleaned_sms])
        # 3. predict
        result = model.predict(vectorized_sms)[0]
        # 4. display
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
