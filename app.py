# ===============================
# 11. 🌐 Interface utilisateur Streamlit (exemple robuste)
# ===============================

import streamlit as st
import joblib
import re
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import pandas as pd

# Historique de session
if 'history' not in st.session_state:
    st.session_state['history'] = []

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words and word not in string.punctuation]
    return ' '.join(tokens)

def load_model_and_vectorizer(model_name):
    if model_name == 'Naive Bayes':
        model = joblib.load('models/model_nb.pkl')
    elif model_name == 'Decision Tree':
        model = joblib.load('models/model_dt.pkl')
    else:
        model = joblib.load('models/model.pkl')
    vectorizer = joblib.load('models/vectorizer.pkl')
    return model, vectorizer



st.title("🛡️ Système Intelligent de Filtrage d'Emails")
st.markdown("""
Bienvenue sur l'interface de détection de spams. Saisissez un email ou chargez un fichier pour tester le modèle.\
Ce projet combine NLP et Machine Learning pour la sécurité des communications.
""")

tab1, tab2 = st.tabs(["Prédiction", "Exploration des données (EDA)"])

with tab1:
    st.subheader("Analyse d'un email unique")
    model_choice = st.selectbox("Choisissez le modèle :", ["SVM (par défaut)", "Naive Bayes", "Decision Tree"])
    email = st.text_area("Entrez le texte de l'email à analyser", height=200)
    if st.button("Analyser") and email:
        with st.spinner("Analyse en cours..."):
            try:
                model, vectorizer = load_model_and_vectorizer(model_choice)
                clean = preprocess_text(email)
                X = vectorizer.transform([clean])
                pred = model.predict(X)[0]
                result = 'Spam' if pred == 1 else 'Ham'
                color = 'red' if pred == 1 else 'green'
                st.markdown(f"<h2 style='color:{color};'>Résultat : {result}</h2>", unsafe_allow_html=True)
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)[0][pred]
                    st.info(f"Score de confiance : {proba:.2f}")
                if pred == 1:
                    st.warning("Ce message est classé comme spam car il présente des caractéristiques similaires à des spams connus.")
                else:
                    st.success("Ce message est considéré comme légitime (ham).")
                # Ajout à l'historique
                st.session_state['history'].append({
                    'texte': email,
                    'résultat': result,
                    'score': float(proba) if 'proba' in locals() else None,
                    'modèle': model_choice
                })
            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {e}")


# Onglet EDA : visualisation des graphiques et wordclouds
with tab2:
    st.subheader("Exploration des données (EDA)")
    try:
        df = pd.read_csv('data/DataSet_Emails.csv')
        st.write("Aperçu du jeu de données :", df.head())
        st.write("Distribution des classes :")
        st.bar_chart(df['label_text'].value_counts())
        import matplotlib.pyplot as plt
        from wordcloud import WordCloud
        import io
        # Wordcloud spam
        st.write("Nuage de mots - Spam")
        spam_texts = df[df['label_text'] == 'spam']['text'].dropna().astype(str)
        text_spam = " ".join(spam_texts)
        wc_spam = WordCloud(width=800, height=400, background_color='white').generate(text_spam)
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        ax1.imshow(wc_spam, interpolation='bilinear')
        ax1.axis('off')
        st.pyplot(fig1)
        # Wordcloud ham
        st.write("Nuage de mots - Ham")
        ham_texts = df[df['label_text'] == 'ham']['text'].dropna().astype(str)
        text_ham = " ".join(ham_texts)
        wc_ham = WordCloud(width=800, height=400, background_color='white').generate(text_ham)
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.imshow(wc_ham, interpolation='bilinear')
        ax2.axis('off')
        st.pyplot(fig2)
    except Exception as e:
        st.error(f"Erreur lors de l'affichage EDA : {e}")


