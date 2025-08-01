{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "#VSC-3e01991d",
        "language": "markdown"
      },
      "source": [
        "# 📧 Système Intelligent de Filtrage des Emails (Spam vs Ham)",
        "",
        "Ce projet vise à développer un système intelligent capable de filtrer les emails en détectant s’ils sont des spams ou non (ham), en utilisant des techniques de traitement automatique du langage (NLP) et de classification supervisée.",
        "",
        "",
        "## 📦 1. Importation des bibliothèques",
        "",
        "Dans cette section, nous importons toutes les bibliothèques nécessaires au projet : traitement des données, visualisations, NLP, Machine Learning et sauvegarde de modèles. Ces outils vont nous permettre de structurer le pipeline de classification d’emails de manière rigoureuse et efficace.",
        "",
        ""
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "#VSC-456c9719",
        "language": "python"
      },
      "source": [
        "# ===============================",
        "# 1. 📦 Importation des bibliothèques",
        "# ===============================",
        "import pandas as pd",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns",
        "import nltk",
        "import re",
        "import string",
        "from nltk.corpus import stopwords",
        "from nltk.stem import PorterStemmer",
        "from nltk.tokenize import word_tokenize",
        "from wordcloud import WordCloud",
        "from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV",
        "from sklearn.feature_extraction.text import TfidfVectorizer",
        "from sklearn.metrics import classification_report, confusion_matrix",
        "from sklearn.naive_bayes import MultinomialNB",
        "from sklearn.tree import DecisionTreeClassifier",
        "from sklearn.svm import SVC",
        "import joblib"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "#VSC-51af45fa",
        "language": "markdown"
      },
      "source": [
        "",
        "",
        "## 📥 2. Chargement et nettoyage des données",
        "",
        "Nous chargeons le jeu de données contenant les emails et leurs étiquettes (spam ou ham), puis procédons à un nettoyage de base :",
        "- Vérification des colonnes",
        "- Suppression des doublons",
        "- Suppression des valeurs manquantes",
        "",
        "Cela permet de garantir la qualité des données avant toute analyse ou modélisation.",
        ""
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "#VSC-fb5fb391",
        "language": "python"
      },
      "source": [
        "# ===============================",
        "# 2. 📥 Chargement & Nettoyage des données",
        "# ===============================",
        "df = pd.read_csv(\"../data/DataSet_Emails.csv\")",
        "# delete duplicated lines",
        "",
        "print(\"Initial shape of the dataset:\", df.shape)",
        "print(\"nb de lighne dup\", df.duplicated().sum())",
        "df.drop_duplicates(inplace=True)",
        "# Drop rows with NaN ",
        "nb= df.shape[0]",
        "print(\"le nb du nan\", df.isnull().sum().sum())",
        "df.dropna(inplace=True)",
        "print(\"lignes supprimer :\", nb-df.shape[0])"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "#VSC-5367dc53",
        "language": "markdown"
      },
      "source": [
        "",
        "---",
        "",
        "## 📊 3. Analyse exploratoire des données (EDA)",
        "",
        "L’EDA permet de comprendre la structure du jeu de données :",
        "- Dimensions, types, aperçu des premières lignes",
        "- Répartition des classes (spam vs ham)",
        "- Analyse visuelle à travers des nuages de mots pour identifier les termes fréquents dans chaque type d’email",
        "",
        "Cette étape est cruciale pour orienter les choix de modélisation.",
        ""
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "#VSC-5774b18a",
        "language": "python"
      },
      "source": [
        "# ===============================",
        "# 3. 📊 Analyse exploratoire (EDA)",
        "# ===============================",
        "# dimensions et aperçu des données",
        "print(\"Dimensions:\", df.shape)",
        "",
        "df.head()",
        ""
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "#VSC-c879c2fd",
        "language": "python"
      },
      "source": [
        "nb= df['label_text'].value_counts()",
        "print(\"Distribution des emails:\\n\", nb)",
        "# distribution des classes",
        "sns.countplot(x='label', data=df)",
        "plt.title(\"Distribution des classes\")",
        "plt.show()",
        ""
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "#VSC-f3c3668d",
        "language": "python"
      },
      "source": [
        "",
        "# ===============================",
        "# generation des nuages de mots",
        "# ===============================",
        "def generate_wordcloud(data, label):",
        "    text = \" \".join(data[data['label_text'] == label]['text'])",
        "    wc = WordCloud(width=800, height=400, background_color='white').generate(text)",
        "    plt.imshow(wc, interpolation='bilinear')",
        "    plt.axis('off')",
        "    plt.title(f\"WordCloud - {label}\")",
        "    plt.show()",
        "generate_wordcloud(df, 'spam')",
        "generate_wordcloud(df, 'ham')"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "#VSC-c60a29cf",
        "language": "markdown"
      },
      "source": [
        "",
        "---",
        "",
        "## 🧽 4. Prétraitement des textes (NLP)",
        "",
        "Avant de vectoriser les emails, nous appliquons plusieurs transformations textuelles :",
        "- Conversion en minuscules",
        "- Tokenisation",
        "- Suppression des stopwords, de la ponctuation et des caractères spéciaux",
        "- Stemming (réduction à la racine des mots)",
        "",
        "Ces étapes permettent de normaliser le texte et d'améliorer la qualité de la représentation vectorielle.",
        ""
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "#VSC-97b0c11d",
        "language": "python"
      },
      "source": [
        "# ===============================",
        "# 4. 🧽 Prétraitement du texte (NLP)",
        "# ===============================",
        "nltk.download('punkt')",
        "nltk.download('stopwords')",
        "",
        "stemmer = PorterStemmer()",
        "stop_words = set(stopwords.words('english'))",
        "",
        "def preprocess(text):",
        "    text = text.lower()",
        "    text = re.sub(r'\\W', ' ', text)",
        "    tokens = word_tokenize(text)",
        "    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words and word not in string.punctuation]",
        "    return ' '.join(tokens)",
        "",
        "df['clean_text'] = df['text'].apply(preprocess)",
        "df.dropna()",
        "df.drop_duplicates()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "#VSC-7c885c1e",
        "language": "markdown"
      },
      "source": [
        "",
        "---",
        "",
        "## 🔢 5. Extraction des caractéristiques",
        "",
        "Nous utilisons la méthode TF-IDF (Term Frequency-Inverse Document Frequency) pour convertir les textes en vecteurs numériques exploitables par les modèles de Machine Learning.",
        "",
        "Nous séparons ensuite les données en variables explicatives `X` et variable cible `y`, puis divisons le jeu de données en ensemble d'entraînement et de test.",
        ""
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "#VSC-50d4e47e",
        "language": "python"
      },
      "source": [
        "# ===============================",
        "# 5. 🔢 Extraction des caractéristiques",
        "# ===============================",
        "vectorizer = TfidfVectorizer()",
        "X = vectorizer.fit_transform(df['clean_text'])",
        "y = df['label']",
        "",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)",
        ""
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "#VSC-32d34016",
        "language": "markdown"
      },
      "source": [
        "",
        "---",
        "",
        "## 🤖 6. Entraînement de plusieurs modèles",
        "",
        "Nous testons différents algorithmes de classification supervisée :",
        "- Naive Bayes",
        "- Arbre de Décision (Decision Tree)",
        "- SVM (Support Vector Machine)",
        "",
        "Le but est de comparer leurs performances afin de sélectionner le plus performant.",
        ""
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "#VSC-06bad610",
        "language": "python"
      },
      "source": [
        "# ===============================",
        "# 6. 🤖 Entraînement de plusieurs modèles",
        "# ===============================",
        "models = {",
        "    'NaiveBayes': MultinomialNB(),",
        "    'SVM': SVC(kernel='linear'),",
        "    'DecisionTree': DecisionTreeClassifier()",
        "}",
        "",
        "for name, model in models.items():",
        "    model.fit(X_train, y_train)",
        "    y_pred = model.predict(X_test)",
        "    print(f\"\\n{name} Results:\")",
        "    print(classification_report(y_test, y_pred))",
        ""
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "#VSC-ee36cf3d",
        "language": "markdown"
      },
      "source": [
        "",
        "---",
        "",
        "## 📈 7. Évaluation des modèles",
        "",
        "À l’aide de métriques standards :",
        "- Matrice de confusion",
        "- Précision",
        "- Rappel",
        "- F1-score",
        "",
        "Nous évaluons la qualité de chaque modèle sur le jeu de test, afin d’avoir une vision claire de leurs performances respectives.",
        ""
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "#VSC-52baa626",
        "language": "markdown"
      },
      "source": [
        "",
        "---",
        "",
        "## 🔁 8. Validation croisée (Cross-validation)",
        "",
        "Pour renforcer la robustesse de l’évaluation, nous utilisons une validation croisée à 5 plis. Cela permet de tester le modèle sur différentes sous-parties du dataset et de vérifier sa stabilité.",
        ""
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "#VSC-2458d1d9",
        "language": "python"
      },
      "source": [
        "# ===============================",
        "# 8. 🔁 Validation croisée",
        "# ===============================",
        "from sklearn.model_selection import cross_val_score",
        "for name, model in models.items():",
        "    scores = cross_val_score(model, X_train, y_train , cv=5 , scoring='f1')",
        "    print(f\"{name }: f1 score = {scores.mean():.3f} (+/- {scores.std():.3f})\")",
        ""
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "#VSC-9853126c",
        "language": "markdown"
      },
      "source": [
        "",
        "---",
        "",
        "## 🔧 9. Optimisation des hyperparamètres",
        "",
        "Nous utilisons `GridSearchCV` pour optimiser les paramètres du modèle SVM. Cette méthode permet de tester plusieurs combinaisons de paramètres pour identifier celle qui donne les meilleurs résultats.",
        ""
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "#VSC-d1d4be75",
        "language": "python"
      },
      "source": [
        "from sklearn.model_selection import RandomizedSearchCV",
        "from sklearn.svm import SVC",
        "",
        "# Define your parameter distributions",
        "param_dist = {",
        "    'C': [0.1, 1, 10],                     # or use continuous: e.g. uniform(0.1, 10)",
        "    'kernel': ['linear', 'rbf'],",
        "    'gamma': ['scale', 0.01, 0.001]",
        "}",
        "",
        "# Setup RandomizedSearchCV",
        "random_search = RandomizedSearchCV(",
        "    estimator=SVC(),",
        "    param_distributions=param_dist,",
        "    n_iter=6,           # number of random combinations to try",
        "    cv=5,",
        "    scoring='accuracy',",
        "    random_state=42,",
        "    verbose=2,",
        "    n_jobs=-1",
        ")",
        "",
        "random_search.fit(X_train, y_train)",
        "",
        "print(\"Best SVM Parameters:\", random_search.best_params_)",
        "print(\"Best Cross‑Val Score:\", random_search.best_score_)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "#VSC-a5946e29",
        "language": "markdown"
      },
      "source": [
        "",
        "---",
        "",
        "## 💾 10. Sauvegarde du meilleur modèle",
        "",
        "Le meilleur modèle et le vectoriseur TF-IDF sont sauvegardés avec `joblib`, afin de pouvoir être réutilisés dans une application (Streamlit par exemple) sans devoir les réentraîner.",
        ""
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "#VSC-06a329f5",
        "language": "python"
      },
      "source": [
        "# ===============================",
        "# 10. 💾 Sauvegarde du meilleur modèle",
        "# ===============================",
        "best_model = random_search.best_estimator_",
        "joblib.dump(best_model, '../models/model.pkl')",
        "joblib.dump(vectorizer, '../models/vectorizer.pkl')"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "#VSC-b4041978",
        "language": "markdown"
      },
      "source": [
        "",
        "---",
        "",
        "## 🌐 11. Interface utilisateur avec Streamlit",
        "",
        "Nous créerons une interface simple avec Streamlit permettant à l'utilisateur d’entrer un texte d’email et d’obtenir en retour une prédiction :",
        "- Zone de saisie de texte",
        "- Bouton \"Analyser\"",
        "- Affichage clair : Spam ou Ham",
        "- (Optionnel) Affichage de la probabilité",
        "",
        "Cette interface permettra une utilisation interactive du modèle en production.",
        "",
        ""
      ]
    }
  ]
}