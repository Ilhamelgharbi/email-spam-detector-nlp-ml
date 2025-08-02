# 📄 Rapport Détaillé – Système Intelligent de Filtrage des Emails  
**Projet BMSecurity – Ilham Elgharbi**

---

## 1. Présentation du Projet

Ce projet a pour objectif de développer une solution intelligente de **filtrage automatique des emails** afin de distinguer les messages **spams** des **hams** (emails légitimes). Il combine des techniques de **traitement du langage naturel (NLP)** avec des **modèles d’apprentissage supervisé** pour atteindre une haute performance de classification. Une application interactive a été développée à l’aide de **Streamlit**.

---

## 2. Méthodologie Générale

Le projet suit un pipeline complet :

- 🔍 **Analyse exploratoire (EDA)**  
- 🧹 **Prétraitement des données textuelles**
- 📐 **Vectorisation TF-IDF**
- 🤖 **Entraînement de plusieurs modèles**
- 📊 **Évaluation & validation croisée**
- 🧪 **Optimisation des hyperparamètres**
- 💾 **Sauvegarde du modèle et du vectorizer**
- 🖥️ **Interface utilisateur (Streamlit)**

---

## 3. Analyse Exploratoire

- **Dimensions du dataset** : nombre total d'emails, équilibre des classes
- **Distribution spam vs ham** : histogrammes, ratio
- **Statistiques descriptives** : longueurs des messages
- **Wordclouds** : visualisation des mots fréquents dans les spams et les hams

---

## 4. Prétraitement des Textes

- Suppression des doublons et emails vides
- Passage en minuscules
- Suppression de la ponctuation et des caractères spéciaux
- Tokenisation
- Suppression des stopwords (`nltk.corpus.stopwords`)
- Racinisation (stemming) avec `PorterStemmer`

---

## 5. Vectorisation

Les textes prétraités sont transformés en vecteurs numériques avec **TF-IDF** :

- `TfidfVectorizer(max_features=3000)`
- Réduction du bruit et mise en valeur des mots pertinents

---

## 6. Modélisation

### Modèles testés :

| Modèle              | Remarques                              |
|---------------------|----------------------------------------|
| Naive Bayes         | Simple, rapide, efficace pour le texte |
| Decision Tree       | Interprétable mais moins performant    |
| SVM (LinearSVC)     | Meilleur compromis performance/robustesse |

---

## 7. Évaluation et Résultats

### Métriques sur le modèle SVM optimisé :

- **Précision** : ~0.98  
- **Rappel** : ~0.97  
- **F1-score** : ~0.975  
- **Validation croisée** : 5-fold CV, scores stables

> Le modèle SVM linéaire offre le meilleur équilibre entre précision et généralisation. Il surpasse Naive Bayes et Decision Tree sur ce jeu de données.

---

## 8. Optimisation

Optimisation du modèle SVM via **RandomizedSearchCV** :

- Hyperparamètre exploré : `C` (régularisation)
- Nombre d’itérations : 20
- Validation croisée intégrée

---

## 9. Sauvegarde et Reproductibilité

- Le modèle (`model.pkl`) et le vectorizer (`vectorizer.pkl`) sont sauvegardés dans le dossier `models/`
- Toutes les étapes du projet sont documentées dans le notebook `Notebook_Spam_Classifier_Clean.ipynb`
- Reproductibilité assurée via environnement Python standard

---

## 10. Interface Utilisateur (Streamlit)

- Application web permettant de tester un email en direct
- Interface simple : champ texte + bouton "Prédire"
- Résultat instantané : `✅ Ham` ou `🚫 Spam`
- Visualisation intégrée : nuages de mots, histogrammes

---


## 12. Technologies Utilisées

| Catégorie           | Outils / Bibliothèques                 |
|---------------------|----------------------------------------|
| Langage             | Python 3                               |
| NLP                 | NLTK                                   |
| Modélisation        | Scikit-learn                           |
| Interface           | Streamlit                              |
| Visualisation       | Matplotlib, Seaborn                    |
| Sérialisation       | Joblib                                 |

---

## 13. Auteur

**Ilham Elgharbi**  
Projet réalisé dans le cadre du bootcamp IA – **BMSecurity**  
📍 Maroc – 2025

---

## 14. Synthèse Pédagogique (Bonus)

### Pourquoi ce pipeline fonctionne-t-il bien ?
- Le **nettoyage et la vectorisation TF-IDF** améliorent fortement la qualité des entrées.
- Le **SVM** est particulièrement adapté aux données textuelles à forte dimension.
- L’**évaluation croisée** garantit la fiabilité des résultats sur des données nouvelles.

### Pourquoi recommander le SVM ?
- Il offre une **excellente précision** et **peu de sur-apprentissage**.
- Son temps d’inférence est rapide, ce qui le rend **adapté à la production**.

---


## 15. Rapport sur l’application Streamlit (`app.py`)

L’application Streamlit développée dans `app.py` permet de rendre le modèle accessible et interactif :

- **Interface utilisateur** : formulaire simple pour saisir un email, bouton de prédiction, affichage immédiat du résultat (`✅ Ham` ou `🚫 Spam`).
- **Visualisations intégrées** : histogrammes de distribution, nuages de mots pour explorer les données directement dans l’application.
- **Expérience pédagogique** : l’utilisateur peut tester différents textes et observer l’impact du contenu sur la prédiction, ce qui facilite la compréhension du modèle.
- **Déploiement** : l’application peut être lancée localement ou déployée sur le web, rendant le système utilisable par des non-techniciens.
- **Robustesse** : le modèle et le vectorizer sont chargés depuis les fichiers sauvegardés, garantissant la cohérence entre l’entraînement et l’inférence.

Cette application est un atout majeur pour la démonstration, la validation et l’adoption du système de filtrage intelligent.

---

## 16. Conclusion

Le **Système Intelligent de Filtrage des Emails** développé démontre une forte efficacité dans la détection des spams. Il peut être rapidement intégré dans une solution réelle grâce à la modularité de son code, la robustesse du modèle, et l’interface utilisateur simple et claire (notamment via l’application Streamlit).

---

