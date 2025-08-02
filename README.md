# Email Spam Detector – BMSecurity

## Description du projet
Ce projet permet de détecter automatiquement les spams dans les emails grâce au NLP et au Machine Learning. Il inclut un notebook complet, un modèle sauvegardé et une interface utilisateur Streamlit.

## Structure des fichiers
```
. 
├── models/                  # Modèles et vectorizer sauvegardés
├── Notebook_Spam_Classifier_Clean.ipynb  # Notebook principal
├── app.py                   # Interface utilisateur Streamlit
├── requirements.txt         # Dépendances Python
└── README.md                # Documentation
```

## 🚀 Installation rapide

1. **Cloner le projet**
   ```bash
   git clone https://github.com/votre-utilisateur/email-spam-detector-nlp-ml.git
   cd email-spam-detector-nlp-ml
   ```

2. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

3. **Lancer l'application Streamlit**
   ```bash
   streamlit run app.py
   ```

4. **Explorer le pipeline complet**
   - Ouvrez le notebook `Notebook_Spam_Classifier_Clean.ipynb` pour voir toutes les étapes (EDA, NLP, entraînement, sauvegarde du modèle).
