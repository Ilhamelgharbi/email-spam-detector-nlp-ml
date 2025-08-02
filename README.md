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

## Instructions pour l’exécution
1. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
2. Lancer l’application :
   ```bash
   streamlit run app.py
   ```
3. Ouvrir le notebook pour explorer le pipeline complet.
