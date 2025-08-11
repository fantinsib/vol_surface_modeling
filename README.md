# vol_surface_modeling

Volatility Surface Modeling
Cette application interactive en Python permet de modéliser une surface de volatilité à partir de données d’options calls. Elle utilise notamment :

- Le calibrage de smiles de volatilité avec le modèle SVI (Stochastic Volatility Inspired)

- L’extrapolation des paramètres en une surface complète via le modèle de Dupire

Contenu du projet

- main.py : point d’entrée de l’application Streamlit

- notebook.ipynb : carnet de notes détaillant les étapes de modélisation et de calibration

- tools.py et tools_dupire.py : fonctions auxiliaires pour le calibrage, interpolation et visualisation
- sample_data.csv : des données de test synthétiques

Pour exécuter l'application, exécuter dans un terminal : 

- streamlit run main.py
