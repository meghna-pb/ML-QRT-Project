# QRT Data Challenge : Guess the winner

## Description

Ce projet vise à prédire les résultats des matchs de football en utilisant diverses méthodes de Machine Learning, y compris des réseaux de neurones. Les données sont préparées et traitées pour entraîner plusieurs modèles, et des techniques d'explicabilité comme LIME et SHAP sont utilisées pour interpréter les résultats. 

## Structure du Projet

- `GetData.py`: Script pour récupérer et charger les données nécessaires.
- `PrepareData.py`: Script pour préparer et traiter les données avant de les utiliser dans les modèles.
- `main.ipynb`: Notebook Jupyter contenant les étapes d'entraînement des modèles, d'évaluation des performances, et d'explication des résultats.

## Prérequis

Pour exécuter ce projet, vous devez avoir les bibliothèques suivantes installées :

- pandas
- numpy
- scikit-learn
- keras
- keras-tuner
- lime
- shap

Vous pouvez installer ces bibliothèques en utilisant pip :

```bash
pip install pandas numpy scikit-learn keras keras-tuner lime shap
```

## Utilisation

1. **Récupération des données** : Exécutez `GetData.py` pour charger les données nécessaires au projet.

2. **Préparation des données** : Exécutez `PrepareData.py` pour préparer et traiter les données.

3. **Entraînement des modèles et évaluation** : Ouvrez et exécutez les cellules du notebook `main.ipynb`. Ce notebook contient toutes les étapes nécessaires pour entraîner les modèles, les évaluer, et expliquer les résultats.

## Modèles

Plusieurs modèles sont entraînés et comparés :
1. **Clustering** (modèle non supervisé): Modèle simple et interprétable, avec 3 sous-parties (analyse graphique, silhouette score, accuracy)
2. **Régression Logistique** : Modèle simple et interprétable.
3. **Random Forest Classifier** (méthode ensembliste): Modèle plus complexe et performant.
4. **Pipeline de Random Forest Classifiers** (méthode ensembliste): Modèle plus complexe et performant.
5. **Réseau de Neurones** : Modèle très performant mais moins interprétable.

## Importance des features et contributions

Les outils d'explicabilité comme LIME et SHAP sont implémentés pour comprendre et expliquer les prédictions de notre réseau de neurones.
   
### Résultats (se trouvant également dans le main.ipynb)
| Méthode                                          | Accuracy     |
|--------------------------------------------------|--------------|
| Benchmark                                        | 0.4498       |
| Clustering avec 3 clusters (analyse graphique)   | 0.4873       |
| Différenciation de cluster  (silhouette score)   | 0.4357       |
| Différenciation de cluster  (accuracy)           | 0.4832       |
| Régression logistique                            | 0.4890       |
| Random Forest Classifier                         | 0.4934       |
| Pipeline de modèles Random Forest Classifier     | 0.4722       |
| Keras Classifier (Réseau de Neurones)            | 0.5058       |
