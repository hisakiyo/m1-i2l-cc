# Projet apprentissage artificiel

Gauthier MARETTE - M1 I2L APP

*Ce repository contient l'ensemble du code réalisé dans le cadre du projet en apprentissage artificiel.*

## Sommaire
1. Analyse des données fournies et statistiques des bases
2. Algorithmes d'apprentissage utilisés
3. Analyse des performances obtenues
4. Analyse des résultats obtenus

## 1. Analyse des données fournies et statistiques des bases

Lors de ce projet, nous avons pris possession de deux bases de données contenants chacune de nombreuses données à traiter.
Nous allons commencer par la première base:
<p align="center">
  <img src="https://i.ibb.co/JtDnMQC/Sans-titre.png" />
</p>
Nous pouvons voir que la base contient 101 exemple de données et 26 caractéristiques par lignes. Nous constatons également qu'elle est composée de 7 classes différentes au niveau de la colonne à prédire (Z). Il s'agit donc ici d'un problème de classification.

Maintenant, passons à la deuxième base:
<p align="center">
  <img src="https://i.ibb.co/j8b4c6s/image.png" />
</p>
Nous pouvons voir que la base contient 17379 lignes au total, elle a 15 caractéristiques et n'a pas de classe vu qu'il s'agit d'un problème de régression. Ici, la colonne à prédire est toujours "Z".

## 2.  Algorithmes d’apprentissage utilisés

Pour effectuer l'ensemble des tests, j'ai pu utiliser 3 algorithmes afin de faire de la prédiction de données.

### Réseau de neurones
Tout d'abord, j'ai utilisé un réseau de neurones, en classification et en régression.
<p align="center">
  <img src="https://www.juripredis.com/upload/actualites/Machine_learning/reseaux_neurones_feed_forwarded_2.png" />
</p>
Ce système est très efficace mais assez long à entrainer afin d'obtenir des résultats probant.

### K Plus proches voisins
J'ai ensuite utilisé l'algorithme des K plus proches voisins, en classification et en régression.

<p align="center">
  <img src="https://miro.medium.com/max/405/1*0Pqqx6wGDfFm_7GLebg2Hw.png" />
</p>

Cet algorithme utilise un nombre défini de voisin pour définir son état (ici k=3 ou k=7). C'est un algorithme assez efficace et qui ne demande pas beaucoup de ressource machine.

### Arbre de décision

Pour finir, j'ai pu utiliser un arbre de décision, cette fois ci aussi, en classification et en régression.
<p align="center">
  <img width="600px" src="https://miro.medium.com/max/1400/0*PB7MYQfzyaLaTp1n" />
</p>

Il est également très efficace et demande assez peu de ressource machine.

## 3. Analyse des performances obtenues

### Données data 1 

| Algorithme utilisé     | Temps d'éxecution |
|------------------------|-------------------|
| Réseau de neurones     | 0,05s             |
| K plus proches voisins | 0,008s            |
| Arbre de décision      | 0,003s            |

### Données data 2
| Algorithme utilisé     | Temps d'éxecution |
|------------------------|-------------------|
| Réseau de neurones     | 1,35s             | 
| K plus proches voisins | 0,14s             | 
| Arbre de décision      | 0,05s             |

## 4. Analyse des résultats obtenus

### Données data 1 (Classification)

| Algorithme utilisé     | Train score		 | Test score
|------------------------|-------------------|-
| Réseau de neurones     | 1                 | 0.3870
| K plus proches voisins | 0.5857            | 0.3548
| Arbre de décision      | 1                 | 0.9355

Ci-dessus, le tableau contenant toutes les statistiques. Le test score est le score obtenu une fois que le modèle a été entrainé par les données de train.
On peut donc voir ici, que pour les données 1, l'arbre de décision est l'algorithme le plus adapté car c'est celui qui a la plus grosse efficacité en étant celui qui prend le moins de temps à s'exécuter. (93.55% de réussite en prédiction pour 0,003s d'exécution)

### Données data 2 (Régression)
| Algorithme utilisé     | Train score		 | Test score| MAE | MSE | R2
|------------------------|-------------------|------------|-----|-----|---
| Réseau de neurones     | 1                 | 0.9998|1.51|4.37|0.99
| K plus proches voisins | 0.5857            | 0.9832|15.27|553.84|0.98
| Arbre de décision      | 1                 | 0.9991|2.56|27.32|0.99

Ci-dessus, le tableau contenant toutes les statistiques pour les données du dataset 2. Plus le MAE et le MSE sont bas, plus efficace est l'algorithme. 
On peut voir ici que l'algorithme qui est le plus efficace est le réseau de neurone, il arrive à prédire plus de 99.98% des données une fois entrainé. Cependant, c'est celui qui prend le plus de temps à s'exécuter. Je pense que l'arbre de décision est ici un meilleur choix, car même s'il n'est pas aussi fiable que le réseau de neurone, son temps d'exécution est drastiquement inférieur (27x plus rapide).