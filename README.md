# Description du projet

Ce projet propose une résolution du problème du sac à dos (`0/1 knapsack`) à l'aide d'une métaheuristique de type colonie de fourmis (`Ant Colony Optimization`, ou ACO), ainsi qu'une variante enrichie par une couche de machine learning.

L'objectif est double :

- disposer d'un solveur ACO de base correct, reproductible et testable ;
- comparer ce solveur à une version ACO + ML qui ajuste dynamiquement certains paramètres de recherche.

Le dépôt est organisé pour permettre à la fois l'expérimentation, la comparaison entre variantes et l'extension future vers des benchmarks plus poussés.

## Modélisation du problème

Le problème traité est le problème du sac à dos binaire :

- on dispose d'une capacité maximale ;
- on dispose d'un ensemble d'objets, chacun ayant une valeur et un poids ;
- chaque objet peut être pris ou non, mais au plus une fois ;
- l'objectif est de maximiser la valeur totale sans dépasser la capacité.

Dans le code, cette modélisation repose principalement sur les types suivants :

- `KnapsackItem` : représente un objet avec sa `value`, son `weight` et éventuellement un nom ;
- `KnapsackInstance` : représente une instance complète avec une `capacity` et une collection d'objets ;
- `Solution` : représente une solution construite par l'algorithme, avec les indices sélectionnés, la valeur totale, le poids total et l'information de faisabilité.

Cette modélisation permet de séparer clairement :

- les données du problème ;
- les paramètres de l'algorithme ;
- les résultats d'exécution ;
- les métriques de comparaison entre solveurs.

## Métaheuristique de base

La première version du solveur est une implémentation ACO dédiée au problème du sac à dos.

Son fonctionnement général est le suivant :

1. Initialiser un niveau de phéromones pour chaque objet.
2. Construire plusieurs solutions faisables à chaque itération.
3. Évaluer les solutions obtenues.
4. Faire évaporer les phéromones.
5. Déposer de nouvelles phéromones en fonction des bonnes solutions observées.
6. Répéter ce cycle pendant un nombre fixé d'itérations.

La construction de solution est contrainte par la capacité :

- seuls les objets encore sélectionnables et compatibles avec la capacité restante peuvent être ajoutés ;
- le choix d'un objet dépend à la fois de la phéromone associée et d'une heuristique fondée sur le ratio valeur/poids ;
- la meilleure solution trouvée est mémorisée au fil des itérations.

Cette version sert de baseline pour toutes les comparaisons.

## Machine Learning

La seconde version conserve la même boucle ACO, mais introduit une adaptation dynamique des paramètres de recherche.

### Choix

Le choix fait dans ce projet est volontairement léger et interprétable :

- utiliser `scikit-learn` ;
- apprendre à prédire les paramètres `alpha`, `beta` et `evaporation` ;
- ne pas remplacer l'ACO par un modèle, mais utiliser le ML comme mécanisme d'ajustement du comportement de recherche.

Le modèle actuellement utilisé est une régression multi-sortie. Si `scikit-learn` n'est pas disponible, une logique de repli simple est prévue pour éviter de bloquer complètement la prédiction.

### Description du rôle

Le rôle du machine learning dans ce projet est de piloter dynamiquement les paramètres de l'ACO en fonction de l'état courant de la recherche.

La variante `MLTunedKnapsackACOSolver` :

- observe les solutions produites lors des dernières itérations ;
- extrait un petit ensemble de caractéristiques numériques ;
- prédit de nouveaux paramètres ACO ;
- réinjecte ces paramètres dans la suite de l'exécution.

L'idée n'est donc pas d'apprendre directement la solution optimale, mais d'ajuster la stratégie d'exploration/exploitation en cours d'exécution.

### Données utilisées

Les données ML actuellement utilisées dans le dépôt sont des enregistrements structurés de type `TrainingRecord`.

Chaque enregistrement contient :

- `remaining_capacity_ratio` ;
- `diversity_ratio` ;
- `normalized_best_value` ;
- `stagnation_ratio` ;
- `mean_value_ratio` ;
- les paramètres cibles `alpha`, `beta` et `evaporation`.

Dans l'état actuel du projet, ces données sont synthétiques et servent surtout à :

- valider l'API ;
- tester l'adaptation dynamique ;
- garantir que la comparaison baseline vs ML est techniquement exploitable.

## Implémentation

L'implémentation principale se trouve dans le package `src/metaheuristique/`.

Les éléments importants sont :

- `types.py` : structures de données partagées (`KnapsackItem`, `KnapsackInstance`, `Solution`, `ACOParams`, `RunResult`, `ComparisonSummary`) ;
- `knapsack_aco.py` : solveur ACO de base ;
- `ml.py` : extraction de features et modèle de réglage des paramètres ;
- `knapsack_aco_ml.py` : solveur ACO enrichi par ML ;
- `comparison.py` : exécution batch, récupération des résultats bruts et comparaison entre variantes.

Le projet contient également un script d'exécution :

- `scripts/compare_acos.py`

Ce script permet de :

- lancer les deux variantes sur une instance de démonstration ;
- utiliser les mêmes seeds, le même budget d'itérations et la même taille de colonie ;
- récupérer les résultats run par run ;
- afficher un résumé global et un détail par seed.

## Expérimentations

Les expérimentations sont pour l'instant structurées autour des tests automatisés et du script de comparaison.

Les tests couvrent notamment :

- un micro-cas déterministe dont la solution optimale attendue est connue ;
- la reproductibilité du baseline pour une seed donnée ;
- les cas limites comme une capacité nulle ou une liste d'objets vide ;
- l'entraînement minimal de la partie ML ;
- la validité des paramètres prédits ;
- la comparaison batch entre baseline et variante ML.

Le script `scripts/compare_acos.py` permet de produire une sortie de comparaison du type :

- meilleure valeur atteinte ;
- valeur moyenne sur plusieurs exécutions ;
- écart-type ;
- nombre de solutions invalides ;
- temps total d'exécution ;
- écart baseline / ML pour chaque seed.

À ce stade, ces expérimentations servent d'abord à comparer proprement les deux approches, et non à affirmer de manière générale que la variante ML est meilleure.

## Analyse et limites

### Apports

Ce dépôt apporte déjà plusieurs éléments utiles :

- une base Python claire pour expérimenter l'ACO sur le problème du sac à dos ;
- une séparation nette entre solveur baseline, solveur ML et couche de comparaison ;
- une exécution reproductible grâce à l'utilisation explicite des seeds ;
- une API qui permet de récupérer les résultats bruts puis de les comparer proprement ;
- une première intégration ML qui reste simple à comprendre et à faire évoluer.

### Limite

Le projet présente aussi plusieurs limites importantes :

- les données d'entraînement ML sont actuellement synthétiques et peu nombreuses ;
- la variante ML n'est pas encore validée sur un vrai corpus de benchmarks ;
- la comparaison actuelle porte surtout sur l'infrastructure de test et non sur une étude statistique approfondie ;
- les instances utilisées restent modestes ;
- aucun claim de supériorité robuste de l'approche ML ne doit être tiré de l'état actuel du dépôt.

### Pistes d'améliorations

Parmi les évolutions naturelles du projet :

- ajouter des jeux d'instances plus variés et plus grands ;
- générer ou collecter des données d'entraînement plus réalistes ;
- tester d'autres modèles de prédiction de paramètres ;
- étendre les métriques de comparaison ;
- ajouter un mode benchmark plus complet avec export des résultats ;
- étudier plus finement l'influence de `alpha`, `beta`, `evaporation` et `tuning_interval`.

# Execution

Installation des dépendances :

```powershell
python -m pip install -e .
```

Lancer les tests :

```powershell
python -m pytest
```

Lancer la comparaison baseline vs ACO + ML :

```powershell
python scripts/compare_acos.py
```

Exemple avec paramètres personnalisés :

```powershell
python scripts/compare_acos.py --iterations 40 --colony-size 12 --seeds 7 17 23 31 --tuning-interval 1
```

# Structure du dépôt

```text
.
|-- .github/
|   `-- agents/
|       `-- knapsack-aco-ml.agent.md
|-- scripts/
|   `-- compare_acos.py
|-- src/
|   `-- metaheuristique/
|       |-- __init__.py
|       |-- comparison.py
|       |-- knapsack_aco.py
|       |-- knapsack_aco_ml.py
|       |-- ml.py
|       `-- types.py
|-- tests/
|   |-- conftest.py
|   |-- test_comparison.py
|   |-- test_knapsack_aco.py
|   `-- test_knapsack_aco_ml.py
|-- .gitignore
|-- pyproject.toml
`-- README.md
```
