# projet_2a_squelettisation
# Estimation et Analyse de la Pose Humaine

Estimation de la pose humaine à l'aide de modèles squelettiques pour analyser le comportement humain. 

## Table des Matières
- [Introduction](#introduction)
- [Fonctionnalités](#fonctionnalités)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Méthodes](#méthodes)
- [Résultats](#résultats)
- [Applications](#applications)

## Introduction
Ce projet, réalisé dans le cadre du projet 2A du Département Informatique à l'École des Mines de Nancy, vise à traduire les données d'estimation de la pose humaine en informations utilisables pour diverses applications telles que la robotique et le sport. Le projet utilise à la fois une approche géométrique et une approche basée sur l'apprentissage automatique pour reconnaître les poses humaines.

## Fonctionnalités
- **Estimation de la Pose Humaine** : Utilisation de YOLOv8 pour la génération de modèles squelettiques.
- **Approche Géométrique** : Définition manuelle des poses en utilisant les propriétés des points clés.
- **Approche d'Apprentissage Automatique** : Reconnaissance automatique des poses à l'aide d'algorithmes d'apprentissage supervisé.
- **Prédictions en Temps Réel** : Mise en œuvre de la détection et de la prédiction de poses en temps réel.
- **Commandes Robotiques** : Traduction des gestes humains en commandes pour robots.
- **Analyse Sportive** : Surveillance et analyse des mouvements athlétiques.

## Installation
1. **Cloner le dépôt** :
    ```bash
    git clone https://github.com/mines-nancy/projet_2a_squelettisation
    cd projet_2a_squelettisation
    ```
3. **Installer les paquets requis** :
    ```bash
    pip install -r requirements.txt
    ```

## Utilisation
1. **Reconnaissance de pose sur image par l'approche géométrique** :
    ```bash
    cd src/manual_approach
    python pose_prediction.py
    ```
    En sauvegardant votre photo de la pose en format .png dans le dossier `src/entries/picture_originals`.
   
2. **Collecte des Données** :
    ```bash
    cd src/machine_learning_approach
    python data_collection.py --class-name <nom de la classe> --collect-for 60 --start-delay 10 --file-name <nom du fichier csv à créer ou à utiliser>
    ```
    
3. **Entraînement du Modèle** :
    ```bash
    python model_training.py --training-data <csv à utiliser> --model-name <nom du modèle que vous voulez entraîner>
    ```
    
4. **Prédiction des Poses** :
    ```bash
    python pose_prediction.py --model-name <modèle à utiliser>
    ```

5. **Exécution des Commandes Robotiques** :
    ```bash
    cd src/applications
    python robot_command.py
    ```
    À condition d'avoir en sa possession un robot LEGO BOOST de bloc central de contrôle MoveHub.

6. **Surveillance de l'Entraînement** :
    ```bash
    python workout_monitoring.py
    ```
    En sauvegardant votre vidéo de pompes dans le dossier `src/entries/video_originals` sous format .mp4.

## Méthodes
### Approche Géométrique
Définit les postures en utilisant les propriétés des points clés en 2D, calculant des angles et des distances pour reconnaître les poses prédéfinies.

### Approche d'Apprentissage Automatique
Collecte des données, entraîne des modèles supervisés (par exemple, Régression Logistique, SVM, RandomForest), et prédit les poses à l'aide des modèles entraînés. Un exemple entraîné est fourni qui permet de reconnaître la danse [YMCA - Village People](https://www.youtube.com/watch?v=CS9OO0S5w2k&ab_channel=VillagePeople).
L'approche automatique a été inspirée par ce projet : [mediapipe-ymca](https://github.com/youngsoul/mediapipe-ymca).

## Résultats
L'approche basée sur l'apprentissage automatique surpasse la méthode géométrique en termes de précision et de robustesse, bien qu'elle nécessite un ensemble de données d'entraînement plus large et plus diversifié. Pour développer davantage ce projet, il serait pertinent d'estimer les poses via une méthode 3D et de prétraiter les données afin d'éliminer les rotations et translations de l'individu, et de normaliser les points afin d'adapter le modèle à des anatomies différentes.

## Applications
### Robotique
Reconnaît les gestes humains pour contrôler un robot construit avec LEGO et MoveHub, permettant des commandes telles que "Avancer", "Tourner à gauche", "Tourner à droite" et "Arrêter". Le robot a été commandé à travers cette librairie : [pylgbst](https://github.com/undera/pylgbst).

### Sport
Surveille les mouvements athlétiques, fournissant des retours en temps réel sur des exercices comme les pompes, les squats et plus encore en utilisant le modèle YOLOv8.
L'analyse de l'exercice est réalisée à l'aide de l'objet AIGym de la bibliothèque [ultralytics.solutions](https://docs.ultralytics.com/guides/workouts-monitoring/).


---

*École des Mines de Nancy - N22 - FICM*

*Juin 2024*
