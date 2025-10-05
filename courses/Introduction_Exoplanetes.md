---
title: Introduction aux Exoplanètes
level: Débutant
author: RAY AUTRA TEAM
duration: 1 heure
description: Découvrez les bases de la détection et de l'analyse des exoplanètes avec la mission Kepler
---

# Introduction aux Exoplanètes

## Qu'est-ce qu'une Exoplanète ?

Une **exoplanète** (ou planète extrasolaire) est une planète qui orbite autour d'une étoile autre que notre Soleil. La première exoplanète confirmée autour d'une étoile similaire au Soleil a été découverte en 1995.

### Points Clés
- Plus de 5000 exoplanètes confirmées à ce jour
- Découvertes grâce à différentes méthodes de détection
- Certaines pourraient abriter la vie

## La Mission Kepler

Le télescope spatial **Kepler** (2009-2018) a révolutionné notre compréhension des exoplanètes.

### Objectifs de Kepler
1. Déterminer la fréquence des planètes de type terrestre
2. Identifier les planètes dans la zone habitable
3. Caractériser les propriétés des systèmes planétaires

### Méthode de Détection : Le Transit

Kepler utilise la **méthode du transit** :
- Observation de la luminosité des étoiles
- Détection de baisses périodiques de luminosité
- Ces baisses indiquent le passage d'une planète devant l'étoile

```
Luminosité ↑
    |     ___________     ___________
    |    /           \   /           \
    |___/             \_/             \___
    |---------------------------------> Temps
         Transit 1        Transit 2
```

## Paramètres Importants

### Paramètres Orbitaux

**koi_period** : Période orbitale
- Temps pour une orbite complète
- Mesuré en jours
- Exemple : Terre = 365.25 jours

**koi_duration** : Durée du transit
- Temps que la planète passe devant l'étoile
- Mesuré en heures
- Dépend de la taille de la planète et de l'étoile

**koi_depth** : Profondeur du transit
- Diminution de la luminosité stellaire
- Mesuré en parties par million (ppm)
- Indique la taille relative de la planète

### Paramètres Planétaires

**koi_prad** : Rayon planétaire
- Taille de la planète
- Mesuré en rayons terrestres (R⊕)
- Terre = 1.0 R⊕, Jupiter ≈ 11.2 R⊕

**koi_teq** : Température d'équilibre
- Température théorique de la planète
- Mesuré en Kelvin
- Dépend de la distance à l'étoile

**koi_insol** : Flux d'insolation
- Quantité d'énergie reçue de l'étoile
- Flux terrestre = 1.0
- Détermine la zone habitable

### Paramètres Stellaires

**koi_steff** : Température effective stellaire
- Température de surface de l'étoile
- Mesuré en Kelvin
- Soleil ≈ 5778 K

**koi_srad** : Rayon stellaire
- Taille de l'étoile hôte
- Mesuré en rayons solaires (R☉)
- Soleil = 1.0 R☉

**koi_slogg** : Gravité de surface stellaire
- Logarithme de la gravité de surface
- Indique l'évolution stellaire
- Soleil ≈ 4.44

## Classification des Objets Kepler (KOI)

Le système ED-RAY AUTRA classifie les KOI en trois catégories :

### 1. CANDIDATE (Candidat)
- Signal de transit détecté
- Nécessite une confirmation supplémentaire
- Pourrait être une vraie exoplanète

### 2. CONFIRMED (Confirmé)
- Exoplanète vérifiée
- Confirmée par observations supplémentaires
- Haute confiance dans la détection

### 3. FALSE POSITIVE (Faux Positif)
- Signal causé par autre chose qu'une planète
- Peut être dû à :
  - Étoiles binaires à éclipses
  - Variations stellaires
  - Erreurs instrumentales

## La Zone Habitable

La **zone habitable** (ou "zone Goldilocks") est la région autour d'une étoile où l'eau liquide peut exister à la surface d'une planète.

### Critères
- Température permettant l'eau liquide (0-100°C)
- Dépend de la luminosité de l'étoile
- Ni trop chaud, ni trop froid

### Calcul Simplifié
Pour notre système solaire :
- Limite intérieure : ~0.95 UA
- Limite extérieure : ~1.37 UA
- La Terre est à 1.0 UA (parfait !)

## Utilisation d'ED-RAY AUTRA

### Prédiction d'une Exoplanète

1. **Collecte des Données**
   - Paramètres orbitaux (période, durée, profondeur)
   - Propriétés planétaires (rayon, température)
   - Caractéristiques stellaires

2. **Analyse par IA**
   - Le modèle analyse 45+ paramètres
   - Utilise un réseau de neurones entraîné sur 9000+ KOI
   - Précision de 92.79%

3. **Interprétation des Résultats**
   - Classe prédite (CANDIDATE, CONFIRMED, FALSE POSITIVE)
   - Niveau de confiance (probabilité)
   - Visualisation 3D interactive

### Exemple Pratique

Imaginons une détection avec ces paramètres :
- **koi_period** : 365 jours (similaire à la Terre)
- **koi_prad** : 1.2 R⊕ (légèrement plus grande que la Terre)
- **koi_teq** : 288 K (15°C, température terrestre)
- **koi_insol** : 1.0 (même flux que la Terre)

→ Cette planète serait probablement classée comme **CANDIDATE** ou **CONFIRMED** avec une haute probabilité d'être habitable !

## Exercices

### Exercice 1 : Identifier la Zone Habitable
Une étoile a une température de 4000 K (plus froide que le Soleil).
- Question : La zone habitable sera-t-elle plus proche ou plus éloignée que celle du Soleil ?
- Réponse : Plus proche, car l'étoile est moins lumineuse

### Exercice 2 : Calculer la Taille Relative
Une planète a un rayon de 2.5 R⊕.
- Question : Combien de fois est-elle plus grande que la Terre ?
- Réponse : 2.5 fois plus grande en rayon

### Exercice 3 : Interpréter un Transit
Un transit dure 4 heures avec une profondeur de 10000 ppm.
- Question : Que peut-on déduire sur la planète ?
- Réponse : Planète relativement grande (profondeur importante) avec une orbite probablement proche de l'étoile (transit court)

## Ressources Supplémentaires

### Sites Web
- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- [Kepler Mission](https://www.nasa.gov/mission_pages/kepler/main/index.html)
- [Exoplanet Exploration](https://exoplanets.nasa.gov/)

### Bases de Données
- **Cumulative KOI Table** : Données complètes de tous les KOI
- **Confirmed Planets** : Liste des exoplanètes confirmées
- **Stellar Parameters** : Caractéristiques des étoiles hôtes

### Outils
- **ED-RAY AUTRA** : Classification automatique par IA
- **Lightkurve** : Analyse des courbes de lumière Kepler
- **PyKE** : Outils Python pour les données Kepler

## Conclusion

La détection d'exoplanètes est un domaine fascinant qui combine :
- Astronomie observationnelle
- Analyse de données massives
- Intelligence artificielle
- Recherche de vie extraterrestre

Avec ED-RAY AUTRA, vous avez accès à un outil puissant pour explorer ces mondes lointains et contribuer à notre compréhension de l'univers !

---

**Prochaines Étapes** :
1. Explorez le module de prédiction
2. Créez votre propre dataset
3. Entraînez le modèle avec vos données
4. Partagez vos découvertes avec la communauté !

**Bon apprentissage ! 🌌🔭**
