# Spotify Popularity Lab

**Les caractéristiques musicales permettent-elles de distinguer les niveaux de faible popularité ?**

Projet d'analyse de données sur **4 073 titres Spotify**, avec Python, pandas et scikit-learn : exploration, classification supervisée, déséquilibre des classes et regroupement par k-means.

[README principal et installation](README.md) · [Méthode et limites](docs/methodology.md) · [Données](data/README.md)

![Répartition des classes](reports/figures/class_distribution.png)

## Résultats à retenir

La popularité prend dix valeurs, de 0 à 9. La classe 0 représente **63,25 %** des titres ; la classe 9 ne contient que **quatre chansons différentes**.

Le meilleur candidat parmi les arbres et k-NN testés, selon l'accuracy moyenne de validation croisée, est **k-NN avec deux voisins** : **48,68 %**. Sur le test réservé, il atteint **51,15 %**, contre **63,26 %** pour une prédiction systématique de la classe majoritaire. Son accuracy équilibrée vaut **11,92 %** et son F1 macro **0,120** sur ce test : la reconnaissance des classes rares reste faible.

Ces résultats montrent les limites du modèle et des données. Le projet ne prétend pas prédire les futurs succès musicaux.

## Démarche

- Vérifier la qualité des données et étudier les distributions.
- Séparer les données d'apprentissage, de validation et de test.
- Calculer les paramètres de standardisation sur l'apprentissage uniquement.
- Sur-échantillonner les classes rares à l'intérieur de chaque pli, sans modifier la validation.
- Comparer les arbres de profondeur 1 à 5 et k-NN de 2 à 20 voisins.
- Sélectionner les attributs dans chaque pli, puis évaluer le candidat retenu sur le test réservé.
- Former dix clusters sans fournir la popularité, sur dix découpages différents.

La silhouette moyenne du clustering vaut **0,141** : les groupes ne sont pas nettement séparés. Rand vaut **0,531** ; il mesure un accord sur des paires et ne signifie pas 53,1 % de popularités correctement prédites.

## Exécution

Après installation et téléchargement du CSV selon le README principal :

```bash
spotify-analysis --data data/raw/unpopular_songs.csv --output reports
python -m unittest discover -s tests -v
```

Les graphiques et tableaux du dépôt proviennent d'une exécution réelle. Les huit tests portent notamment sur les fuites de données, les index et le rééquilibrage. Le workflow GitHub Actions est fourni pour les prochaines publications.

Le CSV brut n'est pas publié : le dépôt donne sa source et l'empreinte du fichier utilisé. Les informations d'artiste, de titre et d'identifiant ne sont pas utilisées comme attributs.

Auteur : **Mohand Ouidir HOCINE** — [IDIR2626](https://github.com/IDIR2626). Source des données : **Etienne GCC**, [Spotify Unpopular Songs sur Kaggle](https://www.kaggle.com/datasets/estienneggx/spotify-unpopular-songs/data). Licence du code et de la documentation : MIT.
