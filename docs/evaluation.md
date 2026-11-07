# L'évaluation d'un réseau

Ce document explique comment mesurer, sur notre machine, la qualité d'un réseau sur des positions labellisées, et comment lire le rapport obtenu.

## Évaluer un réseau

```
$> python3 -m scripts.evaluate --network my_torch_network.nn --chessboards datasets/all_test.txt
```

| Option | Rôle |
|---|---|
| `--network` | Le réseau à évaluer, un fichier `.nn` |
| `--chessboards` | Un fichier d'échiquiers où chaque ligne a son label, par exemple le jeu de test |
| `--json` | Écrit aussi toutes les métriques dans un fichier JSON, pour les graphiques des benchmarks |

Le jeu d'évaluation doit être le **jeu de validation** pour comparer des réglages, et le **jeu de test** seulement pour la mesure finale, comme l'explique [le découpage des données](dataset.md).

`--predict` n'affiche toujours que les prédictions : l'évaluation est un outil à part, pour ne rien ajouter à la sortie lue par le correcteur.

## Les métriques

Prenons 10 positions de 3 classes, avec ces réponses du réseau :

```
attendu : 0 0 0 0 1 1 1 2 2 2
prédit  : 0 0 1 2 1 1 0 2 2 2
```

**La matrice de confusion** compte les positions de chaque classe attendue (ligne) selon la classe prédite (colonne). La diagonale contient les bonnes réponses :

|  | Prédit 0 | Prédit 1 | Prédit 2 |
|---|---|---|---|
| **Attendu 0** | 2 | 1 | 1 |
| **Attendu 1** | 1 | 2 | 0 |
| **Attendu 2** | 0 | 0 | 3 |

| Métrique | Définition | Classe 0 de l'exemple |
|---|---|---|
| **Accuracy** | Bonnes réponses / total | 7 / 10 = 0,7 pour toutes les classes |
| **Précision** | Parmi les positions prédites dans la classe, la part qui l'est vraiment | 2 / 3 : la colonne 0 compte 3 positions, dont 2 justes |
| **Rappel** | Parmi les positions de la classe, la part trouvée par le réseau | 2 / 4 : la ligne 0 compte 4 positions, dont 2 trouvées |
| **F1** | 2 × précision × rappel / (précision + rappel) | 2 × (2/3) × (1/2) / (7/6) = 4/7 |
| **Effectif** | Nombre de positions attendues dans la classe | 4 |

**Pourquoi l'accuracy ne suffit pas.** Un réseau qui répondrait toujours `Nothing` aurait 33 % d'accuracy sur nos données, où un tiers des positions sont des `Nothing`, sans jamais détecter un échec. Sa précision et son rappel seraient nuls pour toutes les autres classes, et sa moyenne F1 tomberait à 0,17 sur 3 classes, et à 0,10 sur les 5 classes colorées.

**La moyenne macro** donne le même poids à chaque classe, quel que soit son effectif. On ne connaît pas la répartition des classes du correcteur : un réseau mauvais sur les mats doit se voir, même si les mats sont rares. Elle porte sur les classes attendues ou prédites au moins une fois. Une classe absente des deux côtés n'a aucun sens à évaluer, et ferait baisser la moyenne sans raison.

**Une précision non définie vaut 0.** Si le réseau ne prédit jamais une classe, sa précision demande de diviser par zéro. On la compte alors comme 0, la convention la plus courante.

Les métriques sont calculées dans [my_torch/metrics.py](../my_torch/metrics.py), car elles n'ont rien de propre aux échecs. L'évaluation d'un réseau d'échecs est dans [analyzer/evaluate.py](../analyzer/evaluate.py).

## Les trois niveaux du correcteur

Le correcteur note trois niveaux : `Check` / `Nothing`, puis `Checkmate` / `Check` / `Nothing`, puis la même chose avec les couleurs.

Le rapport donne les métriques avec les classes du réseau, puis à chaque niveau plus simple. Les prédictions et les labels y sont regroupés selon [la règle des labels](labels.md) : la couleur est retirée, puis un mat compte comme un échec. Un réseau à 5 sorties donne ainsi une estimation pour les **trois** niveaux, et un réseau à 3 sorties pour deux d'entre eux.

## Un exemple de rapport

Ce rapport mesure un petit réseau, avec une seule couche cachée de 64 neurones, entraîné 20 epochs sur 1 200 positions. Il se reproduit à l'identique avec ces commandes, où `small.conf` est une configuration avec ces réglages, Adam et des graines fixées :

```
$> python3 -m scripts.generate_dataset --size 1500 --seed 0 --output datasets/all.txt
$> python3 -m scripts.split_dataset --input datasets/all.txt --seed 0
$> ./my_torch_generator small.conf 1
$> ./my_torch_analyzer --train small_1.nn datasets/all_train.txt
$> python3 -m scripts.evaluate --network small_1.nn --chessboards datasets/all_test.txt
```

```
5 classes: accuracy 0.5133 (77 / 150)

    Class             Precision    Recall        F1   Support
    Nothing              0.5660    0.6000    0.5825        50
    Check White          0.3600    0.3600    0.3600        25
    Check Black          0.4545    0.4000    0.4255        25
    Checkmate White      0.4815    0.5200    0.5000        25
    Checkmate Black      0.6522    0.6000    0.6250        25
    Macro average        0.5028    0.4960    0.4986       150

    Confusion matrix (rows: expected, columns: predicted)
                               Nothing      Check White      Check Black  Checkmate White  Checkmate Black
    Nothing                         30                7                6                5                2
    Check White                      8                9                0                8                0
    Check Black                      7                1               10                1                6
    Checkmate White                  4                8                0               13                0
    Checkmate Black                  4                0                6                0               15
```

Le rapport continue avec les niveaux plus simples : **0,5200** d'accuracy sur 3 classes et **0,7133** sur 2 classes.

**Ce qu'on en lit :**

- **Les couleurs ne sont presque jamais confondues.** Sur les 100 échecs et mats, seuls 2 reçoivent la mauvaise couleur, deux `Check Black` prédits avec du blanc. C'est pourquoi l'accuracy change peu entre 5 et 3 classes.
- **La difficulté est de séparer un échec d'un mat.** 8 `Check White` sont prédits `Checkmate White`, et 8 `Checkmate White` sont prédits `Check White`.
- **Le réseau manque aussi des échecs.** 15 des 50 échecs sont prédits `Nothing`, ce qui limite l'accuracy sur 2 classes.

C'est une première mesure, avec très peu de données et un petit réseau. Elle sert de point de départ : l'entraînement du réseau final cherchera à faire beaucoup mieux.

## Comment c'est vérifié

- [tests/test_metrics.py](../tests/test_metrics.py) vérifie chaque métrique sur l'exemple calculé à la main ci-dessus, et les cas limites : réponses parfaites, réseau qui répond toujours la même classe, classes absentes, division par zéro.
- [tests/test_evaluate.py](../tests/test_evaluate.py) vérifie les niveaux, avec par exemple un `Check Black` prédit `Check White`, faux avec les couleurs mais juste sur 3 et 2 classes. Il vérifie aussi le rapport, le fichier JSON et les erreurs.

Pour vérifier les tests eux-mêmes, 19 erreurs ont été introduites volontairement dans ce code, et chacune a fait échouer au moins un test.

Ces tests se lancent avec `make tests_run`.
