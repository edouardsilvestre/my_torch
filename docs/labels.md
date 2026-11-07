# Le fichier d'échiquiers et les labels

Ce document décrit le fichier que reçoit l'analyzer, les labels qu'il peut contenir, et la façon dont ces labels deviennent les sorties du réseau.

## Le fichier d'échiquiers

Chaque ligne contient un échiquier en [notation FEN](encodage.md), suivi ou non du résultat attendu :

```
rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3 Checkmate Black
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 Nothing
rnbqkbnr/pppp2pp/8/4pp1Q/3P4/4P3/PPP2PPP/RNB1KBNR b KQkq - 1 3 Check White
```

Les 6 premiers champs forment le FEN, et **tout le reste de la ligne** forme le label. C'est ce qui permet de lire un label qui contient une espace, comme `Check White`.

| Mode | Label |
|---|---|
| Entraînement | Obligatoire sur chaque ligne |
| Prédiction | Facultatif, ligne par ligne |

[analyzer/chessfile.py](../analyzer/chessfile.py) lit les échiquiers dans l'ordre du fichier, et garde le numéro de ligne de chacun. Il tolère :

- les lignes vides, qui sont ignorées et ne donnent pas de prédiction ;
- les espaces en trop, au début, à la fin ou entre les champs ;
- les fins de ligne Windows (`\r\n`) et l'en-tête BOM qu'ajoutent certains éditeurs ;
- une dernière ligne sans retour à la ligne.

Une erreur cite le fichier et le numéro de ligne, puis la raison :

```
chessboards.txt:2: unknown label 'Chek', expected one of: Nothing, Check, Checkmate, Check White, Check Black, Checkmate White, Checkmate Black
chessboards.txt:5: invalid FEN '8/8/8/8/8/8/8/k1K5 w - -': expected 6 fields separated by spaces, got 4
```

Un fichier absent, illisible ou qui n'est pas du texte lève aussi une erreur. Toutes ces erreurs mènent au code de sortie 84.

## Les labels

Les labels sont ceux du sujet, majuscules comprises :

| Sans couleur | Avec couleur |
|---|---|
| `Nothing` | |
| `Check` | `Check White`, `Check Black` |
| `Checkmate` | `Checkmate White`, `Checkmate Black` |

`Nothing` n'a pas de couleur. Le pat, prévu en bonus, n'est pas encore géré.

**La couleur désigne le camp qui donne l'échec ou le mat.** On le déduit des exemples du sujet :

- **Le premier échiquier** est le « mat du fou », le mat le plus rapide possible : la dame noire en h4 mate le roi blanc, et ce sont les blancs qui doivent jouer. Le réseau complet du sujet répond `Checkmate Black`.
- **Le quatrième échiquier** est un échec de la dame blanche en h5 au roi noir. Ce sont les noirs qui doivent jouer, et la réponse est `Check White`.

Dans une position légale, seul le camp qui doit jouer peut être en échec : la couleur du label est donc en général l'opposé du trait.

## Les classes du réseau

Le correcteur évalue trois niveaux de précision. Chacun correspond à un jeu de classes, que [analyzer/labels.py](../analyzer/labels.py) reconnaît au **nombre de sorties** du réseau :

| Sorties | Classes, dans l'ordre des sorties |
|---|---|
| 2 | `Nothing`, `Check` |
| 3 | `Nothing`, `Check`, `Checkmate` |
| 5 | `Nothing`, `Check White`, `Check Black`, `Checkmate White`, `Checkmate Black` |

Le fichier `.nn` n'a donc pas besoin de stocker les classes : la taille de la dernière couche suffit pour savoir comment afficher une prédiction. Un réseau avec un autre nombre de sorties est refusé.

## Utiliser un label plus précis que les classes

Un label peut toujours servir à entraîner un réseau qui distingue moins de choses :

| Label | 2 classes | 3 classes | 5 classes |
|---|---|---|---|
| `Nothing` | `Nothing` | `Nothing` | `Nothing` |
| `Check` | `Check` | `Check` | Refusé |
| `Checkmate` | `Check` | `Checkmate` | Refusé |
| `Check White` | `Check` | `Check` | `Check White` |
| `Checkmate Black` | `Check` | `Checkmate` | `Checkmate Black` |

- **La couleur est retirée** quand les classes n'en ont pas.
- **Un mat compte comme un échec** quand les classes n'ont pas de mat, puisque le roi d'un camp mat est en échec. C'est **notre interprétation** : le sujet ne dit pas ce que le correcteur attend pour un mat dans l'évaluation « Check/Nothing ».
- **Un label sans couleur est refusé pour 5 classes,** avec une erreur, car il ne dit pas quel camp donne l'échec.

Ainsi, un seul dataset dont les labels ont une couleur peut entraîner les trois réseaux.

## Les cibles et les prédictions

**Pour entraîner,** chaque label devient une cible one-hot : une ligne avec un 1 dans la colonne de sa classe, et des 0 ailleurs. Avec 5 classes, `Checkmate Black` devient `[0, 0, 0, 0, 1]`. Les cibles sont des nombres décimaux : elles n'ont que 2 à 5 colonnes, donc les stocker en octets, comme les entrées, ne ferait presque rien gagner.

**Pour prédire,** le réseau donne un score par classe, et l'analyzer affiche le label de la classe qui a le score le plus haut.

## Comment c'est vérifié

- [tests/test_labels.py](../tests/test_labels.py) vérifie chaque case du tableau ci-dessus, les cibles calculées à la main, et qu'encoder puis décoder redonne les labels.
- [tests/test_chessfile.py](../tests/test_chessfile.py) lit le fichier du sujet avec et sans labels, et vérifie chaque tolérance et chaque erreur, avec son numéro de ligne.

Pour vérifier les tests eux-mêmes, 23 erreurs ont été introduites volontairement dans ce code, et chacune a fait échouer au moins un test.

Ces tests se lancent avec `make tests_run`.
