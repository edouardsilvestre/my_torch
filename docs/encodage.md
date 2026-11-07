# L'encodage de l'échiquier

Le réseau ne comprend que des vecteurs de nombres, alors que l'analyzer reçoit des échiquiers écrits en notation FEN. Ce document explique comment un FEN est lu, puis transformé en 769 nombres, et pourquoi.

## Le format FEN

Un FEN décrit une position en 6 champs séparés par des espaces :

```
rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1
```

| Champ | Exemple | Sens |
|---|---|---|
| Placement des pièces | `rnbqkbnr/.../RNBQKBNR` | Les 8 rangées, de la 8ᵉ à la 1ʳᵉ, séparées par `/`. Une lettre est une pièce, en majuscule pour les blancs (`PNBRQK`) et en minuscule pour les noirs. Un chiffre est un nombre de cases vides |
| Trait | `b` | Le camp qui doit jouer : `w` pour les blancs, `b` pour les noirs |
| Roques | `KQkq` | Les roques encore possibles, ou `-` |
| Prise en passant | `d3` | La case où un pion peut être pris en passant, ou `-` |
| Demi-coups | `0` | Le nombre de coups depuis la dernière prise ou le dernier mouvement de pion |
| Coups | `1` | Le numéro du coup |

## Ce qui est vérifié

[analyzer/fen.py](../analyzer/fen.py) vérifie **la syntaxe** de chaque champ :

- il y a exactement 6 champs ;
- il y a 8 rangées de 8 cases, sans caractère inconnu ni deux chiffres de suite ;
- le trait vaut `w` ou `b` ;
- les roques sont `-`, ou des lettres de `KQkq` sans répétition ;
- la prise en passant est `-`, ou une case de la 3ᵉ ou de la 6ᵉ rangée ;
- les deux compteurs sont des entiers positifs ou nuls.

Un FEN invalide lève une `AnalyzerError` qui cite le FEN et la raison, par exemple :

```
invalid FEN '8/8/8/8/8/8/8/8 w - - 0': expected 6 fields separated by spaces, got 5
```

**La légalité de la position n'est pas vérifiée.** Une position sans roi, avec deux rois blancs ou avec des pions sur la dernière rangée est acceptée, car le réseau n'en a pas besoin pour prédire. Une erreur donne le code 84 : refuser une position du correcteur coûterait tout le fichier, sans rien apporter.

Les 6 champs sont tous exigés, comme dans les exemples du sujet. C'est aussi ce qui permet de séparer sans ambiguïté le FEN du label qui le suit sur une ligne, même quand ce label contient une espace, comme `Check White`.

## L'encodage choisi

[analyzer/encoding.py](../analyzer/encoding.py) transforme une position en **769 nombres valant 0 ou 1** :

| Entrées | Contenu |
|---|---|
| 0 à 767 | 12 plans de 64 cases, un par type de pièce et par couleur, dans l'ordre `P N B R Q K p n b r q k` |
| 768 | Le trait : 1 si les blancs jouent, 0 si ce sont les noirs |

Dans chaque plan, une case vaut 1 si la pièce correspondante s'y trouve. Les cases suivent l'ordre de lecture du FEN : a8 d'abord, h1 en dernier. L'entrée d'une pièce est donc :

```
indice = numéro du plan × 64 + ligne × 8 + colonne       (ligne 0 = 8ᵉ rangée, colonne 0 = colonne a)
```

Par exemple, dans la position initiale, le roi blanc en e1 est l'entrée `5 × 64 + 7 × 8 + 4 = 380`. Cette position a 33 entrées à 1 : ses 32 pièces, et le trait des blancs.

## Pourquoi cet encodage

**Une entrée par pièce et par case.** L'autre idée simple serait une entrée par case, avec une valeur par pièce, par exemple 1 pour un pion et 9 pour une dame. Elle imposerait un ordre artificiel : le réseau devrait apprendre qu'une valeur de 5 n'est pas « entre » un fou et une dame, mais une tour. Avec un plan par pièce, chaque entrée a un sens simple, « telle pièce est sur telle case », que la première couche peut combiner directement.

**Le trait est indispensable.** Seul le camp qui doit jouer peut être en échec ou mat. La même disposition de pièces peut être un mat avec un trait, et une position sans rien de particulier avec l'autre.

**Pas de calcul d'échecs dans l'encodage.** On pourrait ajouter les cases attaquées par chaque camp. Mais savoir si le roi est sur une case attaquée, c'est déjà presque détecter l'échec : l'encodage ferait le travail à la place du réseau. Le sujet impose une solution par apprentissage supervisé, et c'est au réseau d'apprendre ces relations.

## Ce qui n'est pas encodé

| Information | Pourquoi |
|---|---|
| Les roques | Ils ne changent jamais l'état d'échec, et on ne peut pas roquer pour sortir d'un échec |
| La prise en passant | Elle peut, très rarement, être la seule parade à un mat. C'est **une limite connue** : elle pourra être ajoutée si les mesures montrent que ces positions comptent. [La génération du dataset](dataset.md) en donne un exemple concret |
| Les deux compteurs | Ils n'ont aucun effet sur l'échec et le mat |

Ces champs sont tout de même lus et vérifiés, pour refuser un FEN mal formé.

**Le plateau n'est pas retourné selon le trait.** On pourrait toujours présenter la position du point de vue du camp qui joue, en retournant le plateau quand les noirs jouent. Le réseau n'aurait alors qu'un seul cas à apprendre au lieu de deux. Mais cette idée suppose que seul le camp qui joue peut être en échec : sur une position où c'est l'autre camp, elle donnerait une mauvaise couleur. C'est une piste à mesurer dans les benchmarks.

## Le stockage en octets

Le vecteur est stocké en entiers sur 1 octet (`uint8`), et non en nombres décimaux sur 8 octets. Il ne contient que des 0 et des 1, et NumPy les convertit lors du calcul :

| Stockage | Mémoire pour un million de positions |
|---|---|
| `uint8` (choisi) | 769 Mo |
| `float64` | 6,2 Go |

Un test vérifie que l'entraînement est **strictement identique** avec les deux formats.

Sur notre machine, la lecture et l'encodage traitent quelques dizaines de milliers de positions par seconde : environ 77 000 FEN lus et 128 000 positions encodées.

## Comment c'est vérifié

- [tests/test_fen.py](../tests/test_fen.py) vérifie la position initiale, un plateau presque vide, les FEN du sujet, les positions illégales mais bien écrites, et 25 FEN invalides avec leur raison.
- [tests/test_encoding.py](../tests/test_encoding.py) vérifie des valeurs calculées à la main, l'ordre des cases et des plans, et que décoder le vecteur redonne chaque case de 300 échiquiers aléatoires.

Pour vérifier les tests eux-mêmes, 27 erreurs ont été introduites volontairement dans la lecture et l'encodage, et chacune a fait échouer au moins un test.

Ces tests se lancent avec `make tests_run`.
