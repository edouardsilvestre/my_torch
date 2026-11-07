# La génération du dataset

Pour apprendre, le réseau a besoin de milliers de positions **avec leur bonne réponse**. Ce document explique comment le dépôt les produit, comment refaire exactement le même dataset, et comment le découper pour mesurer honnêtement la précision.

## Générer un dataset

Depuis la racine du dépôt :

```
$> python3 -m scripts.generate_dataset --size 1500 --seed 0 --output datasets/train.txt
1500 positions from 853 games in 122 s, written to datasets/train.txt
    Nothing          500
    Check White      250
    Check Black      250
    Checkmate White  250
    Checkmate Black  250
```

| Option | Rôle | Par défaut |
|---|---|---|
| `--size` | Nombre de positions | Obligatoire |
| `--seed` | Graine des parties aléatoires | `0` |
| `--output` | Fichier à écrire, ses dossiers sont créés au besoin | Obligatoire |
| `--shares` | Parts de `Nothing`, `Check` et `Checkmate`, par exemple `2:1:1` | `1:1:1` |

Le fichier a le format du [fichier d'échiquiers](labels.md), une position `FEN label` par ligne : `--train` le lit directement.

**Les datasets ne sont pas poussés,** car ils sont trop lourds : le dossier `datasets/` est ignoré par git. Le script, lui, est dans le dépôt, et **les mêmes options redonnent le même fichier, au bit près**. Par exemple, la commande ci-dessus produit un fichier dont l'empreinte SHA-256 commence par `c6f0de6d551ad502`, ce que l'on peut vérifier avec `sha256sum datasets/train.txt`.

**Durée :** environ 0,08 seconde par position sur notre machine, soit 40 minutes pour 30 000 positions. La génération des mats coûte le plus cher, car il faut une partie qui se termine par un mat pour chacun.

## Des règles des échecs écrites pour l'occasion

Labelliser une position demande de savoir si le roi du camp qui joue est attaqué, et, s'il l'est, si ce camp a encore un coup légal. Il faut donc connaître **tous les coups légaux**, y compris le roque, la prise en passant et la promotion.

Ces règles sont écrites dans le paquet [dataset/](../dataset), sans bibliothèque externe :

| Fichier | Rôle |
|---|---|
| [dataset/board.py](../dataset/board.py) | Plateau modifiable, cases attaquées, échec |
| [dataset/moves.py](../dataset/moves.py) | Coups légaux, et position obtenue après un coup |
| [dataset/labels.py](../dataset/labels.py) | Label d'une position |
| [dataset/generate.py](../dataset/generate.py) | Parties aléatoires et dataset équilibré |

**L'analyzer n'utilise jamais ces règles.** Elles servent uniquement à créer les exemples d'entraînement. Pour prédire, l'analyzer n'a que le réseau, comme l'exige le sujet : une solution par apprentissage supervisé. Les deux paquets sont séparés pour que ce soit évident.

### Vérifier les règles avec perft

Une erreur dans les règles donnerait des labels faux, et le réseau apprendrait ces erreurs. Pour la détecter, on utilise le **perft**, le test standard des générateurs de coups. Il compte toutes les suites de coups légaux d'une longueur donnée depuis une position, et compare aux totaux publiés sur le wiki chessprogramming.

| Position | Ce qu'elle met à l'épreuve | Plus grande profondeur vérifiée | Total |
|---|---|---|---|
| Position initiale | Coups ordinaires | 4 | 197 281 |
| « Kiwipete » | Roques, prises en passant, promotions, clouages | 3 | 97 862 |
| Position 3 | Prise en passant qui découvre le roi | 5 | 674 624 |
| Position 4 | Promotions et roques en position compliquée | 4 | 422 333 |
| Position 5 | Promotions avec prise | 3 | 62 379 |
| Position 6 | Milieu de partie | 3 | 89 890 |

Les 1,6 million de positions comptées correspondent exactement aux totaux publiés. Ces vérifications se refont avec `python3 -m scripts.perft`, en une trentaine de secondes. Les tests en reprennent une partie, plus rapide.

## Le label d'une position

[dataset/labels.py](../dataset/labels.py) suit les règles des échecs :

| Situation | Label |
|---|---|
| Le roi du camp qui joue n'est pas attaqué | `Nothing` |
| Il est attaqué, et ce camp a au moins un coup légal | `Check` + couleur |
| Il est attaqué, et ce camp n'a aucun coup légal | `Checkmate` + couleur |
| Le roi du camp qui **ne** joue **pas** est attaqué | Position écartée, car impossible dans une partie |

**La couleur est celle du camp qui donne l'échec ou le mat,** comme dans les exemples du sujet. Les labels ont toujours une couleur : un seul dataset peut ainsi entraîner les réseaux à 2, 3 ou 5 sorties, comme l'explique [le document sur les labels](labels.md).

**Le pat est labellisé `Nothing`** : le roi n'est pas attaqué. Le traiter à part est un bonus du sujet.

**Un exemple qui montre une limite de l'encodage.** Dans `8/3Q4/7R/k7/1Pp5/3N4/8/7K b - b3 0 1`, le roi noir est en échec, et son seul coup légal est de prendre le pion b4 en passant : c'est un `Check White`. Sans la case de prise en passant (`... b - - 0 1`), la même disposition est un `Checkmate White`. Or [l'encodage](encodage.md) ne contient pas la prise en passant : ces deux positions donnent le même vecteur au réseau, qui ne peut pas les distinguer. Ces positions sont très rares.

## Des positions tirées de parties aléatoires

Les positions viennent de parties jouées au hasard depuis la position initiale, sans téléchargement.

**Une préférence pour les échecs et les mats.** Joués purement au hasard, les mats seraient presque absents. À chaque coup :

- si un coup donne mat, il est joué 9 fois sur 10 ;
- sinon, si un coup donne échec, il est joué 3 fois sur 10 ;
- sinon, le coup est tiré au hasard parmi les coups légaux.

Une partie s'arrête au mat, au pat, ou après 300 demi-coups.

**Une seule position par label et par partie.** Les positions successives d'une même partie se ressemblent beaucoup. Chaque partie fournit donc au plus une position de chaque label, tirée au hasard parmi les siennes. Le dataset rassemble ainsi des centaines de parties différentes, au lieu de quelques-unes.

**Aucun doublon.** Une position déjà retenue n'est jamais reprise, même dans une autre partie. Les compteurs de coups, qui ne changent rien à la position, sont ignorés pour cette comparaison.

**Des classes équilibrées.** Par défaut, il y a un tiers de `Nothing`, un tiers d'échecs et un tiers de mats. Les échecs et les mats sont partagés également entre blanc et noir. Quand une part ne tombe pas juste, les positions restantes vont aux labels dont la part arrondie a perdu le plus : le total est toujours exactement `--size`. Les lignes sont ensuite mélangées.

Sur le dataset de 1 500 positions ci-dessus, le nombre moyen de pièces est presque le même dans chaque classe, entre 18,6 et 21. Le réseau ne peut donc pas deviner un mat en comptant les pièces.

## Découper le dataset

Pour savoir si le réseau a vraiment appris, il faut le tester sur des positions **qu'il n'a jamais vues**. Le dataset est donc découpé en trois parties :

| Partie | Part par défaut | Rôle |
|---|---|---|
| Entraînement | 80 % | Le réseau apprend dessus |
| Validation | 10 % | Choisir les réglages : architecture, learning rate, nombre d'epochs |
| Test | 10 % | Mesurer la précision finale, une seule fois, à la fin |

```
$> python3 -m scripts.split_dataset --input datasets/all.txt --ratios 80:10:10 --seed 0
datasets/all_train.txt: 1200 positions (Nothing 400, Check White 200, Check Black 200, Checkmate White 200, Checkmate Black 200)
datasets/all_validation.txt: 150 positions (Nothing 50, Check White 25, Check Black 25, Checkmate White 25, Checkmate Black 25)
datasets/all_test.txt: 150 positions (Nothing 50, Check White 25, Check Black 25, Checkmate White 25, Checkmate Black 25)
```

Les trois fichiers sont écrits à côté du dataset d'origine, au même format : `--train` et `--predict` les lisent directement.

**Le jeu de test ne sert jamais à choisir.** Si l'on choisissait les réglages d'après le jeu de test, on finirait par garder ceux qui marchent par chance sur ces positions-là. Il deviendrait indirectement une donnée d'apprentissage, et la précision mesurée dessus serait trop optimiste. Les choix se font donc sur le jeu de validation, et le test ne sert qu'à la mesure finale. Aucun programme ne peut imposer cette règle, puisqu'il ne sait pas à quoi sert un fichier : elle est écrite ici, et les outils de recherche de réglages utiliseront le jeu de validation.

[dataset/split.py](../dataset/split.py) découpe le dataset ainsi :

- **Label par label.** Chaque label, couleur comprise, est découpé séparément selon les mêmes parts. Les trois parties ont donc les mêmes proportions de chaque label. Une part qui ne tombe pas juste est arrondie par les plus forts restes, comme pour la génération. À égalité de reste, les positions vont aux premières parties : l'écart est d'au plus une position par label. Sur un très petit dataset, un label de 1 ou 2 positions part entièrement dans l'entraînement.
- **Au hasard, avec une graine.** Les positions de chaque label sont mélangées avant d'être découpées. Sinon, la validation et le test contiendraient toujours les dernières lignes du fichier, ce qui serait un biais si le fichier était trié. La même graine redonne les mêmes trois fichiers, au bit près.
- **Sans doublon.** Une position présente deux fois, compteurs de coups mis à part, n'est gardée qu'une fois, et le nombre de doublons retirés est affiché. Sans cela, une même position pourrait être à la fois dans l'entraînement et dans le test : le test mesurerait alors de la mémoire, pas de l'apprentissage. Notre générateur n'en produit pas, mais un dataset venu d'ailleurs pourrait en contenir.

Chaque ligne doit avoir un label valide. Sinon, le script s'arrête en citant le numéro de la ligne, sans écrire de fichier.

## Les limites

- **Des parties aléatoires, pas de vraies parties.** Les positions comportent des coups que personne ne jouerait. Celles du correcteur ressemblent plutôt à de vraies parties, à en juger par l'exemple du sujet. Si la précision en souffre, on pourra ajouter des positions de parties réelles.
- **Une répartition des classes choisie, pas connue.** On ne sait pas quelle proportion de mats contient le fichier du correcteur. `--shares` permet d'en essayer d'autres.
- **La vitesse.** La génération tourne en Python, sur un seul cœur. Pour de très grands datasets, on pourrait la répartir sur plusieurs processus, chacun avec sa propre graine.

## Comment c'est vérifié

- [tests/test_perft.py](../tests/test_perft.py) vérifie les totaux perft de référence, le roque, la prise en passant, la promotion et les clouages.
- [tests/test_dataset.py](../tests/test_dataset.py) vérifie les labels de positions connues, l'équilibre des classes, l'absence de doublons, la reproductibilité au bit près, et que chaque ligne générée a bien le label donné par les règles. Des parties préparées à l'avance garantissent que les cas rares, comme une position revue ou illégale, sont bien testés.
- [tests/test_split.py](../tests/test_split.py) vérifie les proportions de chaque label dans les trois parties, qu'elles sont disjointes et contiennent toutes les lignes, que la graine choisit quelles positions vont dans chaque partie, et le retrait des doublons.

Pour vérifier les tests eux-mêmes, 50 erreurs ont été introduites volontairement dans les règles, la labellisation, la génération et le découpage, et chacune a fait échouer au moins un test.

Ces tests se lancent avec `make tests_run`.
