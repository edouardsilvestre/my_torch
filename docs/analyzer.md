# my_torch_analyzer

`my_torch_analyzer` charge un réseau depuis un [fichier .nn](format_nn.md), puis l'utilise pour analyser un [fichier d'échiquiers](labels.md), ou l'entraîne sur ce fichier.

## Utilisation

```
./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE
```

| Commande | Effet |
|---|---|
| `./my_torch_analyzer --predict net.nn boards.txt` | Prédit l'état de chaque échiquier |
| `./my_torch_analyzer --train net.nn boards.txt` | Entraîne le réseau, puis remplace `net.nn` |
| `./my_torch_analyzer --train --save new.nn net.nn boards.txt` | Entraîne le réseau, puis l'enregistre dans `new.nn` |
| `./my_torch_analyzer --help` | Affiche l'aide du sujet et renvoie 0 |

## Les règles

- Il faut exactement un mode : `--predict` ou `--train`.
- `--save` ne fonctionne qu'avec `--train`, et doit être suivi d'un nom de fichier.
- Il faut exactement deux fichiers : `LOADFILE` puis `CHESSFILE`.
- Une option répétée ou inconnue est une erreur. Les abréviations comme `--pred` ne sont pas acceptées.
- Les options peuvent être placées avant ou après les fichiers. Seul l'ordre des deux fichiers compte.
- `--help` doit être utilisé seul.

La ligne de commande est lue par [analyzer/arguments.py](../analyzer/arguments.py), écrit à la main. Le module standard `argparse` aurait renvoyé le code 2 en cas d'erreur au lieu de 84, accepté des abréviations, et écrit ses propres messages.

## Les erreurs

Le programme écrit un message sur la sortie d'erreur et renvoie 84.

**Une erreur dans la ligne de commande** est suivie de l'aide, pour montrer la bonne utilisation :

```
my_torch_analyzer: --save only works with --train

USAGE
        ./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE
...
```

**Un problème de fichier** n'affiche que le message, qui suffit à trouver la faute :

| Cas | Message |
|---|---|
| Fichier à lire absent | `cannot read LOADFILE 'net.nn': no such file` |
| Dossier à la place d'un fichier | `cannot read CHESSFILE 'boards': is a directory` |
| Fichier illisible | `cannot read CHESSFILE 'boards.txt': permission denied` |
| Dossier de destination absent | `cannot save the network to 'out/new.nn': no such directory` |
| Dossier de destination en lecture seule | `cannot save the network to 'new.nn': permission denied` |
| Réseau qui n'attend pas 769 entrées | `network expects 768 inputs, but chessboards are encoded with 769` |
| Réseau qui n'a pas 2, 3 ou 5 sorties | `a network with 4 outputs cannot predict chess labels: expected 2, 3 or 5 outputs` |
| Ligne invalide dans le fichier d'échiquiers | `chessboards.txt:6: invalid FEN '8/8/8 w - - 0 1': ...` |
| Réseau sans de quoi l'entraîner | `network file 'net.nn' has no loss, optimizer, training settings: it cannot be trained; ...` |
| Label incompatible avec les classes du réseau | `train.txt:3: label 'Check' does not tell which of these classes it is: ...` |
| Fichier d'entraînement sans échiquier | `no labelled chessboard in 'train.txt'` |
| Ctrl+C | `interrupted, nothing was saved` |
| Sortie fermée pendant l'entraînement | `the standard output was closed during training, nothing was saved` |

## Le mode --predict

```
$> ./my_torch_analyzer --predict my_torch_network.nn chessboards.txt
Checkmate Black
Nothing
Nothing
Check White
Nothing
```

Le programme suit ces étapes, dans [analyzer/predict.py](../analyzer/predict.py) :

1. **Il charge le réseau** et vérifie qu'il convient aux échiquiers : 769 entrées, comme [l'encodage](encodage.md), et 2, 3 ou 5 sorties, qui donnent [les classes](labels.md) à afficher.
2. **Il lit tout le fichier d'échiquiers.** Les lignes vides sont ignorées, et les labels attendus sont vérifiés, puis ignorés.
3. **Il encode les échiquiers et les fait passer dans le réseau,** par paquets de 1 024. Un fichier d'un million de positions ne demande ainsi pas des gigaoctets de mémoire pour les calculs du réseau.
4. **Il écrit toutes les prédictions d'un coup,** une par ligne, dans l'ordre du fichier.

**La sortie standard ne contient que les prédictions,** et la sortie d'erreur reste vide en cas de succès. La précision n'est pas affichée quand le fichier contient les labels attendus : on ne sait pas comment le correcteur traite la sortie d'erreur, et les métriques auront leur propre outil.

**Une ligne invalide fait échouer tout le fichier,** avec son numéro de ligne et le code 84, comme le demande le sujet pour toute erreur. Comme tout le fichier est lu avant la première prédiction, rien n'est alors écrit sur la sortie standard : pas de résultats à moitié écrits.

**Un fichier sans échiquier** n'affiche rien et renvoie 0.

**Vitesse :** sur notre machine, environ 28 000 échiquiers par seconde avec un réseau `769 → 256 → 64 → 3`, lecture du fichier et chargement compris, soit 3,5 secondes pour 100 000 échiquiers.

### Une sortie fermée en cours de route

L'exemple du sujet lit les premières prédictions avec `| head -n 5`. `head` ferme alors la sortie après 5 lignes, et l'écriture suivante échoue avec l'erreur `BrokenPipeError`. Ce n'est pas une erreur du programme, c'est le lecteur qui a choisi de ne pas tout lire : le programme s'arrête sans message et renvoie 0.

Pour cela, la sortie est vidée explicitement à la fin de la prédiction. Sans ce `flush()`, une petite sortie resterait en mémoire tampon, et son écriture n'aurait lieu qu'à la fermeture du programme, en dehors de la gestion d'erreur : Python afficherait alors un message d'erreur.

## Le mode --train

```
$> ./my_torch_analyzer --train --save trained.nn my_torch_network.nn train.txt
epoch   1/100 - loss 1.2705 - accuracy 0.6000
epoch   2/100 - loss 1.0602 - accuracy 0.6000
...
epoch 100/100 - loss 0.0005 - accuracy 1.0000
```

Le programme suit ces étapes, dans [analyzer/train.py](../analyzer/train.py) :

1. **Il charge le réseau avec ce qu'il faut pour l'entraîner :** le coût, l'optimiseur avec sa mémoire, et les réglages d'entraînement. Tout fichier créé par [my_torch_generator](generateur.md) les contient. Sinon, le programme s'arrête avec une erreur : il n'invente pas de valeurs par défaut, qui seraient arbitraires.
2. **Il lit le fichier d'échiquiers,** où chaque ligne doit avoir son label. Chaque label est adapté aux classes du réseau, comme l'explique [le document sur les labels](labels.md) : sans couleur pour 3 sorties, et un mat compte comme un échec pour 2 sorties. **Tous les labels sont vérifiés avant de commencer** : un label incompatible donne une erreur avec son numéro de ligne, pas après des heures de calcul.
3. **Il entraîne le réseau** avec les réglages du fichier : nombre d'epochs, taille de batch, mélange et graine. Toutes les lignes servent à l'entraînement, sauf si le réglage `validation_split` en met une part de côté pour la validation, ce que demande [l'early stopping](regularisation.md). La progression s'affiche sur la sortie standard, une ligne par epoch, ce que le sujet permet en mode entraînement.
4. **Il sauvegarde le réseau à la fin,** dans `SAVEFILE`, ou à la place de `LOADFILE` sans `--save`. L'écriture est sûre : le fichier est d'abord écrit sous un nom temporaire, puis renommé.

**Reprendre un entraînement.** Le fichier sauvegardé contient la mémoire de l'optimiseur : relancer `--train` sur ce fichier poursuit l'entraînement, avec le nombre d'epochs des réglages à chaque lancement. L'ordre de mélange, lui, recommence au début, comme l'explique [le format .nn](format_nn.md).

**Ctrl+C arrête l'entraînement sans rien sauvegarder.** Le programme affiche `interrupted, nothing was saved` et renvoie 84. Comme la sauvegarde n'a lieu qu'à la fin, et de façon sûre, le fichier de destination reste exactement tel qu'il était. Sauvegarder après chaque epoch permettrait de ne perdre qu'une epoch, mais n'est pas fait pour l'instant.

**Une sortie fermée pendant l'entraînement est une erreur.** Avec `--train ... | head -n 1`, l'affichage de la progression échoue dès que `head` ferme la sortie, ce qui interrompt l'entraînement avant la sauvegarde. Contrairement à `--predict`, où le lecteur a simplement choisi de ne pas tout lire, le travail demandé n'est pas fait : le programme le dit et renvoie 84, au lieu d'un faux succès.

## Vérifier les fichiers avant de commencer

`LOADFILE` et `CHESSFILE` sont vérifiés dès le départ : ils doivent exister et être lisibles.

En entraînement, la destination l'est aussi : `SAVEFILE`, ou `LOADFILE` sans `--save`. Son dossier doit exister et être accessible en écriture, car le [fichier .nn](format_nn.md) y est d'abord écrit sous un nom temporaire. Sans cette vérification, une destination impossible ne serait découverte qu'au moment de sauvegarder, **après** un entraînement qui peut durer des heures.

En prédiction, rien n'est écrit : un `LOADFILE` dans un dossier en lecture seule est accepté.

## L'organisation du code

| Fichier | Rôle |
|---|---|
| [bin/my_torch_analyzer.py](../bin/my_torch_analyzer.py) | Lanceur, copié à la racine par le Makefile. Il renvoie 84 si un module est introuvable |
| [analyzer/main.py](../analyzer/main.py) | Point d'entrée : lit la ligne de commande et transforme chaque erreur en message et en code 84 |
| [analyzer/arguments.py](../analyzer/arguments.py) | Lecture de la ligne de commande et vérification des fichiers |
| [analyzer/predict.py](../analyzer/predict.py) | Mode `--predict` |
| [analyzer/train.py](../analyzer/train.py) | Mode `--train` |

Comme pour [my_torch_generator](generateur.md), l'exécutable doit rester à la racine du dépôt, où il trouve les paquets `analyzer` et `my_torch`.

## Comment c'est vérifié

- [tests/test_arguments.py](../tests/test_arguments.py) vérifie 8 lignes de commande valides, 22 invalides, et chaque problème de fichier.
- [tests/test_predict.py](../tests/test_predict.py) vérifie que la sortie est exactement la prédiction du réseau pour chaque échiquier, dans l'ordre, pour les trois jeux de classes, et chaque erreur.
- [tests/test_train.py](../tests/test_train.py) vérifie qu'un réseau entraîné sur les 5 positions du sujet prédit ensuite exactement leurs labels, avec 2, 3 ou 5 sorties, ainsi que `--save`, la reprise, et que rien n'est modifié en cas d'erreur ou d'interruption.
- [tests/test_launcher.py](../tests/test_launcher.py) lance le vrai exécutable dans un autre processus, pour vérifier les codes de sortie de bout en bout. Il prédit l'exemple du sujet avec un réseau créé par `my_torch_generator`, enchaîne génération, entraînement et prédiction, interrompt un vrai entraînement par Ctrl+C, et vérifie le comportement quand la sortie est fermée.

Pour vérifier les tests eux-mêmes, des erreurs ont été introduites volontairement dans ce code : 26 dans la ligne de commande, 14 dans le mode `--predict` et 19 dans le mode `--train`. Chacune a fait échouer au moins un test.

Ces tests se lancent avec `make tests_run`.
