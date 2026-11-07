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

**État actuel :** `--predict` fonctionne. `--train` n'est pas encore disponible : il renvoie 84 avec le message `--train is not available yet`, plutôt que de ne rien faire en renvoyant 0 comme si tout s'était bien passé.

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

Comme pour [my_torch_generator](generateur.md), l'exécutable doit rester à la racine du dépôt, où il trouve les paquets `analyzer` et `my_torch`.

## Comment c'est vérifié

- [tests/test_arguments.py](../tests/test_arguments.py) vérifie 8 lignes de commande valides, 22 invalides, et chaque problème de fichier.
- [tests/test_predict.py](../tests/test_predict.py) vérifie que la sortie est exactement la prédiction du réseau pour chaque échiquier, dans l'ordre, pour les trois jeux de classes, et chaque erreur.
- [tests/test_launcher.py](../tests/test_launcher.py) lance le vrai exécutable dans un autre processus, pour vérifier les codes de sortie de bout en bout. Il prédit l'exemple du sujet avec un réseau créé par `my_torch_generator`, et vérifie le comportement quand la sortie est fermée avant ou pendant l'écriture.

Pour vérifier les tests eux-mêmes, des erreurs ont été introduites volontairement dans ce code : 26 dans la ligne de commande et 14 dans le mode `--predict`. Chacune a fait échouer au moins un test.

Ces tests se lancent avec `make tests_run`.
