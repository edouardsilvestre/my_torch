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

**État actuel :** la ligne de commande et les fichiers sont vérifiés, mais les deux modes ne sont pas encore disponibles. Une commande valide renvoie 84 avec le message `--predict is not available yet`, ou `--train`, plutôt que de ne rien afficher en renvoyant 0 comme si tout s'était bien passé.

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

Comme pour [my_torch_generator](generateur.md), l'exécutable doit rester à la racine du dépôt, où il trouve les paquets `analyzer` et `my_torch`.

## Comment c'est vérifié

- [tests/test_arguments.py](../tests/test_arguments.py) vérifie 8 lignes de commande valides, 22 invalides, et chaque problème de fichier.
- [tests/test_launcher.py](../tests/test_launcher.py) lance le vrai exécutable dans un autre processus, pour vérifier les codes de sortie de bout en bout.

Pour vérifier les tests eux-mêmes, 26 erreurs ont été introduites volontairement dans ce code, et chacune a fait échouer au moins un test.

Ces tests se lancent avec `make tests_run`.
