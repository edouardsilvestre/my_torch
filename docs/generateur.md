# my_torch_generator

`my_torch_generator` crée des réseaux vierges à partir de [fichiers de configuration](configuration.md), et les enregistre dans des [fichiers .nn](format_nn.md). Ces réseaux servent ensuite de point de départ à l'entraînement.

## Utilisation

```
$> make
$> ./my_torch_generator --help
USAGE
        ./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]

DESCRIPTION
        config_file_i    Configuration file containing description of a neural network we want to generate.
        nb_i             Number of neural networks to generate based on the configuration file.
$> ./my_torch_generator configs/basic_network.conf 3 && ls basic_network*
basic_network_1.nn  basic_network_2.nn  basic_network_3.nn
```

Chaque configuration est suivie du nombre de réseaux à créer. Plusieurs paires peuvent être données en une seule commande. En cas de succès, le programme n'affiche rien et renvoie 0.

## Les fichiers créés

**Leur nom** est celui de la configuration sans son extension, suivi du numéro du réseau, à partir de 1 : `basic_network.conf 3` donne `basic_network_1.nn`, `basic_network_2.nn` et `basic_network_3.nn`.

**Ils sont écrits dans le dossier courant,** comme dans l'exemple du sujet, même si la configuration est ailleurs : `./my_torch_generator configs/xor.conf 2` crée `xor_1.nn` et `xor_2.nn` là où la commande est lancée. Un fichier existant du même nom est remplacé.

**Chaque fichier contient** le réseau, le coût, un optimiseur neuf et les réglages d'entraînement de la configuration. Il peut donc être entraîné directement.

## Les graines

Les réseaux d'une même configuration doivent être différents : sinon, en générer plusieurs n'aurait aucun intérêt.

| La configuration... | Le réseau `i` reçoit... | Conséquence |
|---|---|---|
| fixe `network.seed = s` | la graine `s + i - 1` | Les réseaux sont différents, et la même commande redonne les mêmes fichiers. Le premier est le réseau que décrit la configuration elle-même |
| ne fixe pas de graine | des poids tirés au hasard | Les réseaux sont différents, et changent à chaque commande |

## Les erreurs

Le programme écrit un message sur la sortie d'erreur et renvoie 84 si :

| Cas | Exemple | Aide affichée |
|---|---|---|
| Aucun argument | `./my_torch_generator` | Oui |
| Un nombre impair d'arguments | `./my_torch_generator basic_network.conf` | Oui |
| Un nombre qui n'est pas un entier strictement positif | `0`, `-1`, `2.5`, `+3`, `abc` | Oui |
| Une configuration absente ou invalide | Voir [le fichier de configuration](configuration.md) | Non |
| Deux configurations qui donneraient les mêmes noms de fichiers | La même configuration deux fois, ou `a/net.conf` et `b/net.conf` | Non |
| Un fichier impossible à écrire | Un dossier courant en lecture seule | Non |

L'aide n'accompagne que les erreurs d'arguments : pour une configuration invalide, le message suffit à trouver la faute.

**Tout est vérifié avant d'écrire le premier fichier :** les arguments, toutes les configurations et les noms de fichiers. Une erreur dans la dernière configuration n'écrit donc aucun réseau de la première.

**Une limite :** si l'écriture échoue en cours de route, par exemple parce que le disque est plein, les réseaux déjà écrits restent en place. Chaque fichier, lui, est soit complet, soit absent, grâce à [l'écriture sûre](format_nn.md) des fichiers `.nn`.

## L'organisation du code

| Fichier | Rôle |
|---|---|
| [my_torch/generator.py](../my_torch/generator.py) | Lecture des arguments, génération et gestion des erreurs |
| [bin/my_torch_generator.py](../bin/my_torch_generator.py) | Lanceur : importe le module et renvoie 84 s'il est introuvable, par exemple si NumPy n'est pas installé |
| [Makefile](../Makefile) | Copie le lanceur à la racine sous le nom `my_torch_generator`, et le rend exécutable |

**L'exécutable doit rester à la racine du dépôt.** Il cherche le module `my_torch` dans son propre dossier : une copie placée ailleurs renvoie 84 avec le message `No module named 'my_torch'`. Il peut en revanche être lancé depuis n'importe quel dossier, en l'appelant par son chemin, et c'est dans ce dossier que les réseaux sont écrits.

Le générateur fait partie du module `my_torch`, comme le suggère le sujet : il ne dépend que des configurations et des fichiers `.nn`, et rien en lui n'est propre aux échecs.

## Comment c'est vérifié

[tests/test_generator.py](../tests/test_generator.py) vérifie l'exemple du sujet, le nommage, le dossier d'écriture, les graines et chaque erreur, en contrôlant qu'aucun fichier n'est écrit en cas d'erreur. [tests/test_launcher.py](../tests/test_launcher.py) lance le vrai exécutable dans un autre processus, pour vérifier les codes de sortie de bout en bout.

Pour vérifier les tests eux-mêmes, 28 erreurs ont été introduites volontairement dans le générateur, son lanceur et le Makefile, et chacune a fait échouer au moins un test.

Ces tests se lancent avec `make tests_run`.
