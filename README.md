# my_torch

**my_torch** est un module de réseaux de neurones écrit en Python avec NumPy, sans bibliothèque de deep learning. Il sert ici à analyser des positions d'échecs : à partir d'une position en notation FEN, le réseau pré-entraîné dit s'il n'y a rien, un échec ou un échec et mat, et pour quelle couleur.

```
$> ./my_torch_analyzer --predict my_torch_network.nn boards.txt
Checkmate Black
Nothing
Nothing
Check White
Nothing
```

Le dépôt contient :

- **`my_torch`,** le module : couches denses, activations, initialisations, coûts, optimiseurs, entraînement par mini-batch, régularisation L2, early stopping, dropout et MC dropout, planification du learning rate, sauvegarde et recherche d'hyperparamètres ;
- **`my_torch_analyzer`,** qui prédit ou entraîne sur un fichier de positions ;
- **`my_torch_generator`,** qui crée des réseaux vierges à partir d'un fichier de configuration ;
- **`my_torch_network.nn`,** le réseau pré-entraîné, et tous les scripts qui permettent de le refaire.

## Les résultats

Le réseau poussé est entraîné sur le dataset fourni avec le sujet, 723 019 positions de vraies parties. Ses réglages ont été choisis sur une partie de validation, puis sa recette mesurée sur une partie de **test**, 71 554 positions jamais vues :

| Niveau | Précision |
|---|---|
| `Check` / `Nothing` | **95,7 %** |
| `Checkmate` / `Check` / `Nothing` | **90,0 %** |
| Avec les couleurs | **90,0 %** |

Le réseau poussé est ensuite entraîné avec cette recette sur toutes les positions du dataset : il fait 95,98 % sur l'ensemble de `dataset.zip`, et réussit les **10 tests sur 10** du correcteur. [Le document du réseau final](docs/reseau_final.md) explique comment il a été obtenu, et [les benchmarks](docs/benchmarks.md) comparent les choix possibles, graphiques à l'appui.

## Installation

Il faut **Python 3.12** et **NumPy** :

```
$> pip install -r requirements.txt
$> make
```

`make` crée les deux exécutables à la racine du dépôt. `make clean`, `make fclean` et `make re` font le ménage habituel.

Pour lancer les tests et tracer les graphiques, il faut aussi pytest et matplotlib :

```
$> pip install -r requirements-dev.txt
$> make tests_run
```

## `my_torch_analyzer`

```
./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE
```

| Commande | Effet |
|---|---|
| `./my_torch_analyzer --predict net.nn boards.txt` | Écrit l'état de chaque position, une ligne par position, dans l'ordre du fichier |
| `./my_torch_analyzer --train net.nn boards.txt` | Entraîne le réseau sur les positions labellisées, puis remplace `net.nn` |
| `./my_torch_analyzer --train --save new.nn net.nn boards.txt` | Entraîne le réseau, puis l'enregistre dans `new.nn` |

En entraînement, chaque epoch est affichée sur une ligne. Les réglages de l'entraînement sont ceux enregistrés dans le fichier `.nn`. Une erreur affiche un message et renvoie 84.

Détails : [docs/analyzer.md](docs/analyzer.md).

## `my_torch_generator`

```
$> ./my_torch_generator configs/basic_network.conf 3 && ls basic_network*
basic_network_1.nn  basic_network_2.nn  basic_network_3.nn
```

Chaque fichier de configuration est suivi du nombre de réseaux à créer. Les réseaux sont écrits dans le dossier courant, prêts à être entraînés.

Détails : [docs/generateur.md](docs/generateur.md).

## Les fichiers

**Le fichier d'échiquiers** contient une position par ligne, en notation FEN. Pour l'entraînement, chaque position est suivie de son label :

```
rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3 Checkmate Black
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 Nothing
```

Les labels possibles sont `Nothing`, `Check White`, `Check Black`, `Checkmate White` et `Checkmate Black`. Voir [docs/labels.md](docs/labels.md), et [docs/encodage.md](docs/encodage.md) pour la façon dont une position devient les 769 entrées du réseau.

**Le fichier de configuration** décrit un réseau et son entraînement, en TOML :

```toml
[network]
input_size = 769
layers = [
    { size = 256, activation = "relu" },
    { size = 64, activation = "relu", dropout = 0.1 },
    { size = 5, activation = "identity" },
]
initializer = "he"
seed = 0

[training]
loss = "cross_entropy"
epochs = 20
batch_size = 64
seed = 0
l2 = 0.0001
schedule = { name = "step", factor = 0.5, every = 10 }

[optimizer]
name = "adam"
learning_rate = 0.001
```

Chaque clé est vérifiée : une valeur invalide ou une clé inconnue donne une erreur claire. Voir [docs/configuration.md](docs/configuration.md), et [configs/my_torch_network.conf](configs/my_torch_network.conf) pour la configuration du réseau poussé.

**Le fichier `.nn`** contient le réseau, ses poids, son coût, son optimiseur avec sa mémoire et ses réglages d'entraînement : un entraînement peut reprendre là où il s'était arrêté. C'est une archive NumPy avec une description en JSON, sans `pickle`. Voir [docs/format_nn.md](docs/format_nn.md).

## Utiliser le module depuis le code

```python
import numpy as np
from my_torch import Network, get_loss, get_optimizer, load_network, save_network, train

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
targets = np.eye(2)[[0, 1, 1, 0]]  # XOR, en one-hot

network = Network(2, [(8, "tanh"), (2, "identity")], seed=0)
loss, optimizer = get_loss("cross_entropy"), get_optimizer("adam", learning_rate=0.05)
history = train(network, loss, optimizer, inputs, targets, epochs=200, batch_size=4, seed=0)

print(history["accuracy"][-1])   # 1.0
print(network.predict(inputs))   # [0 1 1 0]
save_network("xor.nn", network, loss, optimizer)
```

`train` accepte aussi un jeu de validation, la régularisation L2, l'early stopping et un planning du learning rate. Un réseau peut aussi être créé depuis une configuration, avec `load_config("net.conf").create_network()`.

| Sujet | Document |
|---|---|
| Les choix de conception et leurs résultats, en résumé | [docs/choix.md](docs/choix.md) |
| L'architecture du module | [docs/architecture.md](docs/architecture.md) |
| La backpropagation, expliquée et vérifiée | [docs/backpropagation.md](docs/backpropagation.md) |
| Les optimiseurs : SGD, Momentum, Nesterov, RMSProp, Adam | [docs/optimiseurs.md](docs/optimiseurs.md) |
| La boucle d'entraînement | [docs/entrainement.md](docs/entrainement.md) |
| La régularisation L2 et l'early stopping | [docs/regularisation.md](docs/regularisation.md) |
| Le dropout et le MC dropout | [docs/dropout.md](docs/dropout.md) |
| La planification du learning rate | [docs/learning_rate.md](docs/learning_rate.md) |
| La recherche d'hyperparamètres | [docs/recherche.md](docs/recherche.md) |
| Les métriques d'évaluation | [docs/evaluation.md](docs/evaluation.md) |
| Pourquoi Python et NumPy | [docs/choix_du_langage.md](docs/choix_du_langage.md) |

## Refaire le dataset et le réseau

**Le réseau poussé** se refait entièrement avec une commande, à partir de `dataset.zip`, l'archive fournie avec le sujet, placée à la racine du dépôt :

```
$> ./scripts/build_network.sh
```

Le script prépare le dataset, le découpe en entraînement, validation et test avec une graine fixe, crée le réseau avec `my_torch_generator`, l'entraîne avec `my_torch_analyzer --train`, puis l'évalue. Toutes les graines sont fixées : il redonne exactement le même réseau, en environ 1 h 30. Voir [docs/reseau_final.md](docs/reseau_final.md).

**Nos propres positions,** tirées de parties aléatoires et labellisées par nos règles des échecs, se génèrent avec :

```
$> python3 -m scripts.generate_dataset --size 1500 --seed 0 --output datasets/train.txt
```

Voir [docs/dataset.md](docs/dataset.md). Les datasets ne sont pas poussés, car ils sont trop lourds : `datasets/` et `dataset.zip` sont ignorés par git.

## Les benchmarks

[docs/benchmarks.md](docs/benchmarks.md) compare, sur le réseau d'échecs et dans les mêmes conditions, les architectures, les activations, les initialisations, les optimiseurs et la taille des batchs, avec les courbes d'apprentissage, les matrices de confusion et les temps d'entraînement et d'inférence. Les effets de [la régularisation L2](docs/regularisation.md), [du dropout](docs/dropout.md) et [du learning rate planifié](docs/learning_rate.md) sont mesurés sur le dataset complet, dans leurs propres documents.

## L'organisation du dépôt

| Dossier | Contenu |
|---|---|
| [my_torch/](my_torch) | Le module de réseaux de neurones, indépendant des échecs |
| [analyzer/](analyzer) | Les échecs : lecture des FEN, encodage, labels, prédiction, entraînement, évaluation |
| [dataset/](dataset) | Nos règles des échecs et la génération de positions labellisées |
| [bin/](bin) | Les points d'entrée des deux exécutables |
| [configs/](configs) | Les configurations : réseau poussé, candidates comparées, benchmarks, espace de recherche |
| [scripts/](scripts) | Préparation des données, entraînement, comparaisons, benchmarks et graphiques |
| [docs/](docs) | La documentation, avec ses graphiques dans [docs/images/](docs/images) |
| [tests/](tests) | Les tests, lancés par `make tests_run` |
