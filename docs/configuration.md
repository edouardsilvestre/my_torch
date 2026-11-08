# Le fichier de configuration

Un fichier de configuration décrit un réseau et la façon de l'entraîner, sans écrire de Python. C'est à partir de lui que seront générés les réseaux vierges.

## Un exemple

Voici [configs/xor.conf](../configs/xor.conf), un petit réseau qui apprend XOR :

```toml
[network]
input_size = 2
layers = [
    { size = 4, activation = "tanh" },
    { size = 2, activation = "identity" },   # scores bruts : la cross-entropy applique softmax
]
initializer = "xavier_normal"
seed = 0

[training]
loss = "cross_entropy"
epochs = 2000
batch_size = 4
seed = 0

[optimizer]
name = "sgd"
learning_rate = 0.5
```

Le dépôt contient aussi [configs/basic_network.conf](../configs/basic_network.conf), un premier réseau pour les échiquiers à 3 sorties, et [configs/my_torch_network.conf](../configs/my_torch_network.conf), la configuration du réseau poussé, choisie parmi les candidates de [configs/candidates/](../configs/candidates) comme l'explique [le réseau pré-entraîné](reseau_final.md). Leur taille d'entrée, 769, est celle de [l'encodage de l'échiquier](encodage.md).

## Pourquoi TOML

Le fichier est écrit en [TOML](https://toml.io), et garde l'extension `.conf` utilisée par le sujet.

| Format | Commentaires | Types | Lecture |
|---|---|---|---|
| **TOML (choisi)** | Oui | Entiers, décimaux, booléens et textes distincts | `tomllib`, inclus dans Python |
| JSON | Non | Oui | `json`, inclus dans Python |
| INI | Oui | Non : tout est du texte | `configparser`, inclus dans Python |

TOML est le seul des trois qui accepte les commentaires et distingue les types. Le fichier peut donc expliquer ses réglages, et `epochs = "20"` est refusé au lieu d'être converti en silence.

`tomllib` fait partie de Python depuis la version 3.11 : c'est la version minimale nécessaire.

## Les clés

### `[network]`

| Clé | Obligatoire | Valeur | Par défaut |
|---|---|---|---|
| `input_size` | Oui | Entier strictement positif | - |
| `layers` | Oui | Liste non vide de couches `{ size = <entier>, activation = "<nom>" }` | - |
| `initializer` | Non | `uniform`, `normal`, `lecun`, `xavier_uniform`, `xavier_normal` ou `he` | `he` |
| `seed` | Non | Entier positif ou nul : graine des poids de départ | Aucune |

Les activations possibles sont `identity`, `sigmoid`, `tanh`, `relu`, `leaky_relu` et `softmax`.

### `[training]`

| Clé | Obligatoire | Valeur | Par défaut |
|---|---|---|---|
| `loss` | Oui | `mse` ou `cross_entropy` | - |
| `epochs` | Oui | Entier strictement positif | - |
| `batch_size` | Oui | Entier strictement positif | - |
| `shuffle` | Non | `true` ou `false` : mélange des données à chaque epoch | `true` |
| `seed` | Non | Entier positif ou nul : graine du mélange | Aucune |
| `l2` | Non | Nombre positif ou nul : force de la [régularisation L2](regularisation.md) | `0` |
| `patience` | Non | Entier strictement positif : epochs sans progrès avant [l'early stopping](regularisation.md) | Aucune |
| `monitor` | Non | `val_loss` ou `val_accuracy` : la mesure de validation que surveille l'early stopping | `val_loss` |
| `validation_split` | Non | Part des positions que `--train` met de côté pour la validation, de 0 inclus à 1 exclu | `0` |

Le nombre d'epochs et la taille de batch n'ont pas de valeur par défaut : une valeur choisie d'avance serait arbitraire, et elle dépend trop du problème.

### `[optimizer]`

| Clé | Optimiseurs | Par défaut |
|---|---|---|
| `name` | Obligatoire : `sgd`, `momentum`, `rmsprop` ou `adam` | - |
| `learning_rate` | Tous | `0.01` pour `sgd` et `momentum`, `0.001` pour les autres |
| `momentum`, `nesterov` | `momentum` | `0.9`, `false` |
| `decay` | `rmsprop` | `0.9` |
| `beta1`, `beta2` | `adam` | `0.9`, `0.999` |
| `epsilon` | `rmsprop`, `adam` | `1e-8` |

Le rôle et les limites de chaque paramètre sont expliqués dans [les optimiseurs](optimiseurs.md). La liste des paramètres acceptés est lue directement dans le code des optimiseurs : elle ne peut pas diverger de ce qu'ils acceptent vraiment.

## Les règles de vérification

**Toute clé inconnue est refusée.** Une faute de frappe comme `learning_rat` donne une erreur, au lieu d'être ignorée en laissant la valeur par défaut. C'est aussi le cas d'un paramètre qui appartient à un autre optimiseur, comme `beta1` pour `sgd`.

**Les types sont vérifiés.** Une taille doit être un entier, `2.5` et `true` sont refusés. Un nombre décimal peut s'écrire comme un entier (`learning_rate = 1`), mais il doit être fini : `nan` et `inf` sont refusés.

**Les valeurs sont vérifiées par le module lui-même.** Un learning rate négatif ou un momentum de `1` sont refusés avec le message de l'optimiseur.

**Avec la cross-entropy, la dernière couche doit utiliser `identity`.** La cross-entropy applique déjà softmax aux scores de la dernière couche, comme l'explique [la backpropagation](backpropagation.md). Une dernière couche `softmax` l'appliquerait deux fois, et toute autre activation fausserait les scores. Avec `mse`, la dernière activation est libre.

Chaque message donne le fichier, puis la clé fautive :

```
invalid configuration 'network.conf': optimizer: unknown key 'learning_rat'
invalid configuration 'network.conf': training.epochs must be a positive integer, got 0
invalid configuration 'network.conf': with the cross_entropy loss, the last layer must use the identity activation, got 'softmax': the loss already applies softmax to the scores
```

Une erreur de syntaxe TOML donne sa ligne et sa colonne. Toutes ces erreurs sont des `MyTorchError`, qui mènent au code de sortie 84.

## Utiliser une configuration

```python
from my_torch import load_config, save_network

config = load_config("configs/xor.conf")
network = config.create_network()
save_network(
    "xor.nn", network, config.create_loss(), config.create_optimizer(),
    config.training_settings(),
)
```

- **`create_network(seed=None)`** crée un réseau vierge. Sans argument, il utilise la graine de la configuration. Une autre graine permet de créer plusieurs réseaux différents à partir de la même configuration.
- **`create_optimizer()`** crée un nouvel optimiseur à chaque appel, car un optimiseur garde une mémoire propre à un réseau.
- **`training_settings()`** donne les huit réglages de `[training]` autres que le coût, avec leur valeur par défaut s'ils sont absents. Ce sont les réglages enregistrés dans le [fichier .nn](format_nn.md), avec le coût et l'optimiseur.

## Les deux graines

| Graine | Ce qu'elle fixe |
|---|---|
| `network.seed` | Les poids de départ |
| `training.seed` | L'ordre de mélange des données à chaque epoch |

Les séparer permet de faire varier l'une sans l'autre : par exemple, comparer plusieurs poids de départ avec le même ordre de mélange.

## Comment c'est vérifié

[tests/test_config.py](../tests/test_config.py) vérifie notamment :

- la lecture d'une configuration valide et les valeurs par défaut ;
- le refus de 41 configurations invalides, avec la clé fautive dans le message ;
- que le réseau construit est celui que décrit la configuration, et que deux graines donnent deux réseaux différents ;
- que les réglages enregistrés dans un fichier `.nn` sont relus et vérifiés ;
- que les deux exemples du dépôt donnent des réseaux valides, et que `xor.conf` apprend XOR.

Pour vérifier les tests eux-mêmes, 30 erreurs ont été introduites volontairement dans ce code, et chacune a fait échouer au moins un test.

Ces tests se lancent avec `make tests_run`.

## Ce qui viendra

La régularisation, le dropout et le learning rate variable ajouteront leurs propres clés au format. Comme toute clé inconnue est refusée, aucune de ces clés n'est encore acceptée.
