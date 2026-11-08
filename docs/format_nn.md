# Le format .nn

Ce document décrit ce que contient un fichier `.nn`, pourquoi ce format a été choisi, et comment un fichier est vérifié avant d'être utilisé.

## Utilisation

```python
from my_torch import load_network, save_network

save_network("my_torch_network.nn", network, loss, optimizer, settings)

saved = load_network("my_torch_network.nn")
saved.network.predict(inputs)
```

Seul le réseau est obligatoire. Le coût, l'optimiseur et les réglages servent à reprendre un entraînement : `load_network` renvoie `None` pour chacun d'eux s'il n'a pas été sauvegardé.

## Ce que contient un fichier

Un fichier `.nn` est une **archive NumPy** (`.npz`) : un fichier zip qui contient des tableaux, chacun au format `.npy`.

| Entrée | Contenu |
|---|---|
| `metadata` | La description du réseau, en texte JSON |
| `layers/<i>/weights`, `layers/<i>/biases` | Les poids et les biais de la couche `i` |
| `optimizer/<mémoire>/<i>/<paramètre>` | La mémoire de l'optimiseur pour chaque paramètre : `velocities` pour le momentum, `squares` pour RMSProp, `means` et `squares` pour Adam |
| `optimizer/steps` | Le nombre de pas déjà faits par Adam |

Voici la description d'un réseau `768 → 256 → 64 → 3` entraîné avec Adam :

```json
{
    "format": "my_torch",
    "version": 1,
    "input_size": 768,
    "architecture": [[256, "relu"], [64, "relu"], [3, "identity"]],
    "dropout": [0.2, 0.2, 0.0],
    "loss": "cross_entropy",
    "optimizer": {
        "name": "adam",
        "hyperparameters": {"learning_rate": 0.001, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-08}
    },
    "settings": {"epochs": 20, "batch_size": 64, "shuffle": true, "seed": null, "l2": 0.0,
                 "patience": null, "monitor": "val_loss", "validation_split": 0.0}
}
```

Les réglages de `settings` sont ceux de la section `[training]` du [fichier de configuration](configuration.md), sans le coût, qui est enregistré à part. Un fichier plus ancien, qui n'a que les quatre premiers réglages, se charge toujours : les réglages absents prennent leur valeur par défaut, qui ne change rien à l'entraînement.

`dropout` donne le taux de [dropout](dropout.md) de chaque couche. Un fichier écrit avant l'ajout du dropout n'a pas ce champ : il se charge sans dropout.

## Ce qui est nécessaire, et ce qui ne l'est pas

Le sujet demande de réfléchir à ce qui est réellement nécessaire.

**Pour prédire**, il faut reconstruire exactement le même calcul : la taille des entrées, la taille et l'activation de chaque couche, puis les poids et les biais.

**Pour reprendre un entraînement**, il faut en plus :

- **le coût**, qui donne le gradient de départ ;
- **l'optimiseur**, avec son nom et ses hyperparamètres ;
- **sa mémoire** : sans elle, le momentum repartirait d'une vitesse nulle, et Adam referait sa correction des premiers pas ;
- **les réglages d'entraînement**, comme la taille de batch.

**Ce qui n'est pas sauvegardé :**

| Élément | Pourquoi |
|---|---|
| L'initialisation et la graine du réseau | Elles ne servent qu'à créer les premiers poids, que le fichier contient déjà |
| Les gradients et les dernières entrées des couches | Ils sont recalculés à chaque pas |
| L'historique des métriques | Il décrit un entraînement passé, et ne change pas le réseau |
| L'état du mélange des données | Voir la limite ci-dessous |

## Pourquoi ce format

| Format | Exact | Taille pour 213 507 paramètres | Sûr à charger |
|---|---|---|---|
| **Archive NumPy (choisi)** | Oui | 1,71 Mo | Oui, avec `allow_pickle=False` |
| JSON pur | Oui | 4,68 Mo | Oui |
| Pickle | Oui | Proche du binaire | **Non** |

**Les poids sont stockés en binaire.** Un nombre `float64` occupe 8 octets, et il est relu à l'identique. En texte, il faut en moyenne 20 caractères pour l'écrire sans perte, d'où un fichier JSON 2,7 fois plus gros. La lecture en est aussi plus lente : 82 ms pour décoder ce JSON, contre 7 ms pour charger entièrement le fichier `.nn`.

**La description reste lisible.** Elle est écrite en JSON dans l'archive, ce qui permet de la vérifier champ par champ.

**Aucun code n'est exécuté au chargement.** Pickle, le format de sérialisation natif de Python, peut exécuter n'importe quel code caché dans un fichier, et dépend du nom des classes Python. Le fichier est lu avec `allow_pickle=False` : NumPy refuse alors tout tableau d'objets Python, et ne lit que des nombres et du texte. C'est ce qui répond à la consigne de ne pas utiliser de sérialisation fragile.

Avec un optimiseur Adam, le fichier passe à 5,1 Mo, car il contient deux valeurs de mémoire par paramètre.

## Une écriture sûre

Le fichier est d'abord écrit sous un nom temporaire dans le même dossier, puis renommé à la place de l'ancien. Le renommage est immédiat : un programme interrompu pendant l'écriture laisse donc l'ancien réseau intact, et le fichier temporaire est supprimé. Cela compte pour `--train`, qui écrase le fichier chargé quand `--save` n'est pas donné.

`save_network` refuse aussi les valeurs non finies (`NaN` ou infini) : un réseau qui a divergé ne peut pas remplacer un bon réseau par erreur.

## Les vérifications au chargement

`load_network` lève une `MyTorchError`, qui mène au code de sortie 84, si :

- le fichier n'existe pas, ne peut pas être lu ou n'est pas une archive valide, par exemple s'il est tronqué ou contient un octet corrompu ;
- la description manque, n'est pas du JSON, ou n'est pas celle d'un réseau `my_torch` ;
- la **version** du format n'est pas `1`. Si le format change plus tard, un ancien fichier sera refusé avec un message clair, au lieu de donner des résultats faux ;
- un champ manque ou est en trop, ou une valeur est invalide : taille, activation, coût ou optimiseur inconnus, hyperparamètre hors limites ;
- un tableau manque, est en trop, n'a pas la bonne forme, ne contient pas de nombres décimaux ou contient des valeurs non finies ;
- la mémoire de l'optimiseur ne correspond pas à l'optimiseur ou au réseau.

Chaque message commence par `invalid network file` suivi du chemin du fichier, puis donne la raison, par exemple :

```
invalid network file 'net.nn': unsupported format version 2, expected 1
```

## Reprendre un entraînement

```python
saved = load_network("my_torch_network.nn")
train(saved.network, saved.loss, saved.optimizer, inputs, targets, 10, 64)
```

Un test vérifie que 10 epochs, une sauvegarde, un chargement puis 10 autres epochs donnent **exactement** les mêmes poids que 20 epochs d'un seul tenant, pour chaque optimiseur.

**Une limite :** cette égalité exacte suppose `shuffle=False`. L'état du générateur qui mélange les données n'est pas sauvegardé : après une reprise, les ordres de mélange ne sont pas ceux qu'aurait donnés l'entraînement d'un seul tenant. L'apprentissage reprend correctement, mais pas à l'identique.

## Comment c'est vérifié

[tests/test_serialization.py](../tests/test_serialization.py) vérifie notamment :

- les mêmes prédictions après un aller-retour, pour chaque activation, et les poids restaurés à l'identique ;
- la reprise exacte de l'entraînement pour chaque optimiseur ;
- l'écriture sûre et le refus des valeurs non finies ;
- le refus de 37 fichiers incohérents, ainsi que des fichiers absents, vides, tronqués, corrompus ou contenant un objet Python.

[tests/test_optimizers.py](../tests/test_optimizers.py) vérifie l'export et la restauration de la mémoire des optimiseurs.

Pour vérifier les tests eux-mêmes, 43 erreurs ont été introduites volontairement dans ce code, et chacune a fait échouer au moins un test.

Ces tests se lancent avec `make tests_run`.
