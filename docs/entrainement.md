# L'entraînement

Ce document explique comment le module `my_torch` entraîne un réseau sur un jeu de données, et comment cette boucle est vérifiée.

## Le rôle de la boucle

Un pas d'apprentissage enchaîne trois opérations : le forward, la [backpropagation](backpropagation.md) et la mise à jour par [l'optimiseur](optimiseurs.md). La fonction `train` répète ce pas sur tout le jeu de données, autant de fois qu'on le demande :

```python
from my_torch import Network, get_loss, get_optimizer, train

network = Network(768, [(256, "relu"), (64, "relu"), (3, "identity")], seed=0)
history = train(
    network, get_loss("cross_entropy"), get_optimizer("adam"),
    inputs, targets, epochs=20, batch_size=64, seed=0,
    validation=(validation_inputs, validation_targets), verbose=True,
)
```

`train` est une fonction à part, et non une méthode de `Network`. Le réseau reste une structure, qui sait calculer ses sorties et ses gradients, mais ne connaît ni le coût ni l'optimiseur.

## Epochs et batchs

Une **epoch** est un passage complet sur le jeu d'entraînement. Pendant une epoch, les exemples sont découpés en **batchs**, et l'optimiseur fait un pas par batch.

| Mode | `batch_size` | Pas par epoch, pour 1 000 exemples | Gradient |
|---|---|---|---|
| Batch complet | taille du jeu | 1 | Exact, mais un seul pas par epoch |
| Mini-batch | par exemple 64 | 16 | Estimé sur 64 exemples |
| Stochastique | 1 | 1 000 | Estimé sur un seul exemple, très bruité |

Un petit batch fait plus de pas par epoch, mais chacun repose sur une estimation plus bruitée du gradient. Un batch plus grand donne une estimation plus fiable, et profite mieux des produits matriciels de NumPy. Le learning rate et la taille de batch se règlent donc ensemble : les tests sur XOR utilisent par exemple un learning rate de `0.1` en stochastique, contre `0.5` avec des batchs de 2 ou 4 exemples. La comparaison des tailles de batch sur le réseau d'échecs fait partie des benchmarks.

Si la taille du jeu n'est pas un multiple de `batch_size`, le dernier batch est plus petit : aucun exemple n'est ignoré. Comme le gradient du coût est une moyenne sur le batch, ce dernier pas n'est pas plus grand que les autres. Un `batch_size` plus grand que le jeu revient au batch complet.

## Le mélange

Au début de chaque epoch, l'ordre des exemples est tiré au hasard. Les batchs changent donc d'une epoch à l'autre, et le réseau ne peut pas profiter d'un ordre particulier des données, par exemple un fichier trié par classe.

Le mélange est **reproductible** : la même graine `seed` redonne les mêmes ordres. Cette graine est distincte de celle du réseau, qui fixe les poids de départ. Les exemples et leurs cibles sont mélangés ensemble, et le jeu de données reçu n'est pas modifié.

`shuffle=False` garde l'ordre d'origine. En batch complet, `train` fait alors exactement les mêmes calculs qu'une boucle écrite à la main : c'est ce qui permet aux scripts de redonner les tableaux de [la backpropagation](backpropagation.md) et des [optimiseurs](optimiseurs.md) à l'identique. Mélanger change l'ordre des additions, donc les arrondis.

## Le jeu de validation

Le paramètre facultatif `validation` reçoit un second jeu de données. Il est évalué à la fin de chaque epoch, mais ne sert **jamais** à apprendre : un test vérifie que les poids obtenus sont identiques, avec ou sans validation.

Comparer le coût d'entraînement à celui de validation permet de repérer l'**overfitting** : le réseau continue de progresser sur les exemples qu'il voit, mais plus sur les autres.

## L'historique

`train` renvoie un dictionnaire, avec une liste de valeurs par métrique et une valeur par epoch :

```python
{
    "loss": [0.74, 0.73, ...],
    "accuracy": [0.75, 0.75, ...],
    "val_loss": [0.74, 0.73, ...],
    "val_accuracy": [0.75, 0.75, ...],
}
```

| Clé | Contenu | Présente si |
|---|---|---|
| `loss` | Coût sur le jeu d'entraînement | Toujours |
| `accuracy` | Part des exemples bien classés | Les cibles sont en one-hot |
| `val_loss`, `val_accuracy` | Les mêmes mesures sur le jeu de validation | `validation` est fourni |

Les valeurs sont des nombres Python ordinaires : l'historique peut être enregistré en JSON, puis servir à tracer les courbes d'apprentissage.

## Comment les métriques sont calculées

**Après l'epoch, sur tout le jeu.** Le coût d'entraînement est recalculé à la fin de chaque epoch, avec les poids obtenus. La moyenne des coûts des batchs aurait été gratuite, mais chacun aurait été mesuré avec des poids différents, encore en cours d'apprentissage. Le recalcul donne une valeur exacte, comparable à celle de la validation.

**Par morceaux de 1 024 exemples.** Le jeu est évalué par morceaux, pour ne pas garder en mémoire les sorties de toutes les couches pour tout le jeu à la fois. Chaque morceau compte en proportion de son nombre d'exemples, donc le résultat est le même que sur le jeu entier, aux arrondis près.

**L'accuracy seulement pour une classification.** Elle n'est calculée que si chaque cible contient un seul 1 et des 0, sur au moins deux colonnes. Pour une régression, elle n'aurait pas de sens.

**Ce calcul a un coût.** C'est un forward de plus par epoch. Il reste faible devant l'entraînement d'un réseau de la taille du réseau d'échecs. Sur un réseau minuscule comme celui de XOR, il coûte en revanche plus cher que l'apprentissage lui-même, ce qui ralentit les tests et les scripts.

## L'affichage

Avec `verbose=True`, `train` écrit une ligne par epoch sur la sortie standard :

```
epoch  1/12 - loss 0.7421 - accuracy 0.7500 - val_loss 0.7421 - val_accuracy 0.7500
epoch 12/12 - loss 0.6879 - accuracy 0.7500 - val_loss 0.6879 - val_accuracy 0.7500
```

L'affichage est désactivé par défaut. Il n'est utile que pendant un entraînement, et le sujet ne pose aucune limite sur la sortie en mode `--train`.

## Les erreurs

`train` lève une `MyTorchError` si :

- `epochs` ou `batch_size` n'est pas un entier strictement positif ;
- les entrées ou les cibles ne sont pas des tableaux à deux dimensions ;
- les entrées et les cibles n'ont pas le même nombre d'exemples ;
- le jeu d'entraînement ou de validation est vide.

## Comment c'est vérifié

[tests/test_training.py](../tests/test_training.py) vérifie notamment :

- la taille des batchs, et que chaque exemple est vu une fois par epoch ;
- que le mélange change d'une epoch à l'autre, et que la même graine redonne le même ordre ;
- que chaque cible reste associée à son exemple ;
- qu'en batch complet sans mélange, les poids sont identiques à ceux d'une boucle écrite à la main ;
- que la validation ne change rien à l'apprentissage ;
- les valeurs de l'historique, le calcul par morceaux et l'affichage ;
- que le coût diminue et que le réseau apprend XOR, en batch complet, en mini-batch et en stochastique ;
- la régularisation L2 et l'early stopping, décrits dans [leur propre document](regularisation.md).

Pour vérifier les tests eux-mêmes, 30 erreurs ont été introduites volontairement dans la boucle, et chacune a fait échouer au moins un test.

Ces tests se lancent avec `make tests_run`.

## Ce qui viendra

La boucle est prévue pour accueillir plus tard :

- l'arrêt propre d'un long entraînement avec Ctrl+C ;
- le dropout, actif seulement pendant l'apprentissage ;
- un learning rate qui évolue au fil des epochs.
