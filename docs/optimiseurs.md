# Les optimiseurs

Ce document explique comment le module `my_torch` met à jour les paramètres d'un réseau, et comment ce calcul est vérifié.

## Le rôle de l'optimiseur

La [backpropagation](backpropagation.md) calcule le gradient de chaque paramètre, c'est-à-dire le sens dans lequel il fait monter le coût. Elle ne modifie aucun paramètre : c'est le rôle de l'optimiseur, qui décide de combien bouger chacun d'eux.

```python
network.backward(loss.backward(network.forward(inputs), targets))   # calcule les gradients
optimizer.step(network.layers)                                      # met à jour les paramètres
```

Séparer ces deux rôles permet de changer d'optimiseur sans toucher à la backpropagation.

## L'interface commune

Tous les optimiseurs héritent de `Optimizer`, et s'utilisent donc de la même façon :

- **`step(layers)`** parcourt les couches et retire à chaque paramètre ce que renvoie `compute_update`. Appeler `step` avant `backward` lève une `MyTorchError`.
- **`compute_update(key, gradient)`** contient la formule propre à chaque optimiseur. `key` identifie le paramètre, par exemple `(0, "weights")`, pour retrouver ce que l'optimiseur a gardé en mémoire à son sujet.

`get_optimizer` crée un optimiseur à partir de son nom et de ses paramètres :

```python
get_optimizer("sgd", learning_rate=0.5)
get_optimizer("momentum", learning_rate=0.05, momentum=0.8, nesterov=True)
get_optimizer("adam")
```

Un optimiseur garde en mémoire des valeurs propres à un réseau. Il faut donc en créer **un par réseau**, sans quoi ces mémoires se mélangent.

## SGD

```
mise à jour = learning_rate × gradient
```

C'est la descente de gradient la plus simple. Le learning rate règle la taille du pas : trop grand, le coût peut osciller ou diverger ; trop petit, l'apprentissage est très lent.

Ici, le gradient est calculé sur le batch reçu : le choix entre batch complet, mini-batch et stochastique revient à la [boucle d'entraînement](entrainement.md).

## Momentum

SGD ne regarde que le gradient du moment. Là où le coût forme une vallée étroite, il rebondit d'une paroi à l'autre et avance lentement dans le fond.

Le momentum accumule les gradients passés dans une **vitesse** :

```
vitesse = momentum × vitesse + gradient
mise à jour = learning_rate × vitesse
```

Les directions qui se répètent s'additionnent, et celles qui changent de signe à chaque pas s'annulent. À `momentum = 0`, on retrouve SGD. À `momentum = 1`, la vitesse ne s'atténuerait jamais : cette valeur est refusée.

## Nesterov

Le momentum calcule le gradient là où se trouvent les paramètres. Nesterov le calcule là où la vitesse va les emmener, le **point d'anticipation**, ce qui lui permet de corriger sa trajectoire plus tôt.

La version d'origine demande de faire passer le réseau par ce point, avec des paramètres différents de ceux qu'on garde. On utilise une forme réécrite, qui ne garde que le point d'anticipation :

```
vitesse = momentum × vitesse + gradient
mise à jour = learning_rate × (gradient + momentum × vitesse)
```

Les deux formes suivent la même trajectoire. Notons `θ` les paramètres de la version d'origine, `v` la vitesse, `μ` le momentum, `lr` le learning rate et `φ = θ - lr × μ × v` le point d'anticipation :

```
v' = μ × v + gradient(φ)
θ' = θ - lr × v'

φ' = θ' - lr × μ × v'
   = θ - lr × v' - lr × μ × v'
   = φ + lr × μ × v - lr × v' - lr × μ × v'
   = φ - lr × gradient(φ) - lr × μ × v'
   = φ - lr × (gradient(φ) + μ × v')
```

La dernière ligne est la forme réécrite, appliquée à `φ`. Un test le vérifie sur 50 pas.

## RMSProp

SGD et Momentum utilisent le même learning rate pour tous les paramètres, alors que certains reçoivent de très grands gradients, et d'autres de tout petits. RMSProp divise le pas de chaque paramètre par la taille récente de ses gradients :

```
carré = decay × carré + (1 - decay) × gradient²
mise à jour = learning_rate × gradient / (√carré + epsilon)
```

La racine ramène la moyenne des carrés à l'échelle du gradient. Multiplier tous les gradients par 1000 laisse donc les pas presque inchangés. `epsilon` évite une division par zéro quand un gradient est resté nul, par exemple sur un neurone ReLU inactif.

**Une limite mesurée.** Le pas ne diminue pas près de la solution, puisque le gradient y est divisé par sa propre taille. Sur la régression, le paramètre finit par sauter d'un côté à l'autre de la solution à chaque pas, et l'écart vaut alors `0.5 × learning_rate`. Avec un learning rate de `0.01`, cet écart reste de `5e-3` après 2 000 comme après 20 000 pas. Il faut donc réduire le learning rate pour s'approcher davantage de la solution.

## Adam

Adam combine deux mémoires : la moyenne des gradients pour la direction, proche du momentum, et la moyenne de leurs carrés pour la taille du pas, comme RMSProp.

```
moyenne = beta1 × moyenne + (1 - beta1) × gradient
carré   = beta2 × carré   + (1 - beta2) × gradient²

moyenne corrigée = moyenne / (1 - beta1^pas)
carré corrigé    = carré   / (1 - beta2^pas)

mise à jour = learning_rate × moyenne corrigée / (√carré corrigé + epsilon)
```

Les moyennes partent de 0, donc elles sont trop petites au début, et la **correction** compense ce démarrage. Sans elle, le premier pas vaudrait environ 3,16 fois le learning rate. Avec elle, il vaut environ 1 fois le learning rate. RMSProp n'a pas cette correction : son premier pas vaut aussi environ 3,16 fois le learning rate.

Le compteur de pas augmente de 1 à chaque appel à `step`, et non à chaque paramètre.

## Des exemples chiffrés

Dans ces exemples, le paramètre part de 0 et le learning rate vaut `0.1`.

**Momentum et Nesterov**, avec un momentum de `0.9` et un gradient constant de 1 :

| Pas | Vitesse | Mise à jour Momentum | Paramètre Momentum | Mise à jour Nesterov | Paramètre Nesterov |
|---|---|---|---|---|---|
| 1 | `1` | `0.1` | `-0.1` | `0.19` | `-0.19` |
| 2 | `1.9` | `0.19` | `-0.29` | `0.271` | `-0.461` |
| 3 | `2.71` | `0.271` | `-0.561` | `0.3439` | `-0.8049` |

**RMSProp**, avec un `decay` de `0.9` et des gradients de `2` puis `-1` :

| Pas | Gradient | Carré | Mise à jour | Paramètre |
|---|---|---|---|---|
| 1 | `2` | `0.4` | `0.31623` | `-0.31623` |
| 2 | `-1` | `0.46` | `-0.14744` | `-0.16879` |

**Adam**, avec les `beta` par défaut et les mêmes gradients :

| Pas | Gradient | Moyenne | Carré | Moyenne corrigée | Carré corrigé | Mise à jour | Paramètre |
|---|---|---|---|---|---|---|---|
| 1 | `2` | `0.2` | `0.004` | `2` | `4` | `0.1` | `-0.1` |
| 2 | `-1` | `0.08` | `0.004996` | `0.42105` | `2.49925` | `0.02663` | `-0.12663` |

## Les paramètres

| Optimiseur | Paramètre | Par défaut | Valeurs acceptées |
|---|---|---|---|
| Tous | `learning_rate` | `0.01` pour SGD et Momentum, `0.001` pour RMSProp et Adam | `> 0` |
| Momentum | `momentum` | `0.9` | `[0, 1[` |
| Momentum | `nesterov` | `False` | `True` ou `False` |
| RMSProp | `decay` | `0.9` | `[0, 1[` |
| RMSProp, Adam | `epsilon` | `1e-8` | `> 0` |
| Adam | `beta1` | `0.9` | `[0, 1[` |
| Adam | `beta2` | `0.999` | `[0, 1[` |

Ces valeurs par défaut sont des points de départ courants. Celles d'Adam viennent de sa publication d'origine. Aucune n'a été mesurée sur le réseau d'échecs. Elles ne suffisent pas toujours : avec 2 000 epochs sur XOR, Adam et RMSProp à `0.001` donnent rarement un coût inférieur à `0.01`.

## Le learning rate

Ces tableaux se reproduisent depuis la racine du dépôt, en cinq minutes environ, avec `python3 -m scripts.optimizer_learning_rates`. Ils portent sur les problèmes jouets des tests : une régression `y = 2a - 3b + 1` avec une couche linéaire et MSE, et XOR avec 4 neurones `tanh` et la cross-entropy. Le momentum vaut `0.9`.

**Régression, 500 pas, graine 0 :**

| Learning rate | SGD | Momentum | Nesterov |
|---|---|---|---|
| 0.5 | Converge | Converge | Converge |
| 0.6 | Converge | Converge | Converge |
| 0.7 | Converge | Converge | Diverge |
| 0.9 | Converge | Converge | Diverge |
| 1.1 | Diverge | Converge | Diverge |
| 1.5 | Diverge | Converge | Diverge |
| 1.8 | Diverge | Converge | Diverge |
| 2.0 | Diverge | Diverge | Diverge |

**XOR, 2000 epochs, graines réussies sur 30 :**

| Learning rate | SGD | Momentum | Nesterov |
|---|---|---|---|
| 0.5 | 30 | 30 | 30 |
| 1 | 30 | 30 | 29 |
| 2 | 30 | 29 | 17 |
| 5 | 16 | 1 | 0 |
| 10 | 0 | 0 | 0 |

**Le bon learning rate dépend à la fois de l'optimiseur et du problème.** Sur la régression, le momentum supporte des valeurs plus grandes que SGD. Sur XOR, il échoue plus tôt. Nesterov est le plus sensible des trois sur ces deux problèmes.

On entend parfois qu'il faut diviser le learning rate par 10 avec un momentum de `0.9`, car la vitesse tend vers `gradient / (1 - momentum)`. Ce calcul n'est juste que pour un gradient constant, et il ne prédit pas ce que montrent ces tableaux sur ces deux problèmes.

## Comment c'est vérifié

**Les formules.** [tests/test_optimizers.py](../tests/test_optimizers.py) vérifie les valeurs ci-dessus, calculées à la main, et des cas particuliers : `momentum = 0` donne SGD, un gradient nul ne déplace pas le paramètre, chaque paramètre a sa propre mémoire.

**L'apprentissage.** [tests/test_learning.py](../tests/test_learning.py) vérifie que chaque optimiseur apprend XOR et trouve les paramètres de la régression, sur 3 graines. Ce tableau mesure ces mêmes réglages sur plus de graines :

| Optimiseur | XOR : learning rate | Réussite sur 100 graines | Régression : learning rate | Pire écart sur 20 graines |
|---|---|---|---|---|
| SGD | 0.5 | 100 | 0.1 | 2.2e-15 |
| Momentum | 0.05 | 100 | 0.05 | 4.4e-16 |
| Nesterov | 0.05 | 100 | 0.05 | 4.4e-16 |
| RMSProp | 0.1 | 100 | 0.01 | 5.0e-03 |
| Adam | 0.01 | 100 | 0.01 | 1.2e-05 |

**La détection des erreurs.** Pour vérifier les tests eux-mêmes, 32 erreurs ont été introduites volontairement dans le code. Les tests de formules les ont toutes détectées, et les tests d'apprentissage seuls n'en ont détecté que 14. Par exemple, un réseau apprend encore XOR quand l'option Nesterov est ignorée, ou quand Adam n'a pas sa correction. Les deux types de tests sont donc nécessaires.

Ces tests se lancent avec `make tests_run`.

## Ce que ces mesures ne montrent pas

- **Quel optimiseur est le meilleur.** Chaque optimiseur a ici un learning rate choisi pour ces tests : ces réglages ne permettent pas de les classer.
- **Quel optimiseur choisir pour le réseau d'échecs.** XOR et cette régression sont minuscules, et sans mini-batch. Ce choix devra s'appuyer sur les benchmarks de ce réseau.
