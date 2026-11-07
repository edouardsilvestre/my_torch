# La backpropagation

Ce document explique comment le module `my_torch` calcule les gradients, et comment ce calcul est vérifié.

## Le problème

Pour apprendre, un réseau doit savoir, pour chacun de ses paramètres, si l'augmenter ou le diminuer fait baisser le coût, et dans quelle mesure. Ce nombre est le **gradient** du paramètre.

On pourrait l'estimer en décalant chaque paramètre d'un petit pas dans les deux sens et en mesurant la variation du coût : ce sont les **différences finies**. Cela demande deux passages dans le réseau par paramètre, soit environ 427 000 passages à chaque étape pour un réseau `768 → 256 → 64 → 3` et ses 213 507 paramètres.

La backpropagation obtient tous ces gradients en **un seul passage arrière**.

## Le principe

La backpropagation applique la règle de dérivation en chaîne, à l'envers du réseau. Chaque élément reçoit le gradient de ce qu'il a produit, et le transforme en gradient de ce qu'il a reçu.

```
Forward :   X → Dense 1 → activation 1 → Dense 2 → activation 2 → sorties → coût

Backward :  dW₁, db₁ ← Dense 1 ← activation 1 ← dW₂, db₂ ← Dense 2 ← activation 2 ← gradient du coût
```

Le chemin arrière applique les mêmes étapes que le forward, dans l'ordre inverse. L'activation est donc traitée avant la couche.

## Le cycle d'apprentissage

Une étape d'apprentissage comporte quatre opérations :

```python
outputs = network.forward(inputs)            # 1. calcule les sorties
gradient = loss.backward(outputs, targets)   # 2. gradient du coût par rapport aux sorties
network.backward(gradient)                   # 3. remplit le gradient de chaque couche
optimizer.step(network.layers)               # 4. met à jour les paramètres
```

La mise à jour la plus simple retire à chaque paramètre son gradient, multiplié par le learning rate. On le retire, car il indique le sens dans lequel le coût **monte**.

La backpropagation s'arrête à l'étape 3 : elle calcule les gradients, mais ne modifie aucun paramètre. L'étape 4 revient à [l'optimiseur](optimiseurs.md). Séparer ces deux rôles permet de changer de méthode de mise à jour sans toucher à la backpropagation.

La fonction `train` répète ce cycle sur chaque batch du jeu de données : voir [l'entraînement](entrainement.md).

## La couche dense

Une couche dense calcule `Z = X @ W + b`. Les exemples sont rangés par ligne : `X` a pour forme `(batch, entrées)`.

Pour un exemple, le neurone `j` calcule `Z_j = Σᵢ X_i × W_ij + b_j`. On note `g_j` l'effet de `Z_j` sur le coût : c'est le gradient que la couche reçoit.

| Si on augmente de 1... | ...`Z_j` augmente de... | Gradient | Dans le code |
|---|---|---|---|
| Le poids `W_ij` | `X_i` | `X_i × g_j` | `self.inputs.T @ gradient` |
| Le biais `b_j` | 1 | `g_j` | `gradient.sum(axis=0)` |
| L'entrée `X_i` | `W_ij`, pour chaque neurone `j` | `Σⱼ W_ij × g_j` | `gradient @ self.weights.T` |

`Dense.backward` garde les gradients de `W` et de `b` dans `weights_gradient` et `biases_gradient`, puis renvoie le gradient des entrées, qui devient le gradient reçu par la couche précédente.

Le calcul a besoin des entrées `X` : `Dense.forward` les mémorise. C'est pourquoi appeler `backward` avant `forward` lève une `MyTorchError`.

## Les activations

Une activation calculée case par case fait remonter le gradient en le multipliant par sa dérivée, évaluée en `z`, la valeur avant activation : `gradient * derivative(z)`.

Softmax est un cas à part : chacune de ses sorties dépend de tous les `z`, donc sa dérivée est une matrice par exemple. En la multipliant par le gradient et en simplifiant, cette matrice disparaît :

```
backward = s * (gradient - Σ(s * gradient))        avec s = softmax(z)
```

`Activation.backward` donne ainsi une interface unique, et le réseau n'a aucun cas particulier à gérer.

## La fonction de coût

Le gradient de départ vient de `Loss.backward`.

Pour classer, on utilise la cross-entropy, calculée à partir des **scores bruts** de la dernière couche, dont l'activation est `identity`. Son gradient se simplifie :

```
gradient = (softmax(z) - cible) / taille_du_batch
```

Il est négatif pour la bonne classe, dont le score doit monter, et positif pour les autres.

Enchaîner Softmax puis `ln(p)` casserait quand le réseau se trompe avec assurance : `p` vaut 0, donc `ln(p)` vaut moins l'infini, alors que c'est le moment où il a le plus à apprendre. En calculant le logarithme directement à partir des scores, le coût et le gradient restent exacts.

## La moyenne sur le batch

Le coût est une moyenne sur le batch, donc son gradient contient déjà la division par la taille du batch. La couche fait donc une **somme** sur les exemples, et non une moyenne : une moyenne diviserait deux fois.

Le coût garde ainsi la même échelle quelle que soit la taille du batch, ce qui évite de devoir ajuster le learning rate en fonction d'elle.

## Un exemple chiffré

Prenons le réseau `Network(2, [(2, "relu"), (1, "identity")])`, avec ces paramètres :

```
Couche 1 : W = [[1, -1],    b = [0, -1]        Couche 2 : W = [[2],    b = [1]
                [0,  2]]                                       [3]]
```

**Forward**, sur `X = [[1, 2], [3, 0]]` :

| Étape | Exemple `[1, 2]` | Exemple `[3, 0]` |
|---|---|---|
| Couche 1, `z` | `[1, 2]` | `[3, -4]` |
| Après `relu` | `[1, 2]` | `[3, 0]` |
| Couche 2 et `identity` | `9` | `7` |

**Gradient de départ**, avec MSE et la cible `[[8], [8]]` : `2 × écart / 2`, soit `[[1], [-1]]`.

**Backward, couche 2 :**

| Élément | Résultat |
|---|---|
| Après `identity` | `[[1], [-1]]` |
| `dW₂` | `[[-2], [2]]` |
| `db₂` | `[0]` |
| Gradient vers la couche 1 | `[[2, 3], [-2, -3]]` |

**Backward, couche 1 :**

| Élément | Résultat |
|---|---|
| Après `relu` | `[[2, 3], [-2, 0]]` |
| `dW₁` | `[[-4, 3], [4, 6]]` |
| `db₁` | `[0, 3]` |

Le `-3` devenu `0` montre l'effet de ReLU : pour le second exemple, le second neurone avait `z = -4`, donc sa sortie était annulée. Il n'a pas contribué au résultat, et il ne reçoit aucune correction pour cet exemple.

## Comment c'est vérifié

**Le gradient checking.** [tests/test_backpropagation.py](../tests/test_backpropagation.py) compare les gradients de tous les paramètres aux différences finies, qui n'utilisent pas nos formules. Le test porte sur 24 petits réseaux : 6 activations cachées, combinées à 4 couples de dernière activation et de coût.

Cette diversité est nécessaire : remplacer `z` par la sortie après activation ne change rien sur un réseau qui n'utilise que `relu` et `identity`, car `relu(z) > 0` équivaut à `z > 0`. L'erreur n'est visible qu'avec les autres activations.

**Les valeurs calculées à la main.** Le même fichier vérifie l'exemple ci-dessus, et [tests/test_layers.py](../tests/test_layers.py) le backward d'une couche seule.

**L'apprentissage.** [tests/test_learning.py](../tests/test_learning.py) vérifie qu'un petit réseau apprend XOR, et qu'une couche linéaire seule n'y arrive pas.

Ces tests se lancent avec `make tests_run`.

## Résultats sur XOR

XOR n'est pas séparable par une droite : il faut une couche cachée et une activation. Ce tableau donne le taux de réussite (les 4 réponses justes) sur 100 graines, avec une descente de gradient simple, 2000 epochs et un learning rate de 0,5. Il se reproduit depuis la racine du dépôt, en trois minutes environ, avec `python3 -m scripts.xor_convergence`.

| Architecture | Réussite | Coût médian | Graines en échec |
|---|---|---|---|
| Une seule couche linéaire | 0 / 100 | 0.6931 | ... |
| 2 neurones `relu` | 22 / 100 | 0.4774 | ... |
| 2 neurones `tanh` | 64 / 100 | 0.0017 | ... |
| 4 neurones `relu` | 66 / 100 | 0.0011 | ... |
| 8 neurones `relu` | 96 / 100 | 0.0008 | 10, 58, 62, 66 |
| 4 neurones `tanh` | 100 / 100 | 0.0012 |  |

`...` signifie que plus de 10 graines ont échoué : le script ne les liste pas.

Ce que ces résultats montrent :

- La couche linéaire échoue toujours. Son coût de 0,6931 vaut `ln(2)` : elle répond 50/50 sur tout.
- L'apprentissage ne réussit pas à tous les coups : il dépend des poids de départ.
- Sur ce problème, ajouter des neurones améliore la réussite.

Ce qu'ils ne montrent pas :

- **Pourquoi certaines graines échouent.** L'hypothèse la plus probable pour ReLU est celle des neurones morts, dont le gradient est nul, mais elle n'a pas été vérifiée.
- **Quelle activation choisir pour le réseau d'échecs.** XOR est un problème minuscule, et ces réglages ne sont pas ceux du réseau final. Ce choix devra s'appuyer sur les benchmarks de ce réseau.
