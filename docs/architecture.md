# L'architecture du module

Ce document décrit comment `my_torch` est découpé, comment les pièces travaillent ensemble pendant un entraînement, et pourquoi elles ont été séparées ainsi.

## Deux parties indépendantes

| Partie | Rôle | Connaît les échecs ? |
|---|---|---|
| [my_torch/](../my_torch) | Un module de réseaux de neurones générique : créer, entraîner, évaluer, sauvegarder | Non |
| [analyzer/](../analyzer) et [dataset/](../dataset) | Les échecs : lire les FEN, les encoder, les labels, les règles du jeu | Oui |

`my_torch` ne sait rien des échecs : il reçoit des tableaux de nombres et des cibles. Les tests l'entraînent d'ailleurs sur XOR et sur une régression, sans un seul échiquier. Tout ce qui concerne les échecs passe par `analyzer`, qui transforme une position en 769 nombres, comme l'explique [l'encodage](encodage.md), puis appelle `my_torch`.

Cette séparation permet de tester le module sur des problèmes dont on connaît la réponse, et de changer l'encodage des échecs sans toucher au module.

## Les pièces du module

| Fichier | Rôle |
|---|---|
| [layers.py](../my_torch/layers.py) | `Dense` : une couche entièrement connectée, `Z = X @ W + b`, avec son forward et son backward |
| [activations.py](../my_torch/activations.py) | `identity`, `sigmoid`, `tanh`, `relu`, `leaky_relu`, `softmax` : chacune avec sa dérivée |
| [initializers.py](../my_torch/initializers.py) | Les poids de départ : `uniform`, `normal`, `lecun`, `xavier_uniform`, `xavier_normal`, `he` |
| [network.py](../my_torch/network.py) | `Network` : une pile de couches, chacune suivie de son activation et de son dropout |
| [losses.py](../my_torch/losses.py) | `mse` et `cross_entropy`, avec le gradient du coût par rapport aux sorties |
| [optimizers.py](../my_torch/optimizers.py) | SGD, Momentum (et Nesterov), RMSProp, Adam : comment changer les poids à partir de leurs gradients |
| [training.py](../my_torch/training.py) | `train` : la boucle par epochs et mini-batchs, avec validation, L2, early stopping et planning du learning rate |
| [schedules.py](../my_torch/schedules.py) | Les plannings du learning rate |
| [mc_dropout.py](../my_torch/mc_dropout.py) | Les prédictions par MC dropout |
| [metrics.py](../my_torch/metrics.py) | Matrice de confusion, précision, rappel, F1 |
| [config.py](../my_torch/config.py) | La lecture et la vérification des fichiers de configuration |
| [serialization.py](../my_torch/serialization.py) | Le format `.nn` : sauvegarde et chargement |
| [generator.py](../my_torch/generator.py) | Le programme `my_torch_generator` |
| [search.py](../my_torch/search.py) | La recherche d'hyperparamètres |

## Un pas d'apprentissage

```
             forward                         backward
X ──► Dense ──► activation ──► ... ──► scores ──► coût
        ▲                                │
        │       gradients de W et b      ▼
        └──────────────── Dense ◄── activation ◄── ... ◄── gradient du coût

optimizer.step(layers) : W ← W - mise à jour(gradient de W), idem pour b
```

1. **Le forward.** `Network.forward` fait passer le batch dans chaque couche, puis dans son activation. Chaque couche garde ses entrées, et chaque activation garde ses valeurs `Z` : le backward en aura besoin.
2. **Le coût.** `loss.backward(scores, cibles)` donne le gradient du coût moyen par rapport aux scores.
3. **Le backward.** `Network.backward` remonte les couches à l'envers. Chaque activation multiplie le gradient par sa dérivée, chaque couche calcule les gradients de ses poids et de ses biais, puis le gradient de ses entrées, qu'elle passe à la couche d'avant. [La backpropagation](backpropagation.md) détaille ces calculs.
4. **Le pas.** `optimizer.step(layers)` change les poids à partir des gradients.

**La backpropagation et l'optimiseur sont séparés.** Le backward calcule les gradients sans modifier aucun poids, et l'optimiseur les utilise sans savoir d'où ils viennent. Changer d'optimiseur ne demande donc aucune modification du réseau, et la régularisation L2 s'insère simplement entre les deux, en ajoutant `l2 × W` aux gradients.

## Les choix de conception

**Des batchs en lignes.** Un batch est une matrice `(exemples, entrées)` et les poids sont de forme `(entrées, neurones)` : une couche calcule tout le batch en un seul produit matriciel, `X @ W`. C'est ce qui rend Python assez rapide, comme l'explique [le choix du langage](choix_du_langage.md).

**La cross-entropy reçoit des scores bruts.** La dernière couche utilise `identity`, et le coût applique lui-même softmax. Le gradient se simplifie alors en `probabilités - cibles`, plus stable que de dériver softmax et le logarithme séparément. La configuration refuse une dernière couche `softmax` avec la cross-entropy, qui appliquerait softmax deux fois.

**Tout se choisit par son nom.** `get_activation("relu")`, `get_loss("cross_entropy")`, `get_optimizer("adam", learning_rate=0.001)`, `get_schedule("plateau")` : un fichier de configuration ou un fichier `.nn` décrit un réseau avec des noms, sans code. Un nom inconnu donne une erreur qui liste les noms possibles.

**Tout est reproductible.** Chaque source de hasard a sa graine : les poids de départ, l'ordre des exemples, les masques du dropout, le découpage du dataset, la recherche d'hyperparamètres. Le même entraînement redonne les mêmes poids, ce qui a permis de refaire le réseau poussé avec `--train` et de retrouver exactement la précision de validation mesurée.

**Les erreurs sont vérifiées tôt.** Les tailles, les valeurs des réglages et les fichiers sont vérifiés avant tout calcul, avec un message qui dit ce qui ne va pas. Le module lève des `MyTorchError`, l'analyzer des `AnalyzerError` : les programmes les affichent et renvoient 84, comme le demande le sujet.

**Un fichier `.nn` sans code exécutable.** Il contient des tableaux NumPy et une description en JSON, et se lit sans `pickle` : un fichier `.nn` reçu ne peut pas exécuter de code. Il contient aussi la mémoire de l'optimiseur, pour reprendre un entraînement exactement là où il s'était arrêté. Voir [le format .nn](format_nn.md).

## Comment c'est vérifié

Chaque pièce a ses tests dans [tests/](../tests), plus de 1 000 au total, lancés par `make tests_run`. Les calculs du forward et du backward sont comparés à des différences finies, les optimiseurs à des valeurs calculées à la main, et l'entraînement à des boucles écrites à la main. Pour chaque fonctionnalité, des erreurs ont été introduites volontairement dans le code pour vérifier que les tests les détectent.
