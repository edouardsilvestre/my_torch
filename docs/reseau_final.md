# Le réseau pré-entraîné

Le sujet impose de pousser un réseau pré-entraîné dont le nom commence par `my_torch_network` : c'est lui que le correcteur évalue. Ce document explique comment [my_torch_network.nn](../my_torch_network.nn) a été obtenu, comment le refaire, et ce qu'il vaut.

## Le refaire

```
$> ./scripts/build_network.sh
```

[scripts/build_network.sh](../scripts/build_network.sh) enchaîne les commandes exactes qui ont produit le réseau :

1. **Préparer le dataset fourni avec le sujet,** `dataset.zip`, qui doit se trouver à la racine du dépôt. Les 723 019 positions sont rassemblées dans `datasets/official.txt`, comme l'explique [le document sur les données](dataset.md).
2. **Le découper,** avec la graine 0, en 572 449 positions d'entraînement, 71 556 de validation et 71 554 de test. Les 7 460 positions en double sont retirées avant le découpage.
3. **Créer un réseau vierge** depuis [configs/my_torch_network.conf](../configs/my_torch_network.conf), avec `my_torch_generator`.
4. **L'entraîner** sur le jeu d'entraînement avec `my_torch_analyzer --train`, en 24 epochs, avec une régularisation L2 de 1e-4.
5. **L'évaluer** sur la validation, puis sur le test.

L'ensemble prend environ 2 heures sur notre machine. Toutes les graines sont fixées, donc relancer le script redonne le même réseau. Le sujet exige de garder tous les scripts d'entraînement. `dataset.zip` n'est pas poussé : tout le monde le reçoit avec le sujet, et le sujet n'impose pas de pousser les datasets.

## Pourquoi le dataset fourni

Une première version du réseau, poussée avant, avait été entraînée sur 80 000 positions de nos [parties aléatoires](dataset.md). Elle obtenait 71,8 % sur nos propres positions de test. Le premier essai du correcteur ne lui a pourtant donné que **2 tests sur 10** : le lancement, et le niveau faible de « Check/Nothing ».

Mesurée ensuite sur le jeu de test du dataset fourni, cette première version ne faisait que **53,6 %**. Les positions du sujet viennent de vraies parties, alors que le réseau n'avait vu que des parties aléatoires : il voyait par exemple des échecs dans plus de la moitié des positions `Nothing`. **Le problème était donc les données, et non le réseau.**

Le dataset fourni contient 723 019 positions labellisées, du même genre que celles du correcteur. Nos propres règles des échecs donnent exactement le même label sur un échantillon de 18 000 d'entre elles, ce qui confirme aussi notre sens de la couleur.

## Le choix de la configuration

Trois configurations ont été entraînées 10 epochs sur le dataset fourni, puis comparées sur sa **validation**. Elles ont toutes 5 sorties, ReLU, l'initialisation He, Adam, un learning rate de 0,001 et des batchs de 64 :

| Configuration | Couches cachées | Validation 5 classes | Validation 2 classes | F1 macro | Meilleure epoch | Durée |
|---|---|---|---|---|---|---|
| [o1_256_64](../configs/candidates/official/o1_256_64.conf) | 256, 64 | 84,6 % | 92,2 % | 0,826 | 10 | 12,5 min |
| [o2_512_128](../configs/candidates/official/o2_512_128.conf) | 512, 128 | 84,8 % | 92,6 % | 0,826 | 6 (85,2 %) | 28 min |
| [o3_256_256_64](../configs/candidates/official/o3_256_256_64.conf) | 256, 256, 64 | **85,3 %** | **92,7 %** | **0,833** | 10 | 17,4 min |

La comparaison se refait avec :

```
$> python3 -m scripts.compare_configs --train datasets/official_train.txt \
       --validation datasets/official_validation.txt configs/candidates/official/*.conf
```

**Ce qu'on en lit :**

- **Le surapprentissage a beaucoup diminué.** Après 10 epochs, l'écart entre l'entraînement et la validation est de 10 à 13 points, contre 27 points avec nos 80 000 positions aléatoires. Avec beaucoup plus d'exemples, le réseau apprend moins par cœur.
- **Les écarts entre configurations restent faibles,** moins d'un point. Le réseau le plus large plafonne dès l'epoch 6, alors que les deux autres progressent encore à l'epoch 10.

**Le choix : `o3_256_256_64`.** C'est la meilleure configuration sur toutes les mesures de validation, et elle est plus rapide que la plus large. Ce choix ne s'appuie que sur la validation, et **le jeu de test n'a servi qu'à la mesure finale**.

## Moins d'overfitting avec la régularisation L2

Entraînée 10 epochs sans régularisation, cette configuration faisait 95,8 % sur l'entraînement mais 85,3 % sur la validation. [La régularisation L2 et l'early stopping](regularisation.md) ont ensuite été mesurés sur elle, toujours sur la validation :

| `l2` | Epoch gardée par l'early stopping | Entraînement | Validation 5 classes | Écart |
|---|---|---|---|---|
| 0 | 10 | 95,8 % | 85,3 % | 10,5 points |
| **0,0001** | **24** | 92,7 % | **89,1 %** | **3,6 points** |
| 0,001 | 21 | 82,2 % | 80,6 % | 1,6 point |

**Le réseau poussé utilise `l2 = 1e-4`, entraîné 24 epochs.** La pénalité empêche le réseau de retenir les positions d'entraînement, et il continue de progresser en validation bien plus longtemps. Avec `1e-3`, la pénalité est trop forte : le réseau n'apprend même plus bien l'entraînement.

`--train` ne voit que le jeu d'entraînement : le réseau est donc réentraîné jusqu'à l'epoch 24, sans patience. Comme toutes les graines sont fixées, il est identique au réseau gardé par l'early stopping : sa validation est exactement la même, 89,07 %.

## Ce qu'il vaut

Sur le **jeu de test du dataset fourni**, 71 554 positions que le réseau n'a jamais vues, avec la répartition du dataset (41,6 % de `Nothing`, 34,7 % d'échecs, 23,7 % de mats) :

| Niveau du correcteur | Première version | Dataset fourni, sans L2 | Ce réseau |
|---|---|---|---|
| `Check` / `Nothing` | 71,6 % | 92,6 % | **95,0 %** |
| `Checkmate` / `Check` / `Nothing` | 53,6 % | 84,9 % | **88,9 %** |
| Avec les couleurs | 53,6 % | 84,9 % | **88,9 %** |

**Par classe :**

| Classe | Précision | Rappel | F1 |
|---|---|---|---|
| `Nothing` | 0,95 | 0,93 | 0,94 |
| `Check White` | 0,87 | 0,84 | 0,86 |
| `Check Black` | 0,85 | 0,83 | 0,84 |
| `Checkmate White` | 0,83 | 0,89 | 0,86 |
| `Checkmate Black` | 0,83 | 0,89 | 0,86 |

- **Toutes les classes progressent** par rapport au réseau sans L2, de 3 à 6 points de F1.
- **Les couleurs ne sont jamais confondues :** aucune erreur de couleur sur 41 798 échecs et mats.
- **La difficulté principale reste de séparer un échec d'un mat :** 2 703 échecs sont pris pour des mats, et 1 663 mats pour des échecs, contre 2 859 et 2 645 sans L2.

Sur nos 10 000 positions de test aléatoires, ce réseau fait aussi mieux : 80,7 % avec les couleurs, contre 74,1 % sans L2 et 71,8 % pour la première version.

**Sur les 5 positions de l'exemple du sujet,** il en trouve 4. Il prend toujours le mat du fou pour `Check Black`, mais il ne voit plus d'échec dans la position avec seulement les deux rois. La première version les trouvait toutes les 5. Ces 5 positions pèsent peu face aux 71 554 du jeu de test, mais elles montrent que des erreurs restent possibles sur des positions simples.

## Le fichier poussé

`my_torch_network.nn` pèse 6,7 Mo. Il contient le réseau, `769 → 256 → 256 → 64 → 5`, mais aussi le coût, l'optimiseur Adam avec sa mémoire, et les réglages d'entraînement. `my_torch_analyzer --train` peut donc continuer à l'entraîner.

[tests/test_pushed_network.py](../tests/test_pushed_network.py) vérifie :

- que c'est le seul réseau dont le nom commence par `my_torch_network` ;
- qu'il correspond à sa configuration ;
- qu'il peut encore être entraîné ;
- que le vrai exécutable répond à l'exemple du sujet avec un label valide par ligne.

## Les limites et la suite

- **Un écart de 3,6 points** entre l'entraînement et la validation, contre 10,5 sans régularisation. [Le dropout](dropout.md) le réduit encore, mais fait aussi baisser la validation : il n'est pas utilisé.
- **Des positions inconnues.** Les fichiers du correcteur ne sont pas exactement le dataset fourni : sa précision peut différer de celle mesurée ici.
- **Le format aux niveaux simples.** Le réseau répond toujours avec les couleurs. Le premier essai suggère que le correcteur les accepte, mais ce n'est pas certain.
