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
4. **L'entraîner** sur le jeu d'entraînement avec `my_torch_analyzer --train`, en 10 epochs.
5. **L'évaluer** sur la validation, puis sur le test.

L'ensemble prend environ 18 minutes sur notre machine. Toutes les graines sont fixées, donc relancer le script redonne le même réseau. Le sujet exige de garder tous les scripts d'entraînement. `dataset.zip` n'est pas poussé : tout le monde le reçoit avec le sujet, et le sujet n'impose pas de pousser les datasets.

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

**Le choix : `o3_256_256_64`, avec 10 epochs.** C'est la meilleure configuration sur toutes les mesures de validation, et elle est plus rapide que la plus large. Comme elle progresse encore à l'epoch 10, quelques epochs de plus aideraient peut-être un peu : ce n'est pas mesuré ici. Ce choix ne s'appuie que sur la validation, et **le jeu de test n'a servi qu'à la mesure finale**.

## Ce qu'il vaut

Sur le **jeu de test du dataset fourni**, 71 554 positions que le réseau n'a jamais vues, avec la répartition du dataset (41,6 % de `Nothing`, 34,7 % d'échecs, 23,7 % de mats) :

| Niveau du correcteur | Première version | Ce réseau |
|---|---|---|
| `Check` / `Nothing` | 71,6 % | **92,6 %** |
| `Checkmate` / `Check` / `Nothing` | 53,6 % | **84,9 %** |
| Avec les couleurs | 53,6 % | **84,9 %** |

**Par classe :**

| Classe | Précision | Rappel | F1 |
|---|---|---|---|
| `Nothing` | 0,91 | 0,91 | 0,91 |
| `Check White` | 0,82 | 0,80 | 0,81 |
| `Check Black` | 0,77 | 0,79 | 0,78 |
| `Checkmate White` | 0,81 | 0,84 | 0,83 |
| `Checkmate Black` | 0,81 | 0,81 | 0,81 |

- **Toutes les classes progressent,** surtout `Nothing`, dont le rappel passe de 43 % à 91 %.
- **Les couleurs ne sont presque jamais confondues :** 3 erreurs sur 41 798 échecs et mats.
- **La difficulté principale reste de séparer un échec d'un mat :** 2 859 échecs sont pris pour des mats, et 2 645 mats pour des échecs.

Sur nos 10 000 positions de test aléatoires, ce réseau fait aussi mieux que la première version : 74,1 % contre 71,8 % avec les couleurs.

**Sur les 5 positions de l'exemple du sujet,** il en trouve 3. Il prend le mat du fou pour `Check Black`, et la position avec seulement les deux rois pour `Check Black`, alors qu'il n'y a rien. La première version les trouvait toutes les 5. Ces 5 positions pèsent peu face aux 71 554 du jeu de test, mais elles montrent que des erreurs restent possibles sur des positions simples.

## Le fichier poussé

`my_torch_network.nn` pèse 6,7 Mo. Il contient le réseau, `769 → 256 → 256 → 64 → 5`, mais aussi le coût, l'optimiseur Adam avec sa mémoire, et les réglages d'entraînement. `my_torch_analyzer --train` peut donc continuer à l'entraîner.

[tests/test_pushed_network.py](../tests/test_pushed_network.py) vérifie :

- que c'est le seul réseau dont le nom commence par `my_torch_network` ;
- qu'il correspond à sa configuration ;
- qu'il peut encore être entraîné ;
- que le vrai exécutable répond à l'exemple du sujet avec un label valide par ligne.

## Les limites et la suite

- **Un écart d'environ 10 points** entre l'entraînement et la validation. La régularisation, l'arrêt de l'entraînement au bon moment et le dropout sont prévus pour le réduire.
- **Des positions inconnues.** Les fichiers du correcteur ne sont pas exactement le dataset fourni : sa précision peut différer de celle mesurée ici.
- **Le format aux niveaux simples.** Le réseau répond toujours avec les couleurs. Le premier essai suggère que le correcteur les accepte, mais ce n'est pas certain.
