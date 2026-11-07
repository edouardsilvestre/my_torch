# Le réseau pré-entraîné

Le sujet impose de pousser un réseau pré-entraîné dont le nom commence par `my_torch_network` : c'est lui que le correcteur évalue. Ce document explique comment [my_torch_network.nn](../my_torch_network.nn) a été obtenu, comment le refaire, et ce qu'il vaut.

## Le refaire

```
$> ./scripts/build_network.sh
```

[scripts/build_network.sh](../scripts/build_network.sh) enchaîne les commandes exactes qui ont produit le réseau :

1. **Générer 100 000 positions labellisées** avec la graine 0, comme l'explique [la génération du dataset](dataset.md). Cela prend environ 50 minutes sur les 4 cœurs de notre machine. Le fichier `datasets/all_100000_0.txt` a une empreinte SHA-256 qui commence par `1a8bcac7b77a247d`. S'il existe déjà, il est réutilisé.
2. **Le découper** en 79 999 positions d'entraînement, 10 001 de validation et 10 000 de test.
3. **Créer un réseau vierge** depuis [configs/my_torch_network.conf](../configs/my_torch_network.conf), avec `my_torch_generator`.
4. **L'entraîner** sur le jeu d'entraînement avec `my_torch_analyzer --train`, en 5 epochs.
5. **L'évaluer** sur la validation, puis sur le test.

Une fois le dataset généré, les étapes 2 à 5 prennent environ 80 secondes. Toutes les graines sont fixées : relancer le script redonne **le même réseau, au bit près**. Le sujet exige de garder tous les scripts d'entraînement, pour qu'on puisse refaire le réseau de la même façon. Les datasets ne sont pas poussés, ce que le sujet autorise, mais ils se régénèrent à l'identique.

## Le choix de la configuration

Quatre configurations ont été entraînées 20 epochs, puis comparées sur le **jeu de validation**, avec [scripts/compare_configs.py](../scripts/compare_configs.py). Elles ont toutes 5 sorties, ReLU, l'initialisation He, Adam et des batchs de 64. Elles ne diffèrent que sur un point à la fois :

| Configuration | Couches cachées | Learning rate | Validation 5 classes | Validation 2 classes | Meilleure epoch | Durée |
|---|---|---|---|---|---|---|
| [c1_256_64](../configs/candidates/c1_256_64.conf) | 256, 64 | 0,001 | 72,3 % | 82,8 % | 5 (72,8 %) | 3,5 min |
| [c2_512_128](../configs/candidates/c2_512_128.conf) | 512, 128 | 0,001 | 72,0 % | 82,6 % | 15 (72,7 %) | 9,2 min |
| [c3_256_256_64](../configs/candidates/c3_256_256_64.conf) | 256, 256, 64 | 0,001 | 71,9 % | 82,4 % | 5 (72,9 %) | 4,7 min |
| [c4_256_64_slow](../configs/candidates/c4_256_64_slow.conf) | 256, 64 | 0,0003 | 71,1 % | 82,2 % | 8 (72,3 %) | 3,5 min |

La comparaison se refait avec :

```
$> python3 -m scripts.compare_configs --train datasets/all_100000_0_train.txt \
       --validation datasets/all_100000_0_validation.txt configs/candidates/*.conf
```

**Ce qu'on en lit :**

- **Les quatre configurations se valent.** Moins d'un point les sépare, ce qui est de l'ordre du hasard sur 10 000 positions. Un réseau plus large ou plus profond n'apporte rien ici.
- **Le réseau surapprend.** Sur l'entraînement, la précision atteint 99 % après 20 epochs, contre 72 % sur la validation. La validation plafonne vers l'epoch 5, puis stagne ou baisse légèrement : le réseau apprend ses exemples par cœur au lieu de généraliser.
- **La quantité de données compte beaucoup plus que l'architecture.** Avec 1 200 positions d'entraînement, le même genre de réseau n'atteignait que 51 %, comme le montre [l'évaluation](evaluation.md).

**Le choix : `c1_256_64`, avec 5 epochs.** C'est la configuration la plus simple et la plus rapide, et 5 epochs correspond à son meilleur point sur la validation. Ce choix ne s'appuie que sur la validation : **le jeu de test n'a servi qu'une fois**, pour la mesure ci-dessous.

## Ce qu'il vaut

Sur le **jeu de test**, 10 000 positions que le réseau n'a jamais vues :

| Niveau du correcteur | Accuracy | Moyenne F1 |
|---|---|---|
| `Check` / `Nothing` | **82,8 %** | 0,80 |
| `Checkmate` / `Check` / `Nothing` | **71,8 %** | 0,72 |
| Avec les couleurs | **71,8 %** | 0,71 |

Ces niveaux sont obtenus en regroupant les prédictions du réseau à 5 sorties, comme l'explique [l'évaluation](evaluation.md). La validation donne presque les mêmes valeurs (72,8 % et 82,8 %).

**Par classe, sur le test :**

| Classe | Précision | Rappel | F1 |
|---|---|---|---|
| `Nothing` | 0,77 | 0,69 | 0,73 |
| `Check White` | 0,62 | 0,63 | 0,63 |
| `Check Black` | 0,62 | 0,61 | 0,62 |
| `Checkmate White` | 0,77 | 0,84 | 0,80 |
| `Checkmate Black` | 0,76 | 0,84 | 0,80 |

- **Les mats sont bien reconnus,** avec 84 % de rappel.
- **Les couleurs ne sont presque jamais confondues :** 3 erreurs sur les 6 667 échecs et mats.
- **L'échec simple est le plus difficile.** Le réseau le confond avec `Nothing`, ce qui fait manquer des échecs et en inventer, et avec le mat, quand le roi a encore une parade.

Sur les 5 positions de l'exemple du sujet, le réseau donne exactement la réponse du réseau « full » du sujet : `Checkmate Black`, `Nothing`, `Nothing`, `Check White`, `Nothing`.

## Le fichier poussé

`my_torch_network.nn` pèse 5,1 Mo. Il contient le réseau, `769 → 256 → 64 → 5`, mais aussi le coût, l'optimiseur Adam avec sa mémoire, et les réglages d'entraînement. `my_torch_analyzer --train` peut donc continuer à l'entraîner. Sans la mémoire d'Adam, il ne pèserait que 1,7 Mo.

[tests/test_pushed_network.py](../tests/test_pushed_network.py) vérifie que c'est le seul réseau dont le nom commence par `my_torch_network`, qu'il correspond à sa configuration, qu'il peut encore être entraîné, et que le vrai exécutable prédit l'exemple du sujet.

## Les limites et la suite

- **Le surapprentissage.** C'est la limite principale. La régularisation, l'arrêt de l'entraînement au bon moment et le dropout sont prévus pour la réduire.
- **Des positions de parties aléatoires.** Les positions du correcteur ressemblent plutôt à de vraies parties. La précision sur ses fichiers peut donc être différente de celle mesurée ici.
- **Les couleurs aux niveaux plus simples.** Ce réseau répond toujours avec les couleurs, par exemple `Check White`. On ne sait pas encore si le correcteur l'accepte quand il attend seulement `Check`. Le premier essai du correcteur le dira.
- **Une première version.** Ce réseau sera réentraîné quand les techniques d'optimisation et la recherche automatique des réglages seront prêtes.
