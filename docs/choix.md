# Les choix de conception et leurs résultats

Ce document rassemble tous les choix faits pour le réseau d'échecs, avec pour chacun ce qui le justifie : une mesure, un benchmark ou un raisonnement. Chaque partie renvoie au document qui détaille la mesure.

## Le parcours en un tableau

Précision sur le **jeu de test** du dataset fourni, 71 554 positions jamais vues, à chaque étape :

| Étape | `Check` / `Nothing` | `Checkmate` / `Check` / `Nothing` | Avec les couleurs | Document |
|---|---|---|---|---|
| Entraîné sur nos parties aléatoires | 71,6 % | 53,6 % | 53,6 % | [dataset.md](dataset.md) |
| Entraîné sur le dataset fourni, 10 epochs | 92,6 % | 84,9 % | 84,9 % | [reseau_final.md](reseau_final.md) |
| + régularisation L2 et early stopping | 95,0 % | 88,9 % | 88,9 % | [regularisation.md](regularisation.md) |
| + learning rate planifié : **le réseau poussé** | **95,7 %** | **90,0 %** | **90,0 %** | [learning_rate.md](learning_rate.md) |

Chaque étape n'a été retenue qu'après avoir amélioré la **validation**. Le jeu de test n'a servi qu'à ces mesures finales, jamais à choisir.

## Le module

Le module `my_torch` est générique : il ne sait rien des échecs, et ses tests l'entraînent aussi sur XOR et sur une régression. [L'architecture du module](architecture.md) détaille son découpage. Les choix principaux :

| Choix | Raison |
|---|---|
| Python et NumPy | Les calculs sont des produits de matrices confiés à BLAS : NumPy est environ 3 600 fois plus rapide qu'une triple boucle Python. Voir [le choix du langage](choix_du_langage.md) et [les benchmarks](benchmarks.md#la-vitesse-de-numpy) |
| Batchs en lignes, poids `(entrées, neurones)` | Une couche calcule tout un batch en un seul produit, `X @ W` |
| Cross-entropy sur scores bruts, dernière couche `identity` | Le gradient se simplifie en `probabilités - cibles`, plus stable. Voir [la backpropagation](backpropagation.md) |
| Backpropagation et optimiseur séparés | Changer d'optimiseur, ajouter la L2 ou un planning ne touche pas au réseau |
| Toutes les graines fixées | Le même entraînement redonne les mêmes poids : le réseau poussé se refait exactement, et chaque mesure se reproduit |
| Fichier `.nn` en NumPy et JSON, sans `pickle` | Un fichier reçu ne peut pas exécuter de code, et il garde la mémoire de l'optimiseur pour reprendre un entraînement. Voir [le format .nn](format_nn.md) |
| Fichiers de configuration en TOML, tout vérifié | Un réseau se décrit sans code, et une faute de frappe donne une erreur au lieu d'une valeur par défaut. Voir [la configuration](configuration.md) |

## L'encodage

Une position devient **769 entrées** : 12 plans de 64 cases, un par type de pièce et par couleur, plus une entrée pour le trait. [L'encodage](encodage.md) détaille ces choix :

- **Une entrée par pièce et par case,** plutôt qu'une valeur par case : le réseau n'a pas à apprendre qu'un 5 n'est pas « entre » un fou et une dame.
- **Le trait est indispensable :** seul le camp qui doit jouer peut être en échec ou mat.
- **Aucun calcul d'échecs dans l'encodage,** comme les cases attaquées : ce serait détecter l'échec à la place du réseau, alors que le sujet impose l'apprentissage supervisé.
- **Ni roques, ni compteurs :** ils ne changent jamais l'état d'échec.

## Les données

Le réseau est entraîné sur **le dataset fourni avec le sujet,** 723 019 positions de vraies parties, et non sur nos positions tirées de parties aléatoires. Le premier réseau, entraîné sur 80 000 positions aléatoires, ne faisait que 53,6 % sur les positions du sujet : il voyait des échecs dans plus de la moitié des positions `Nothing`. Le problème venait des données, pas du réseau. Nos propres règles des échecs donnent exactement les labels du dataset fourni sur un échantillon de 18 000 positions, ce qui vérifie aussi notre sens de la couleur. Voir [les données](dataset.md).

Le dataset est découpé une fois, avec une graine, en **80 % d'entraînement, 10 % de validation et 10 % de test,** après avoir retiré 7 460 positions en double. La répartition des labels est gardée telle quelle.

## Le réseau final et ses hyperparamètres

| Hyperparamètre | Valeur | Justification |
|---|---|---|
| Couches cachées | `256 → 256 → 64` | La meilleure de trois architectures comparées sur tout le dataset ([reseau_final.md](reseau_final.md#le-choix-de-la-configuration)). Les petits réseaux gagnent sur 60 000 positions, mais plafonnent à 81,5 % sur tout le dataset ([recherche.md](recherche.md#le-classement-tient-il-avec-toutes-les-données-)) |
| Sorties | 5, `identity` | Une sortie par label, couleurs comprises. Les niveaux plus simples s'en déduisent : un mat compte comme un échec pour `Check` / `Nothing` |
| Activation | `relu` | À égalité avec `leaky_relu` et `tanh` dans le benchmark, mais `tanh` est nettement moins bonne dans la recherche, et `sigmoid` perd 7 points ([benchmarks.md](benchmarks.md#les-activations)) |
| Initialisation | `he` | À égalité sur 3 couches, mais la seule qui garde le signal stable à travers 10 couches ReLU ([benchmarks.md](benchmarks.md#la-stabilité-du-signal-selon-linitialisation)) |
| Optimiseur | Adam, learning rate 0,001 | Les optimiseurs adaptatifs gagnent 3 points sur SGD et Momentum ([benchmarks.md](benchmarks.md#les-optimiseurs)) |
| Taille des batchs | 64 | Deux fois plus rapide que 32 pour 0,4 point de moins ([benchmarks.md](benchmarks.md#la-taille-des-batchs)) |
| Régularisation L2 | 1e-4 | +3,8 points de validation ; 1e-3 est trop forte ([regularisation.md](regularisation.md#ce-que-ça-donne-sur-le-réseau-déchecs)) |
| Dropout | aucun | Réduit l'écart entre l'entraînement et la validation, mais fait baisser la validation ([dropout.md](dropout.md#ce-que-ça-donne-sur-le-réseau-déchecs)) |
| Learning rate | divisé par 2 avant les epochs 17 et 19 | Le planning plateau gagne 1 point de validation ; `multistep` en reproduit les paliers pour `--train` ([learning_rate.md](learning_rate.md#ce-que-ça-donne-sur-le-réseau-déchecs)) |
| Epochs | 19 | L'epoch gardée par l'early stopping sur `val_accuracy` |

## La lutte contre l'overfitting

| Outil | Effet mesuré sur la validation | Retenu |
|---|---|---|
| Plus de données : le dataset fourni au lieu de 80 000 positions | Écart entraînement / validation de 27 à environ 10 points | Oui |
| Early stopping sur `val_accuracy` | Garde la meilleure epoch au lieu de la dernière ; sans L2, s'arrête à l'epoch 13 et garde l'epoch 10 | Oui, pour choisir le nombre d'epochs |
| Régularisation L2, 1e-4 | 85,3 % → **89,1 %**, écart de 10,5 à 3,6 points | Oui |
| Régularisation L2, 1e-3 | 80,6 % : underfitting | Non |
| Dropout 0,1 et 0,2 | 88,5 % et 87,5 %, écart de 3,0 et 2,6 points | Non |
| MC dropout, 20 passes | Aucun gain de précision, mais un écart-type qui signale les positions incertaines | Non, pour `--predict` |
| Learning rate planifié, plateau | 89,1 % → **90,1 %** | Oui |

**Surveiller `val_accuracy` plutôt que `val_loss`** a été décisif : sans L2, le coût de validation remonte dès l'epoch 5, alors que l'accuracy progresse jusqu'à l'epoch 10. C'est l'accuracy que mesure le correcteur.

**Le learning rate planifié creuse l'écart,** de 3,6 à 6,6 points, tout en améliorant la validation : des pas plus petits font mieux apprendre l'entraînement, et la validation suit, moins vite. L'early stopping reste donc indispensable.

## Les limites et les pistes

**Ce qui limite le réseau :**

- **Séparer un échec d'un mat reste difficile :** sur le test, 2 413 échecs sont pris pour des mats, et 1 681 mats pour des échecs. Savoir qu'un échec est un mat demande de vérifier qu'aucun coup ne le pare, un raisonnement sur plusieurs coups qu'un réseau dense apprend mal. Les couleurs, elles, ne sont presque jamais confondues : une seule erreur sur 41 798 échecs et mats.
- **La prise en passant n'est pas encodée :** elle peut, très rarement, être la seule parade à un mat.
- **Les positions du correcteur** ne sont pas exactement celles du dataset fourni : sa précision peut différer de la nôtre.

**Ce qui pourrait l'améliorer, sans avoir été mesuré :**

- **Retourner le plateau selon le trait,** pour que le réseau n'ait qu'un camp à apprendre. Il faudrait alors traiter à part les positions où c'est l'autre camp qui est en échec.
- **Encoder la case de prise en passant.**
- **Un réseau plus large ou plus profond, entraîné plus longtemps :** la recherche d'hyperparamètres n'a pu explorer qu'une part des données, ce qui favorise les petits réseaux. Une recherche sur tout le dataset demanderait plusieurs jours de calcul.
- **Des couches convolutives,** qui verraient les lignes, colonnes et diagonales de l'échiquier ; elles sortent du cadre d'un réseau dense.
- **AdamW,** qui applique la régularisation à part du pas d'Adam, au lieu de la régularisation L2 couplée.

**Ce que le module ne fait pas encore :**

- **`--train` ne sauvegarde qu'à la fin :** un entraînement arrêté avec Ctrl+C est perdu. Une sauvegarde après chaque epoch le rendrait reprenable.
- **`train` mesure l'entraînement et la validation à chaque epoch,** sur toutes les positions : environ 5 % de temps en plus par epoch sur le réseau d'échecs. Une option pour mesurer moins souvent n'a pas été nécessaire.
- **Le parallélisme** se limite à celui de BLAS : les entraînements des benchmarks et de la recherche tournaient en parallèle dans des processus séparés, à la main.
