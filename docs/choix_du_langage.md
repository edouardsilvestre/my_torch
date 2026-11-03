# Choix du langage : Python + NumPy

Le sujet autorise n'importe quel langage qui fonctionne sur l'environnement d'évaluation, mais interdit les bibliothèques de réseaux de neurones (PyTorch, TensorFlow, ...). Nous avons choisi **Python 3 avec NumPy** plutôt que C++. Ce document explique pourquoi.

## NumPy n'est pas une bibliothèque de réseaux de neurones

NumPy fournit uniquement des tableaux multidimensionnels et des opérations d'algèbre linéaire. Il n'offre ni couches, ni différentiation automatique, ni fonctions de coût, ni optimiseurs, ni sauvegarde de modèles. Tout ce que le sujet demande est donc implémenté à la main dans le module `my_torch` :

- forward pass ;
- rétropropagation ;
- initialisation des poids ;
- optimiseurs ;
- régularisation ;
- sérialisation.

NumPy joue le même rôle qu'une bibliothèque de matrices comme Eigen en C++.

## Les performances viennent de l'algèbre linéaire, pas du langage

L'entraînement d'un réseau dense passe l'essentiel de son temps dans des produits matriciels (`W @ X`). NumPy délègue ces calculs à une implémentation BLAS (OpenBLAS ou MKL), écrite en C, Fortran et assembleur, qui utilise :

- les instructions vectorielles SIMD ;
- un découpage en blocs adapté au cache ;
- plusieurs cœurs.

Un produit matriciel écrit à la main en C++ est en général **plus lent** que celui de NumPy. Pour faire mieux en C++, il faudrait utiliser Eigen ou BLAS, c'est-à-dire le même type de code que celui utilisé par NumPy.

Le surcoût réel de Python vient des boucles exécutées par l'interpréteur. Nous le limitons en **vectorisant les calculs sur des mini-batchs entiers** : une seule opération matricielle traite des centaines d'échiquiers à la fois, au lieu de boucler sur chaque exemple ou chaque neurone.

## Le sujet valorise l'expérimentation

Une grande partie de l'évaluation porte sur :

- les benchmarks avec preuves visuelles ;
- l'optimisation des hyperparamètres et les métaheuristiques ;
- la comparaison de plusieurs courbes d'apprentissage ;
- la reproductibilité des entraînements.

Python permet d'itérer vite sur ces expériences. Les scripts d'entraînement, de génération de dataset et de génération des graphiques (matplotlib) restent dans le même écosystème que le module. Le temps gagné sur le développement est réinvesti dans l'exploration des configurations, et c'est là que se joue la qualité du réseau final.

## Un code proche des formules mathématiques

Avec NumPy, le code reste très proche des équations : `z = W @ x + b`, `dW = delta @ x.T`. Ce code est donc plus simple :

- à relire ;
- à vérifier par gradient checking (comparaison avec les différences finies) ;
- à justifier en soutenance.

Il n'y a pas non plus de gestion manuelle de la mémoire, une source de bugs fréquente en C++.

## Limites assumées

| Limite | Impact | Réponse |
|---|---|---|
| Surcoût de l'interpréteur sur les boucles | Lenteur si l'on boucle exemple par exemple | Vectorisation par mini-batch |
| GIL : le multithreading Python est limité | Parallélisme limité en Python pur | BLAS s'exécute hors du GIL et utilise déjà plusieurs cœurs ; `multiprocessing` pour le bonus de parallélisation |
| Temps de démarrage (import de NumPy) | Environ 0,1 s par exécution | Négligeable face au temps de prédiction ou d'entraînement |
| Dépendance à NumPy | NumPy doit être présent sur l'environnement d'évaluation | Dépendance unique, déclarée dans `requirements.txt` |

Le C++ aurait été pertinent pour des opérations très petites et très nombreuses, où le surcoût de l'interpréteur dominerait. C'est par exemple le cas de l'évaluation de millions de positions dans un moteur d'échecs complet (bonus). Ce n'est pas le cas de l'entraînement par mini-batch demandé ici.

## Vérification

Ces affirmations seront mesurées dans les benchmarks : temps d'un produit matriciel NumPy comparé à une implémentation naïve, et temps d'entraînement par epoch selon la taille du batch.
