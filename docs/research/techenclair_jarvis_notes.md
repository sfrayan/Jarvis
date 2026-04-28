# Notes de recherche - TechEnClair J.A.R.V.I.S

Date de lecture : 2026-04-28

## Sources inspectees

- Page publique J.A.R.V.I.S : https://techenclair.fr/pages/jarvis.html
- Depot GitHub TechEnClair : https://github.com/mirasandalucia/TechEnClair
- Dossier public `scripts/` : https://github.com/mirasandalucia/TechEnClair/tree/main/scripts

## Constats

- Le depot GitHub public `mirasandalucia/TechEnClair` n'est pas le code source de
  son assistant Jarvis. Il sert surtout de bibliotheque de ressources Home
  Assistant : scripts YAML, cartes, themes et assets.
- La page J.A.R.V.I.S annonce un assistant vocal pour PC et domotique, avec un
  telechargement deverrouille par captcha, attente gratuite ou don.
- Le code source complet n'est pas expose dans le depot public inspecte.
  Aucune tentative de contournement du captcha ou du mecanisme de telechargement
  n'a ete faite.
- La version presentee met l'accent sur l'installation facile : installeur
  Windows, script Linux, configuration des cles API et personnalisation du
  prenom.
- Le positionnement est tres proche de notre objectif fonctionnel, mais avec une
  approche plus "package utilisateur final" que "architecture locale testee".

## Idees utiles pour Jarvis

- Cataloguer les commandes par domaines avant d'executer quoi que ce soit :
  `system`, `apps`, `folders`, `media`, `home_assistant`, `vision`, `memory`,
  `web_search`, `google_workspace`.
- Ajouter une intention composee pour les routines, par exemple "mode travail",
  qui ouvre plusieurs applications ou prepare un contexte.
- Prevoir une configuration d'identite utilisateur au demarrage, mais dans un
  fichier local type `config/local.yaml`, pas en dur dans le code.
- Garder les commandes Home Assistant pour une iteration dediee, avec client
  HTTP, allowlist, `dry_run`, puis confirmation pour actions sensibles.
- Utiliser les exemples Home Assistant publics comme inspiration de scenarios,
  pas comme dependance directe.
- Conserver l'idee d'une vision ecran capable de repondre a "analyse mon ecran",
  "clique sur X" et "ecris Y", mais avec nos contrats Pydantic et notre kill
  switch.
- Reporter l'interface mobile/web et l'auto-detection IP apres le coeur
  OODA/vision/hands/voice.

## Idees a ne pas reprendre maintenant

- Ne pas rendre les API cloud obligatoires. Notre base reste local-first, cloud
  seulement en fallback optionnel.
- Ne pas executer directement des actions destructives comme shutdown, delete,
  taskkill ou modifications cloud sans garde-fous explicites.
- Ne pas ajouter d'installeur Windows avant que le coeur soit stable et teste.
- Ne pas melanger logique applicative et configuration personnelle.
- Ne pas dependre d'un outil externe pour corriger les erreurs au lieu de tests
  reproductibles.

## Impact propose sur la roadmap

1. Iteration 4C - Router enrichi : transformer `chat/gui/unknown` en intentions
   internes plus detaillees, tout en gardant la sortie publique compatible.
2. Iteration 5A - Screenshot local : capturer l'ecran avec `mss`, encoder
   l'image et alimenter `LocalVisionClient`.
3. Iteration 5B - Hands en dry-run : convertir `VisionDecision.actions` en
   actions pyautogui journalisees, sans clic reel par defaut.
4. Iteration 8 - Home Assistant : ajouter les commandes domotiques inspirees des
   scenarios TechEnClair, avec allowlist et confirmation.
5. Iteration future - Onboarding : prenom utilisateur, diagnostic IP locale,
   verification des modeles et eventuelle interface web.

## Critere de reprise

Cette note sert uniquement de reference produit. Elle ne justifie aucun patch
applicatif sans repasser par la sequence lecture, plan, validation, puis fichier
majeur testable.
