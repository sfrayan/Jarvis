# Jarvis

Agent IA vocal **local-first** pour Windows 11.

Jarvis vise un assistant personnel utile au quotidien : il ecoute, route les
intentions, dialogue quand la demande est incomplete, planifie prudemment,
declenche les actions locales ou navigateur seulement quand elles sont claires,
et garde le controle utilisateur au centre.

> **Etat reel du projet** : le README etait en retard. Jarvis n'est plus un
> simple scaffolding : `main.py` monte deja config, kill switch, EventBus,
> StateMachine, boucle OODA, Ears, Brain, Dialogue, Hands et Voice.

---

## Capacites Actuelles

- **Bootstrap applicatif** : `main.py` charge la config, configure les logs,
  demarre le kill switch, puis lance les services.
- **Ears** : pipeline audio avec VAD Silero et STT faster-whisper.
- **Brain** : routeur d'intentions avec heuristiques locales et fallback Ollama
  en JSON strict.
- **Dialogue** : `DialogueManager` + sessions de tache en RAM pour clarifier,
  planifier et poursuivre une demande.
- **Demo devoir** : "Jarvis, aide-moi a faire un devoir" declenche clarification,
  plan, brouillon en RAM ou recherche web sure.
- **Hands local** : planification prudente d'apps, dossiers, media et systeme,
  via inventaire local readonly et registre de capacites.
- **Hands navigateur** : recherches Google/YouTube via Chrome sans vision quand
  l'intention `web_search` est reconnue.
- **Vision** : capture ecran + client vision local seulement pour le domaine
  `vision`, pas pour les actions simples.
- **Voice** : feedback via `AssistantUtterance`, Piper si configure, fallback log.
- **Securite** : modes `observe`, `dry_run`, `assisted`, `autonomous`, avec kill
  switch prioritaire et blocage des actions sensibles/destructives.

---

## Prerequis

| Composant | Minimum | Recommande |
|---|---|---|
| OS | Windows 10/11 | Windows 11 |
| Python | 3.11 | 3.11.9+ |
| Docker Desktop | backend WSL2 | 4.28+ |
| RAM | 16 Go | 32 Go |
| Disque libre | 30 Go | 60 Go+ |
| Micro / HP | n'importe lequel | casque USB |
| Ollama | local ou Docker | modele local chaud |

Jarvis tourne en natif Windows pour pouvoir acceder au bureau. Les services
Docker restent optionnels selon la configuration : Ollama, Piper, ChromaDB, n8n.

---

## Installation

Toutes les commandes sont a executer en PowerShell depuis `C:\Jarvis`.

```powershell
python -m venv .venv
```

```powershell
.\.venv\Scripts\Activate.ps1
```

```powershell
pip install --upgrade pip
```

```powershell
pip install -r requirements.txt
```

```powershell
copy .env.example .env
```

Pour les services Docker :

```powershell
docker compose up -d
```

Modeles Ollama utiles :

```powershell
docker exec jarvis-ollama ollama pull qwen3:latest
```

```powershell
docker exec jarvis-ollama ollama pull qwen2.5-coder:7b
```

```powershell
docker exec jarvis-ollama ollama pull deepseek-coder:6.7b
```

```powershell
docker exec jarvis-ollama ollama pull qwen2.5vl:7b
```

Verifier l'environnement :

```powershell
python check_env.py
```

---

## Demarrage

```powershell
docker compose up -d
```

```powershell
.\.venv\Scripts\Activate.ps1
```

```powershell
python main.py
```

Arret des services Docker :

```powershell
docker compose down
```

Reset complet des volumes Docker :

```powershell
docker compose down -v
```

---

## Configuration

La configuration par defaut est dans `config/default.yaml`.

Pour les reglages locaux, creer :

```powershell
copy config\local.yaml.example config\local.yaml
```

`config/local.yaml` est ignore par Git. Le loader fusionne un `local.yaml` situe
a cote du fichier `default_path` utilise, ce qui evite que les tests temporaires
soient pollues par la config locale du repo.

Il est possible de choisir un autre fichier de config :

```powershell
$env:JARVIS_CONFIG_PATH = "config/default.yaml"
```

---

## Modes De Securite

| Mode | Comportement |
|---|---|
| `observe` | Observe, route et journalise. Aucune execution reelle. |
| `dry_run` | Planifie et publie les rapports, sans executer. Mode par defaut. |
| `assisted` | Execute les actions sures, demande confirmation pour le sensible. |
| `autonomous` | Execute seulement les actions sures et non destructives. |

Regles permanentes :

- kill switch prioritaire ;
- pas de suppression de fichiers sans confirmation explicite ;
- pas de modification de services Windows sans confirmation explicite ;
- pas d'admin sans explication et confirmation ;
- cloud uniquement en fallback optionnel.

---

## Kill Switch

Le kill switch est demarre avant la boucle OODA.

Declencheurs prevus :

- `F12` ;
- `Echap` maintenu plus d'une seconde ;
- coin haut-gauche via `pyautogui.FAILSAFE`.

Effet attendu : transition vers `EMERGENCY_STOP`, arret propre des services et
priorite absolue sur le reste du pipeline.

---

## Architecture

```text
C:\Jarvis\
|-- main.py                    # Bootstrap runtime
|-- check_env.py               # Diagnostic environnement
|-- docker-compose.yml         # Services locaux optionnels
|-- config\                    # YAML + schemas Pydantic
|-- core\                      # EventBus, StateMachine, OODA loop
|-- ears\                      # AudioStream, VAD, STT, EarsService
|-- brain\                     # Router, Dialogue, sessions, vision locale
|-- hands\                     # Inventaire, capacites, local/browser/vision
|-- voice\                     # Feedback vocal/log, Piper
|-- safety\                    # Kill switch
|-- observability\             # structlog
|-- tests\                     # Tests unitaires et integration
```

Flux principal :

```text
Audio
-> EarsService
-> Transcription
-> BrainService
-> IntentRouter
-> DialogueService
-> HandsPipelineService ou AssistantUtterance
-> VoiceFeedbackService
```

Le `DialogueManager` reste volontairement separe du routeur : le routeur classe
l'intention, le dialogue decide s'il faut clarifier, planifier, produire un
brouillon ou relayer vers Hands.

---

## Exemples

### Devoir

Utilisateur :

```text
Jarvis j'ai un devoir a faire
```

Jarvis demande la consigne, la matiere, le niveau et la date limite. Quand ces
informations sont fournies, il propose un plan puis peut :

- generer un brouillon en RAM via `AssistantDraft` ;
- relayer une recherche Google/YouTube sure vers le planner navigateur.

### Recherche YouTube

```text
cherche une video YouTube sur les chats
```

Le domaine `web_search` passe directement vers `BrowserActionPlanner`, sans
screenshot et sans vision.

### Action locale

```text
ouvre Chrome
```

Jarvis consulte les capacites locales, planifie l'ouverture selon le mode de
securite, puis publie un feedback clair.

### Vision

```text
regarde l'ecran et clique sur enregistrer
```

Le domaine `vision` declenche capture d'ecran, analyse locale et execution
dry-run/controlee selon les garde-fous.

---

## Tests

Unitaires :

```powershell
pytest tests/unit -m unit
```

Etat connu apres 5L :

```text
513 passed
```

Integration :

```powershell
pytest tests/integration -m integration
```

Lint :

```powershell
ruff check .
```

Types :

```powershell
mypy .
```

---

## Roadmap Courte

| Iteration | Contenu | Statut |
|---|---|---|
| 1 | Scaffolding, Docker, config, check_env | Fait |
| 2-5G | Kill switch, OODA, Ears, Brain, Hands, Voice | Fait |
| 5H | Dialogue interactif + sessions de tache | Fait |
| 5I | Feedback naturel et contextuel | Fait |
| 5J | Routines sures de travail/code/recherche | Fait |
| 5K | Demo devoir bout en bout | Fait |
| 5L | Brouillons en RAM via `AssistantDraft` | Fait |
| 5M | README synchronise avec l'etat reel | Fait |

Prochaines pistes utiles :

- sauvegarde optionnelle de brouillons, en `dry_run` par defaut ;
- routines plus completes : devoir, code, recherche, focus ;
- confirmations explicites pour actions sensibles ;
- verification post-action par vision uniquement quand necessaire ;
- memoire temporaire puis preferences utilisateur avec consentement.

---

## Pieges Connus

### Ollama natif vs Docker

Les modeles d'un Ollama natif ne sont pas visibles depuis le conteneur
`jarvis-ollama`. Utiliser `docker exec jarvis-ollama ollama pull <model>` si le
compose gere Ollama, ou configurer `OLLAMA_BASE_URL` vers l'Ollama natif.

### GPU et modeles

Sur une carte avec 6 Go de VRAM, vision locale et STT moyen peuvent ne pas tenir
simultanement sur GPU. Garder un chemin CPU ou alterner les charges.

### Config locale

`config/local.yaml` peut activer `assisted` ou Piper. Les tests unitaires doivent
rester isoles de cette config locale.

### Piper

Si `tts.backend: piper` est configure mais que Piper n'est pas disponible, le
fallback log peut prendre le relais si `fallback_to_log: true`.

---

## Methode De Travail

Le projet suit la discipline de `AGENTS.md` et `CLAUDE.md` :

1. lecture du code existant ;
2. roadmap courte ;
3. validation explicite ;
4. patch petit et testable ;
5. verification avec commandes de test.

Les reponses, commentaires et docs restent en francais. Les noms de classes,
fonctions et variables restent en anglais.

---

## Licence

Projet personnel. Aucune diffusion externe prevue.
