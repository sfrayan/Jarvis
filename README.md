# Jarvis

Agent IA vocal **local-first** pour Windows 11 : écoute permanente, vision de l'écran, actions souris/clavier, voix FR naturelle. Stack 100 % locale par défaut, cloud optionnel.

> **État** : Itération 1 terminée (scaffolding uniquement). Aucune logique applicative — `main.py` arrivera en Itération 2. Consulter la roadmap en fin de document.

---

## Prérequis

| Composant | Minimum | Recommandé |
|---|---|---|
| OS | Windows 10/11 | Windows 11 |
| Python | 3.11 | 3.11.9+ |
| Docker Desktop | avec backend WSL2 | 4.28+ |
| GPU | CPU only possible (lent) | NVIDIA ≥ 8 Go VRAM |
| RAM | 16 Go | 32 Go |
| Disque libre | 30 Go | 60 Go+ |
| Micro / HP | n'importe lequel | casque USB |

**Ollama** peut tourner soit en natif (installé sur Windows), soit dans le conteneur fourni par `docker-compose.yml`. Ce projet utilise par défaut l'Ollama conteneurisé — voir [Pièges connus](#pièges-connus).

---

## Installation (première fois)

Toutes les commandes ci-dessous sont à exécuter en **PowerShell** depuis la racine du repo (`C:\Jarvis`).

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

Démarrer les services dockerisés (Ollama, ChromaDB, Piper TTS, n8n) :

```powershell
docker compose up -d
```

Télécharger les modèles Ollama **dans le conteneur** (important, voir pièges) :

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
```powershell
docker exec jarvis-ollama ollama pull nomic-embed-text
```

Total à télécharger : **~20 Go**. Vérifier ensuite :

```powershell
python check_env.py
```

Objectif : `9/10 OK · 1 WARN · 0 FAIL` (le WARN sur GPU CUDA disparaît si tu installes torch en variante CUDA — facultatif).

---

## Démarrage quotidien

```powershell
docker compose up -d
```

```powershell
.\.venv\Scripts\Activate.ps1
```

```powershell
python check_env.py
```

`python main.py` **sera disponible à partir de l'Itération 2**. Pour l'instant, seul le scaffolding est en place.

Arrêt des services (sans perte de données) :

```powershell
docker compose down
```

Reset complet (⚠ supprime les volumes, donc les modèles Ollama, la base ChromaDB et les workflows n8n) :

```powershell
docker compose down -v
```

---

## Architecture

```
C:\Jarvis\
├── main.py                    # Point d'entrée (Itération 2+)
├── check_env.py               # Diagnostic d'environnement
├── docker-compose.yml         # 4 services : ollama, chromadb, piper-tts, n8n
├── config/                    # YAML par défaut + schéma Pydantic
├── core/                      # Boucle OODA, state machine, event bus
├── ears/                      # VAD (Silero) + STT (faster-whisper)
├── brain/                     # Routeur intent, vision locale, mémoire RAG
├── hands/                     # Souris, clavier, capture d'écran
├── voice/                     # Client TTS Piper
├── safety/                    # Kill switch, allowlist, dry_run
├── integrations/              # n8n, v0, NotebookLM (optionnels)
├── observability/             # structlog, métriques Prometheus
└── tests/                     # unit + integration
```

`jarvis-core` tourne en **natif Windows**, hors Docker, pour accéder au bureau. Il joint les 4 conteneurs via `host.docker.internal`.

---

## Modes de sécurité

Défini par `safety.mode` dans `config/default.yaml` (override via `config/local.yaml` ou variable `JARVIS_SAFETY_MODE`).

| Mode | Comportement |
|---|---|
| `observe` | Écoute + transcrit + loggue. **Aucune action GUI exécutée.** |
| `dry_run` (**défaut**) | Calcule les actions, les loggue, n'exécute pas. |
| `assisted` | Exécute, sauf actions destructives → confirmation vocale. |
| `autonomous` | Exécute tout sauf destructif. |

Pour activer `autonomous`, les deux conditions suivantes doivent être réunies :
1. `safety.mode: autonomous` dans `config/local.yaml`.
2. `JARVIS_I_UNDERSTAND_THE_RISKS=true` dans `.env`.

---

## Kill switch

Le listener `pynput` (Itération 2) surveille trois déclencheurs **globaux**, actifs même quand Jarvis n'a pas le focus :

- **F12** pressé une fois
- **Échap** maintenu > 1 seconde
- **Souris poussée dans le coin haut-gauche** (pyautogui.FAILSAFE)

Effets immédiats : vidage des queues, fermeture des connexions HTTP, transition `EMERGENCY_STOP`, log niveau `CRITICAL`.

Reset : commande vocale « Jarvis, reprends » + confirmation explicite.

---

## Tests

Unitaires rapides (pas d'I/O externe, pas de conteneurs) :

```powershell
pytest tests/unit -m unit
```

Intégration (nécessite Docker up + Ollama joignable) :

```powershell
pytest tests/integration -m integration
```

Lint + types :

```powershell
ruff check .
```

```powershell
mypy .
```

---

## Pièges connus

### Ollama natif vs conteneurisé

Si tu avais Ollama installé **nativement** avant ce projet, ses modèles ne sont **pas** visibles depuis le conteneur `jarvis-ollama`. Le compose utilise un volume Docker séparé (`jarvis_ollama_models`).

**Solution** : `docker exec jarvis-ollama ollama pull <model>` pour chaque modèle (coûte ~20 Go de re-téléchargement mais archi reproductible).

**Alternative** : supprimer le service `ollama` du `docker-compose.yml` et garder `OLLAMA_BASE_URL=http://host.docker.internal:11434` pointant vers l'Ollama natif — économise bande passante et disque, au prix d'une archi mi-dockerisée.

### Contraintes GPU (6 Go VRAM typique)

Sur une RTX 2060 / 3060 / équivalent avec **6 Go VRAM**, `qwen2.5vl:7b` (~5.5 Go en Q4) + `faster-whisper medium` (~1.5 Go) ne tiennent **pas** simultanément sur le GPU. Itération 3+ routera : vision sur GPU, STT sur CPU, ou inverse selon l'état courant.

### Espace disque

Le stack pèse ~20 Go côté Ollama + quelques Go pour les autres conteneurs. Si tu gardes un Ollama natif en parallèle, tu doubles la consommation. `docker system prune -a --volumes` avant reset complet peut libérer beaucoup.

### Torch CPU vs CUDA

`pip install -r requirements.txt` installe torch en variante CPU (pulled transitif par `silero-vad`). Pour activer l'accélération GPU :

```powershell
pip install --force-reinstall torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

### Microphone bloqué par Windows

Paramètres → Confidentialité et sécurité → Microphone → autoriser les applications de bureau.

---

## Roadmap

| Itération | Contenu | Statut |
|---|---|---|
| **1** | Scaffolding : arbo, Docker, config, check_env | ✅ |
| **2** | Kill switch (priorité absolue) + boucle OODA + state machine | ☐ |
| **3** | Ears : VAD Silero + STT faster-whisper | ☐ |
| **4** | Brain : routeur qwen3 + vision qwen2.5vl | ☐ |
| **5** | Hands : actuators pyautogui + capture mss + scaling | ☐ |
| **6** | Voice : client Piper Wyoming + streaming audio | ☐ |
| **7** | Brain cloud fallback : Anthropic Computer Use | ☐ |
| **8** | Integrations : n8n client + RAG ChromaDB | ☐ |
| **9** | Observabilité : Prometheus + Grafana (profile monitoring) | ☐ |

Méthode de travail : lecture → roadmap → attente → patch. Un fichier majeur par livraison. Voir `CLAUDE.md`.

---

## Licence

Projet personnel. Aucune diffusion externe prévue.
