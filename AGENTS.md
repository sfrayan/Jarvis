# Projet Jarvis — Règles permanentes

## Méthode de travail (NON NÉGOCIABLE)

1. **LECTURE** — Lire tout le code existant avant d'écrire quoi que ce soit.
2. **ROADMAP** — Proposer un plan par itérations numérotées. AUCUN code applicatif.
3. **ATTENTE** — S'arrêter, demander validation explicite.
4. **PATCH** — Un fichier majeur à la fois, commenté, testable isolément.
5. **VÉRIFICATION** — Après chaque fichier, proposer la commande de test.

## Langue

- Réponses, commentaires, docs : **français**
- Noms de variables/fonctions/classes : anglais

## Stack imposée

- Python 3.11+, asyncio
- Ollama local (`qwen3:latest`, `qwen2.5-coder:7b`, `deepseek-coder:6.7b` déjà installés)
- Docker Compose classique (PAS de devcontainers)
- Windows 11 + Docker Desktop (WSL2 backend)
- Communication inter-services via `host.docker.internal`
- Pas de paid-API obligatoire. Cloud = fallback optionnel uniquement.

## Sécurité

- Kill switch via `pynput` en PRIORITÉ ABSOLUE, codé et testé avant la boucle OODA.
- Mode par défaut au premier lancement : `dry_run`.
- Pas de suppression de fichiers sans confirmation explicite.

## Qualité

- Type hints partout, `mypy --strict`
- Pydantic v2 pour configs et contrats JSON
- `structlog` (pas de `print`)
- `pytest` + `pytest-asyncio`
- `ruff` pour lint/format