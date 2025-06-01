.PHONY: install test lint format clean build publish dev docker-build docker-run

# Variables
POETRY := poetry
PYTHON := python
PROJECT := colpali_server

# Installation
install:
	$(POETRY) install
	@echo "✅ Installation terminée"

# Tests
test:
	$(POETRY) run pytest tests/ -v
	$(POETRY) run python tests/complete_test.py

# Qualité et formatage du code + détection d'erreurs
format:
	$(POETRY) run black --line-length 79 src/
	$(POETRY) run isort src/
	$(POETRY) run black --line-length 79 src/
	$(POETRY) run isort src/
	ruff format src/
	ruff check src/ --ignore D107 --fix --unsafe-fixes
	ruff format tests/
	ruff check tests/ --ignore D107 --fix --unsafe-fixes
	$(POETRY) run mypy src/
	$(POETRY) run mypy tests/
	
# Nettoyage
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

# Développement
dev:
	@if [ -f ".env" ]; then \
		set -o allexport && source .env && set +o allexport && \
		export ES_HOST=localhost:9200 && \
		export ES_USER=elastic && \
		export ES_PASSWORD=$ELASTIC_PWD && \
		export no_proxy="localhost,127.0.0.1" && \
		$(POETRY) run python -m $(PROJECT).cli; \
	else \
		echo "❌ Fichier .env introuvable"; \
		exit 1; \
	fi

dev-debug:
	@if [ -f ".env" ]; then \
		set -o allexport && source .env && set +o allexport && \
		export ES_HOST=localhost:9200 && \
		export ES_USER=elastic && \
		export ES_PASSWORD=$ELASTIC_PWD && \
		export no_proxy="localhost,127.0.0.1" && \
		$(POETRY) run python -m $(PROJECT).cli --log-level DEBUG; \
	else \
		echo "❌ Fichier .env introuvable"; \
		exit 1; \
	fi

# Elasticsearch (utilise la configuration existante)
es-check:
	@echo "Vérification d'Elasticsearch..."
	@if [ -f ".env" ]; then \
		set -o allexport && source .env && set +o allexport && \
		if curl -k -s -u "elastic:$ELASTIC_PWD" "https://localhost:9200" > /dev/null; then \
			echo "✅ Elasticsearch est accessible sur https://localhost:9200"; \
			curl -k -u "elastic:$ELASTIC_PWD" "https://localhost:9200" | jq '.version.number' || true; \
		else \
			echo "❌ Elasticsearch n'est pas accessible"; \
			echo "Assurez-vous qu'Elasticsearch est en cours d'exécution"; \
			exit 1; \
		fi; \
	else \
		echo "❌ Fichier .env introuvable"; \
		echo "Créez un fichier .env avec ELASTIC_PWD=votre_mot_de_passe"; \
		exit 1; \
	fi

es-info:
	@if [ -f ".env" ]; then \
		set -o allexport && source .env && set +o allexport && \
		echo "📊 Informations Elasticsearch:" && \
		curl -k -s -u "elastic:$ELASTIC_PWD" "https://localhost:9200/_cluster/health?pretty" | jq '.' || \
		echo "Impossible de récupérer les informations"; \
	fi

es-indices:
	@if [ -f ".env" ]; then \
		set -o allexport && source .env && set +o allexport && \
		echo "📑 Indices Elasticsearch:" && \
		curl -k -s -u "elastic:$ELASTIC_PWD" "https://localhost:9200/_cat/indices?v" || \
		echo "Impossible de lister les indices"; \
	fi


# Utilitaires
check-connection:
	@if [ -f ".env" ]; then \
		set -o allexport && source .env && set +o allexport && \
		export ES_HOST=localhost:9200 && \
		export ES_USER=elastic && \
		export ES_PASSWORD=$ELASTIC_PWD && \
		export no_proxy="localhost,127.0.0.1" && \
		$(POETRY) run python -c "from $(PROJECT).utils import test_connection; import asyncio; asyncio.run(test_connection())"; \
	else \
		echo "❌ Fichier .env introuvable"; \
		exit 1; \
	fi

setup-env:
	@if [ ! -f ".env" ]; then \
		echo "Création du fichier .env..."; \
		echo "# Configuration Elasticsearch" > .env; \
		echo "ELASTIC_PWD=your_password_here" >> .env; \
		echo "ES_HOST=localhost:9200" >> .env; \
		echo "ES_USER=elastic" >> .env; \
		echo "" >> .env; \
		echo "# Configuration ColPali" >> .env; \
		echo "COLPALI_MODEL=vidore/colqwen2-v1.0" >> .env; \
		echo "" >> .env; \
		echo "# Index" >> .env; \
		echo "ES_INDEX=image_embeddings" >> .env; \
		echo "✅ Fichier .env créé. Modifiez ELASTIC_PWD avec votre mot de passe."; \
	else \
		echo "ℹ️ Fichier .env existe déjà"; \
	fi

index-sample:
	$(POETRY) run python scripts/index_sample.py

search-sample:
	$(POETRY) run python scripts/search_sample.py

# Documentation
docs:
	$(POETRY) run mkdocs serve

docs-build:
	$(POETRY) run mkdocs build

setup: install-dev setup-env es-check
	@echo "✅ Environnement de développement prêt!"
	@echo "📝 Modifiez le fichier .env avec votre mot de passe Elasticsearch"
	@echo "🔍 Utilisez 'make es-check' pour vérifier la connexion"
	@echo "🚀 Utilisez 'make dev' pour lancer le serveur"
	@echo "🧪 Utilisez 'make test' pour lancer les tests"

# Aide
help:
	@echo "Image RAG MCP - Commandes disponibles:"
	@echo ""
	@echo "Installation:"
	@echo "  make install       - Installer les dépendances"
	@echo "  make install-dev   - Installer avec dépendances de dev"
	@echo "  make setup         - Configuration complète de dev"
	@echo ""
	@echo "Développement:"
	@echo "  make dev           - Lancer le serveur"
	@echo "  make dev-debug     - Lancer en mode debug"
	@echo "  make test          - Lancer les tests"
	@echo "  make test-cov      - Tests avec couverture"
	@echo "  make lint          - Vérifier le code"
	@echo "  make format        - Formater le code"
	@echo ""
	@echo "Elasticsearch:"
	@echo "  make es-check      - Vérifier la connexion ES"
	@echo "  make es-info       - Informations du cluster"
	@echo "  make es-indices    - Lister les indices"
	@echo "  make setup-env     - Créer le fichier .env"
	@echo ""
	@echo "Build:"
	@echo "  make build         - Construire le package"
	@echo "  make publish       - Publier sur PyPI"
	@echo "  make docker-build  - Construire l'image Docker"
	@echo ""
	@echo "Utilitaires:"
	@echo "  make check-connection - Tester la connexion ES"
	@echo "  make clean         - Nettoyer les fichiers"
	@echo "  make help          - Afficher cette aide"