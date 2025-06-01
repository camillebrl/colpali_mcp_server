#!/bin/bash

# IMPORTANT: Rediriger TOUS les messages de debug vers stderr
# MCP nécessite que stdout soit réservé uniquement pour les messages JSON

# Logs de debug vers un fichier séparé
DEBUG_LOG="/tmp/colpali_mcp_startup.log"
exec 2>$DEBUG_LOG

echo "=== Démarrage du serveur ColPali avec Elasticsearch ===" >&2
echo "Date: $(date)" >&2

echo "🔄 Vidage de la mémoire CUDA..." >&2

# 1) Tentative de reset via nvidia-smi (nécessite root)
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "-> Tentative de reset via nvidia-smi (root requis)..." >&2
    sudo nvidia-smi --gpu-reset --kill-processes >/dev/null 2>&1 \
        && echo "✅ nvidia-smi GPU reset réussi" >&2 \
        || echo "⚠️ Échec du reset GPU via nvidia-smi (droits root manquants ?)" >&2
else
    echo "⚠️ nvidia-smi non trouvé, saut du reset GPU" >&2
fi
python3 - << 'EOF' 2>>$DEBUG_LOG
try:
    import torch
    torch.cuda.empty_cache()
    print("✅ torch.cuda.empty_cache() exécuté", file=sys.stderr)
except Exception as e:
    print(f"⚠️ Impossible d'exécuter torch.cuda.empty_cache(): {e}", file=sys.stderr)
EOF
echo "🔄 Mémoire CUDA vidée (autant que possible)" >&2

# Aller dans le répertoire du projet
cd /home/camil/Documents/colpali_server || {
    echo "❌ Impossible d'accéder au répertoire du projet" >&2
    exit 1
}

echo "Répertoire de travail: $(pwd)" >&2

# Variables d'environnement avec validation
export ES_HOST="${ES_HOST:-localhost:9200}"
export ES_USER="${ES_USER:-elastic}"
export ES_PASSWORD="${ES_PASSWORD}"

if [ -z "$ES_PASSWORD" ]; then
    echo "❌ ES_PASSWORD non défini" >&2
    exit 1
fi

echo "✅ Configuration détectée:" >&2
echo "   ES_HOST: $ES_HOST" >&2
echo "   ES_USER: $ES_USER" >&2
echo "   ES_PASSWORD: [DÉFINI]" >&2

# Vérifier si nous sommes dans le bon répertoire
if [ ! -f "pyproject.toml" ]; then
    echo "❌ pyproject.toml non trouvé. Êtes-vous dans le bon répertoire ?" >&2
    exit 1
fi

# Path pour Poetry
export PATH="/home/camil/.local/bin:$PATH"

# Vérifier si Poetry est installé
if ! command -v poetry >/dev/null 2>&1; then
    echo "❌ Poetry n'est pas installé ou n'est pas dans le PATH" >&2
    exit 1
fi
echo "✅ Poetry trouvé: $(poetry --version)" >&2

# Vérifier le projet Poetry
if ! poetry check >/dev/null 2>&1; then
    echo "❌ Projet Poetry invalide" >&2
    exit 1
fi
echo "✅ Projet Poetry valide" >&2

# Test d'import rapide
if ! poetry run python -c "import colpali_server" >/dev/null 2>&1; then
    echo "❌ Impossible d'importer colpali_server" >&2
    exit 1
fi
echo "✅ Imports Python OK" >&2

# Vérifier la connexion Elasticsearch (rapide)
echo "🔍 Vérification de la connexion à Elasticsearch..." >&2
if curl -k -s --connect-timeout 3 -u "$ES_USER:$ES_PASSWORD" "https://$ES_HOST" >/dev/null 2>&1; then
    echo "✅ Connexion à Elasticsearch réussie" >&2
else
    sudo systemctl start elasticsearch
    echo "⚠️ Elasticsearch non immédiatement accessible (continuons quand même)" >&2
fi

echo "🚀 Lancement du serveur ColPali MCP..." >&2
echo "Tous les vérifications passées, démarrage du serveur..." >&2

# Lancer le serveur ColPali
# stdout reste propre pour MCP, stderr va dans le log de debug
exec poetry run python -m colpali_server.cli --log-level INFO 2>>$DEBUG_LOG