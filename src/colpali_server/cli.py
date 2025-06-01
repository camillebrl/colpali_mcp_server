"""Interface CLI pour le serveur MCP Image RAG."""

import argparse
import asyncio
import logging
import signal
import sys

from mcp.server.stdio import stdio_server

from .image_rag_server import ImageRAGServer

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("colpali_server.log"),
        # logging.StreamHandler(sys.stderr)
    ],
)

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Créer le parser d'arguments."""
    parser = argparse.ArgumentParser(
        description="Serveur MCP pour RAG sur screenshots avec ColPali et Elasticsearch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Démarrer avec les paramètres par défaut
  python -m colpali_server.cli
  
  # Spécifier les credentials Elasticsearch
  python -m colpali_server.cli --es-host localhost:9200 --es-user elastic --es-password mypass
  
  # Utiliser un modèle ColPali différent
  python -m colpali_server.cli --model vidore/colpali

Fonctionnalités:
  - Indexation automatique des screenshots par source
  - Recherche intelligente avec sélection d'index automatique
  - Support de multiples sources/sites dans des index séparés
        """,
    )

    # Paramètres Elasticsearch
    es_group = parser.add_argument_group("Elasticsearch")
    es_group.add_argument(
        "--es-host",
        default=None,
        help="Hôte Elasticsearch (défaut: depuis ES_HOST ou localhost:9200)",
    )
    es_group.add_argument(
        "--es-user",
        default=None,
        help="Utilisateur Elasticsearch (défaut: depuis ES_USER ou elastic)",
    )
    es_group.add_argument(
        "--es-password",
        default=None,
        help="Mot de passe Elasticsearch (défaut: depuis ES_PASSWORD)",
    )

    # Paramètres ColPali
    model_group = parser.add_argument_group("Modèle ColPali")
    model_group.add_argument(
        "--model",
        default="vidore/colqwen2-v1.0",
        help="Modèle ColPali à utiliser (défaut: vidore/colqwen2-v1.0)",
    )

    # Paramètres généraux
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Niveau de log (défaut: INFO)",
    )

    parser.add_argument("--version", action="version", version="%(prog)s 1.0.0")

    return parser


async def async_main() -> None:
    """Point d'entrée asynchrone principal."""
    parser = create_parser()
    args = parser.parse_args()

    # Configurer le niveau de log
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    logger.info("🚀 Démarrage du serveur MCP Image RAG pour Screenshots...")
    logger.info(f"🤖 Modèle: {args.model}")

    # Créer le serveur
    try:
        server = ImageRAGServer(
            es_host=args.es_host,
            es_user=args.es_user,
            es_password=args.es_password,
            colpali_model_path=args.model,
        )
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation du serveur: {e}")
        sys.exit(1)

    # Gestion des signaux
    def signal_handler() -> None:
        logger.info("📡 Signal reçu, arrêt du serveur...")
        sys.exit(0)

    if sys.platform != "win32":
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, signal_handler)

    # Lancer le serveur avec stdio
    try:
        async with stdio_server() as (read_stream, write_stream):
            logger.info("✅ Serveur prêt à recevoir des requêtes")
            logger.info("🔧 Outils disponibles:")
            logger.info("   - search_screenshots: Recherche intelligente avec sélection d'index automatique")
            logger.info("   - index_screenshots: Indexation par source depuis le dossier screenshots")
            logger.info("   - list_screenshot_indices: Liste des index disponibles")
            logger.info("   - delete_screenshot_index: Suppression d'un index spécifique")
            await server.run(read_stream, write_stream)
    except KeyboardInterrupt:
        logger.info("👋 Arrêt du serveur par l'utilisateur")
    except Exception as e:
        logger.error(f"❌ Erreur fatale du serveur: {e}")
        import traceback

        logger.error(traceback.format_exc())
        sys.exit(1)


def main() -> None:
    """Point d'entrée principal."""
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        logger.info("👋 Arrêt du serveur")
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
