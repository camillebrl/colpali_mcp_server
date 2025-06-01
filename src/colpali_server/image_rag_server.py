"""Serveur MCP pour RAG sur base de données d'images avec ColPali et Elasticsearch."""

import base64
import logging
import re
import traceback
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import EmbeddedResource, ImageContent, TextContent, Tool
from PIL import Image

from .colpali_model import ColPaliModel
from .elasticsearch_model import ElasticsearchModel

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ScreenshotProcessor:
    """Processeur de screenshots pour l'extraction et l'indexation."""

    def __init__(self, screenshots_dir: str = "./screenshots"):
        """Initialise le processeur de screenshots."""
        self.screenshots_dir = Path(screenshots_dir)
        if not self.screenshots_dir.exists():
            raise FileNotFoundError(f"Répertoire screenshots introuvable: {screenshots_dir}")
        logger.info(f"Répertoire screenshots initialisé: {self.screenshots_dir}")

    def get_source_from_filename(self, filename: str) -> str:
        """Extrait la source/URL depuis le nom de fichier.

        Formats supportés:
        - orange_capex_q1_2025_page_001.png -> orange_capex_q1_2025
        - site_example.com_page.png -> site_example.com
        - document_name.png -> document_name.
        """
        # Supprimer l'extension
        name_without_ext = Path(filename).stem

        # Patterns pour retirer les suffixes de pagination
        pagination_patterns = [
            r"_page_\d+$",  # _page_001
            r"_p\d+$",  # _p1
            r"_\d{3,}$",  # _001 (3+ chiffres à la fin)
            r"_sheet_\d+$",  # _sheet_1
        ]

        # Appliquer les patterns pour nettoyer le nom
        cleaned_name = name_without_ext
        for pattern in pagination_patterns:
            cleaned_name = re.sub(pattern, "", cleaned_name)

        # Si on a réussi à nettoyer quelque chose, utiliser le nom nettoyé
        if cleaned_name != name_without_ext:
            source = cleaned_name
        else:
            # Sinon, chercher un pattern de domaine
            domain_pattern = r"([a-zA-Z0-9-]+\.[a-zA-Z]{2,})"
            match = re.search(domain_pattern, name_without_ext)
            source = match.group(1) if match else name_without_ext

        # Nettoyer et normaliser le nom d'index
        source = re.sub(r"[^a-zA-Z0-9\._-]", "_", source)
        return source.lower()

    def process_screenshots(
        self,
    ) -> dict[str, list[tuple[Image.Image, dict[str, Any]]]]:
        """Traite tous les screenshots et les groupe par source.

        Returns:
            Dictionnaire avec source comme clé et liste d'images/métadonnées comme valeur
        """
        screenshots_by_source: dict[str, list[tuple[Image.Image, dict[str, Any]]]] = {}

        # Extensions d'images supportées
        image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}

        # Parcourir tous les fichiers d'images
        for image_file in self.screenshots_dir.iterdir():
            if image_file.suffix.lower() in image_extensions:
                try:
                    # Charger l'image
                    with Image.open(image_file) as img:
                        pil_image = img.copy()
                        if pil_image.mode != "RGB":
                            pil_image = pil_image.convert("RGB")

                    # Extraire la source
                    source = self.get_source_from_filename(image_file.name)

                    # Créer les métadonnées
                    metadata = {
                        "id": str(uuid.uuid4()),
                        "file_path": str(image_file),
                        "filename": image_file.name,
                        "source": source,
                        "file_size": image_file.stat().st_size,
                        "image_dimensions": f"{pil_image.width}x{pil_image.height}",
                    }

                    # Grouper par source
                    if source not in screenshots_by_source:
                        screenshots_by_source[source] = []

                    screenshots_by_source[source].append((pil_image, metadata))

                except Exception as e:
                    logger.error(f"Erreur lors du traitement de {image_file}: {e}")

        logger.info(
            f"Screenshots traités: {sum(len(images) for images in screenshots_by_source.values())} images dans {len(screenshots_by_source)} sources"
        )
        return screenshots_by_source

    @staticmethod
    def encode_image_to_base64(image: Image.Image) -> str:
        """Encode une image PIL en base64."""
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


class IndexSelector:
    """Sélecteur d'index intelligent basé sur la requête."""

    def __init__(self, es_model: ElasticsearchModel):
        self.es_model = es_model

    def get_all_indices(self) -> list[str]:
        """Récupère tous les index disponibles avec le préfixe screenshot_."""
        try:
            if not self.es_model.es:
                return []

            # Obtenir tous les index
            indices_info = self.es_model.es.cat.indices(format="json")
            screenshot_indices: list[str] = []
            for idx_info in indices_info:
                if isinstance(idx_info, dict):
                    index_name = idx_info.get("index", "")
                    if index_name.startswith("screenshot_") and not index_name.startswith("."):
                        screenshot_indices.append(index_name)

            return screenshot_indices

        except Exception as e:
            logger.error(f"Erreur lors de la récupération des index: {e}")
            return []

    def select_relevant_indices(self, query: str, max_indices: int = 3) -> list[str]:
        """Sélectionne les index les plus pertinents basés sur la requête.

        Args:
            query: Requête de recherche
            max_indices: Nombre maximum d'index à retourner

        Returns:
            Liste des noms d'index les plus pertinents
        """
        available_indices = self.get_all_indices()

        if not available_indices:
            logger.warning("Aucun index screenshot trouvé")
            return []

        # Si un seul index, le retourner
        if len(available_indices) == 1:
            return available_indices

        # Analyser la requête pour trouver des mots-clés de domaine/site
        query_lower = query.lower()

        # Extraire les domaines/URLs potentiels de la requête
        domain_patterns = [
            r"([a-zA-Z0-9-]+\.[a-zA-Z]{2,})",  # domaine.com
            r"(www\.[a-zA-Z0-9-]+\.[a-zA-Z]{2,})",  # www.domaine.com
            r"(https?://[^\s]+)",  # URLs complètes
        ]

        mentioned_domains = set()
        for pattern in domain_patterns:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                # Nettoyer le domaine
                domain = re.sub(r"^https?://", "", match)
                domain = re.sub(r"/.*$", "", domain)
                domain = re.sub(r"^www\.", "", domain)
                mentioned_domains.add(domain)

        # Scorer les index
        index_scores = []

        for index_name in available_indices:
            score = 0

            # Extraire le nom de source de l'index
            source_name = index_name.replace("screenshot_", "")

            # Score basé sur la correspondance exacte de domaine
            for domain in mentioned_domains:
                if domain in source_name or source_name in domain:
                    score += 10

            # Score basé sur les mots-clés dans la requête
            query_words = set(re.findall(r"\w+", query_lower))
            source_words = set(re.findall(r"\w+", source_name.replace("_", " ")))

            # Correspondance de mots
            common_words = query_words.intersection(source_words)
            score += len(common_words) * 2

            # Bonus pour les mots partiels
            for query_word in query_words:
                for source_word in source_words:
                    if (
                        len(query_word) > 3
                        and query_word in source_word
                        or len(source_word) > 3
                        and source_word in query_word
                    ):
                        score += 1

            index_scores.append((index_name, score))

        # Trier par score décroissant
        index_scores.sort(key=lambda x: x[1], reverse=True)

        # Si aucun score significatif, retourner tous les index (limités)
        if not index_scores or index_scores[0][1] == 0:
            logger.info("Aucune correspondance spécifique trouvée, recherche dans tous les index")
            return available_indices[:max_indices]

        # Retourner les index avec les meilleurs scores
        selected_indices = [idx for idx, score in index_scores[:max_indices] if score > 0]

        if not selected_indices:
            selected_indices = [index_scores[0][0]]  # Au moins un index

        logger.info(f"Index sélectionnés pour la requête '{query}': {selected_indices}")
        return selected_indices


class ImageRAGServer:
    """Serveur MCP pour recherche RAG sur images avec ColPali."""

    def __init__(
        self,
        es_host: str | None = None,
        es_user: str | None = None,
        es_password: str | None = None,
        colpali_model_path: str = "vidore/colqwen2-v1.0",
    ):
        """Initialise le serveur RAG."""
        self.server: Server = Server("image-rag-server")

        # Initialiser les modèles
        logger.info("🚀 Initialisation du serveur Image RAG...")

        try:
            # Modèle ColPali pour les embeddings
            logger.info("📦 Chargement du modèle ColPali...")
            self.colpali_model = ColPaliModel(model_path=colpali_model_path)

            # Client Elasticsearch (sans index spécifique)
            logger.info("🔌 Connexion à Elasticsearch...")
            self.es_model = ElasticsearchModel(
                index_name="temp",  # Index temporaire
                es_host=es_host,
                es_user=es_user,
                es_password=es_password,
            )

            # Vérifier la connexion
            stats = self.es_model.get_stats()
            if stats.get("status") == "connected":
                logger.info("✅ Connecté à Elasticsearch")
            else:
                logger.warning("⚠️ Connexion Elasticsearch non établie")

            # Initialiser le sélecteur d'index
            self.index_selector = IndexSelector(self.es_model)

        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation: {e}")
            logger.error(traceback.format_exc())
            raise

        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Configure les gestionnaires MCP."""

        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            """Liste les outils disponibles."""
            tools = [
                Tool(
                    name="search_screenshots",
                    description="Rechercher des screenshots pertinents dans tous les index en utilisant une requête textuelle. L'outil sélectionne automatiquement les index les plus pertinents.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Requête de recherche en langage naturel",
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Nombre de résultats à retourner par index",
                                "default": 5,
                                "minimum": 1,
                                "maximum": 20,
                            },
                            "max_indices": {
                                "type": "integer",
                                "description": "Nombre maximum d'index à rechercher",
                                "default": 3,
                                "minimum": 1,
                                "maximum": 10,
                            },
                        },
                        "required": ["query"],
                    },
                ),
                Tool(
                    name="index_screenshots",
                    description="Créer des index Elasticsearch à partir du dossier 'screenshots'. Chaque source (site/fichier) aura son propre index.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "screenshots_dir": {
                                "type": "string",
                                "description": "Chemin vers le dossier screenshots",
                                "default": "./screenshots",
                            },
                            "overwrite_existing": {
                                "type": "boolean",
                                "description": "Supprimer et recréer les index existants",
                                "default": False,
                            },
                            "batch_size": {
                                "type": "integer",
                                "description": "Taille du batch pour l'indexation",
                                "default": 10,
                                "minimum": 1,
                                "maximum": 50,
                            },
                        },
                    },
                ),
                Tool(
                    name="list_screenshot_indices",
                    description="Lister tous les index de screenshots disponibles avec leurs statistiques",
                    inputSchema={"type": "object", "properties": {}},
                ),
                Tool(
                    name="delete_screenshot_index",
                    description="Supprimer un index de screenshots spécifique",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "index_name": {
                                "type": "string",
                                "description": "Nom de l'index à supprimer (doit commencer par 'screenshot_')",
                            },
                            "confirm": {
                                "type": "boolean",
                                "description": "Confirmer la suppression",
                                "default": False,
                            },
                        },
                        "required": ["index_name", "confirm"],
                    },
                ),
            ]
            return tools

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> list[TextContent | ImageContent | EmbeddedResource]:
            """Exécute un outil."""
            try:
                if name == "search_screenshots":
                    return await self._search_screenshots(arguments)
                elif name == "index_screenshots":
                    return await self._index_screenshots(arguments)
                elif name == "list_screenshot_indices":
                    return await self._list_screenshot_indices(arguments)
                elif name == "delete_screenshot_index":
                    return await self._delete_screenshot_index(arguments)
                else:
                    return [TextContent(type="text", text=f"❌ Outil inconnu: {name}")]

            except Exception as e:
                logger.exception(f"Erreur lors de l'exécution de l'outil {name}")
                return [TextContent(type="text", text=f"❌ Erreur: {str(e)}")]

    async def _search_screenshots(self, args: dict[str, Any]) -> list[TextContent | ImageContent | EmbeddedResource]:
        """Recherche des screenshots pertinents dans les index sélectionnés."""
        query = args["query"]
        top_k = args.get("top_k", 5)
        max_indices = args.get("max_indices", 3)

        logger.info(f"🔍 Recherche: '{query}' (top_k={top_k}, max_indices={max_indices})")

        try:
            # Sélectionner les index pertinents
            selected_indices = self.index_selector.select_relevant_indices(query, max_indices)

            if not selected_indices:
                return [
                    TextContent(
                        type="text",
                        text="❌ Aucun index de screenshots trouvé. Utilisez 'index_screenshots' pour créer des index.",
                    )
                ]

            # Générer l'embedding de la requête
            logger.info("🧮 Génération de l'embedding de la requête...")
            query_embedding = self.colpali_model.generate_query_embedding(query)

            # Rechercher dans chaque index sélectionné
            all_results = []

            for index_name in selected_indices:
                logger.info(f"🔎 Recherche dans l'index: {index_name}")

                # Créer une instance ES avec l'index spécifique
                es_instance = ElasticsearchModel(
                    index_name=index_name,
                    es_host=self.es_model.es_host,
                    es_user=self.es_model.es_user,
                    es_password=self.es_model.es_password,
                )

                # Rechercher
                results = es_instance.search_by_embedding(query_embedding, k=top_k)

                # Ajouter le nom de l'index aux résultats
                for result in results:
                    result["search_index"] = index_name
                    result["source_extracted"] = index_name.replace("screenshot_", "")

                all_results.extend(results)
                es_instance.close()

            if not all_results:
                return [
                    TextContent(
                        type="text",
                        text=f"❌ Aucun résultat trouvé pour: '{query}' dans les index: {', '.join(selected_indices)}",
                    )
                ]

            # Trier tous les résultats par score
            all_results.sort(key=lambda x: x["score"], reverse=True)

            # Limiter le nombre total de résultats
            all_results = all_results[: top_k * max_indices]

            # Préparer la réponse
            content: list[TextContent | ImageContent | EmbeddedResource] = []

            # Texte de résumé
            summary = f"📚 {len(all_results)} résultat(s) trouvé(s) pour: '{query}'\n"
            summary += f"🎯 Index recherchés: {', '.join(selected_indices)}\n\n"

            # Grouper par index pour l'affichage
            results_by_index: dict[str, list[dict[str, Any]]] = {}
            for result in all_results:
                index_name = result["search_index"]
                if index_name not in results_by_index:
                    results_by_index[index_name] = []
                results_by_index[index_name].append(result)

            for index_name, results in results_by_index.items():
                source_name = index_name.replace("screenshot_", "")
                summary += f"📂 Source: {source_name} ({len(results)} résultats)\n"

                for i, result in enumerate(results, 1):
                    summary += f"  {i}. 📷 {result.get('filename', 'N/A')} (Score: {result['score']:.3f})\n"
                    if result.get("metadata"):
                        dims = result["metadata"].get("image_dimensions", "N/A")
                        summary += f"      📐 Dimensions: {dims}\n"
                summary += "\n"

            content.append(TextContent(type="text", text=summary))

            # Ajouter les images encodées en base64 si disponibles
            for result in all_results[:10]:  # Limiter à 10 images pour éviter de surcharger
                if result.get("image_base64"):
                    try:
                        content.append(
                            ImageContent(
                                type="image",
                                data=result["image_base64"],
                                mimeType="image/jpeg",
                            )
                        )

                        # Ajouter une légende pour l'image
                        caption = f"📷 {result.get('filename', 'N/A')} (Source: {result.get('source_extracted', 'N/A')}, Score: {result['score']:.3f})"
                        content.append(TextContent(type="text", text=caption))
                    except Exception as e:
                        logger.warning(f"Impossible d'afficher l'image: {e}")

            return content

        except Exception as e:
            logger.error(f"Erreur lors de la recherche: {e}")
            logger.error(traceback.format_exc())
            return [
                TextContent(
                    type="text",
                    text=f"❌ Erreur lors de la recherche: {str(e)}",
                )
            ]

    async def _index_screenshots(self, args: dict[str, Any]) -> list[TextContent | ImageContent | EmbeddedResource]:
        """Indexe les screenshots depuis le dossier spécifié."""
        screenshots_dir = args.get("screenshots_dir", "./screenshots")
        overwrite_existing = args.get("overwrite_existing", False)
        batch_size = args.get("batch_size", 10)

        logger.info(f"📥 Indexation des screenshots depuis: {screenshots_dir}")

        try:
            # Initialiser le processeur de screenshots
            processor = ScreenshotProcessor(screenshots_dir)

            # Traiter tous les screenshots
            screenshots_by_source = processor.process_screenshots()

            if not screenshots_by_source:
                return [
                    TextContent(
                        type="text",
                        text=f"❌ Aucun screenshot trouvé dans: {screenshots_dir}",
                    )
                ]

            # Statistiques globales
            total_images = sum(len(images) for images in screenshots_by_source.values())
            total_sources = len(screenshots_by_source)
            indexed_sources = 0
            total_indexed = 0

            summary = f"📊 Indexation de {total_images} screenshots de {total_sources} sources\n\n"

            # Indexer chaque source dans son propre index
            for source, images_metadata in screenshots_by_source.items():
                index_name = f"screenshot_{source}"

                logger.info(f"📂 Traitement de la source '{source}' -> index '{index_name}'")

                # Vérifier si l'index existe
                if self.es_model.es and self.es_model.es.indices.exists(index=index_name):
                    if overwrite_existing:
                        logger.info(f"🗑️ Suppression de l'index existant: {index_name}")
                        self.es_model.es.indices.delete(index=index_name)
                    else:
                        logger.info(f"⚠️ Index existant ignoré: {index_name}")
                        summary += f"⚠️ {source}: Index existant ignoré ({len(images_metadata)} images)\n"
                        continue

                # Créer une instance ES pour cet index
                es_instance = ElasticsearchModel(
                    index_name=index_name,
                    es_host=self.es_model.es_host,
                    es_user=self.es_model.es_user,
                    es_password=self.es_model.es_password,
                )

                # Traiter toutes les images de cette source
                source_indexed = 0

                # Extraire toutes les images de cette source
                all_images = [img for img, _ in images_metadata]
                all_metadata = [metadata for _, metadata in images_metadata]

                # Traiter par batch pour optimiser l'utilisation mémoire
                for i in range(0, len(all_images), batch_size):
                    batch_images = all_images[i : i + batch_size]
                    batch_metadata = all_metadata[i : i + batch_size]

                    logger.info(
                        f"🧮 Génération des embeddings pour le batch {i//batch_size + 1}/{(len(all_images) + batch_size - 1)//batch_size} (source: {source})"
                    )

                    try:
                        # Générer tous les embeddings du batch en une fois
                        embeddings = self.colpali_model.generate_embeddings(batch_images)

                        # Préparer les documents pour l'indexation
                        batch_documents = []
                        for _j, (image, metadata, embedding) in enumerate(
                            zip(
                                batch_images,
                                batch_metadata,
                                embeddings,
                                strict=False,
                            )
                        ):
                            # Encoder l'image en base64
                            image_base64 = ScreenshotProcessor.encode_image_to_base64(image)

                            # Préparer le document
                            doc = {
                                "col_pali_vectors": embedding,
                                "metadata": metadata,
                                "image_base64": image_base64,
                                "filename": metadata["filename"],
                                "source": source,
                            }

                            batch_documents.append(doc)

                        # Indexer ce batch
                        if batch_documents:
                            success = es_instance.bulk_index_documents(batch_documents)
                            if success:
                                source_indexed += len(batch_documents)
                                logger.info(f"💾 Batch indexé: {len(batch_documents)} images")
                            else:
                                logger.error(f"❌ Échec de l'indexation du batch {i//batch_size + 1}")

                    except Exception as e:
                        logger.error(
                            f"❌ Erreur lors du traitement du batch {i//batch_size + 1} pour la source {source}: {e}"
                        )
                        # Continuer avec le batch suivant
                        continue

                es_instance.close()

                if source_indexed > 0:
                    indexed_sources += 1
                    total_indexed += source_indexed
                    summary += (
                        f"✅ {source}: {source_indexed}/{len(images_metadata)} images indexées dans '{index_name}'\n"
                    )
                else:
                    summary += f"❌ {source}: Échec de l'indexation\n"

            summary += "\n📈 Résumé final:\n"
            summary += f"Sources traitées: {indexed_sources}/{total_sources}\n"
            summary += f"Images indexées: {total_indexed}/{total_images}\n"

            return [TextContent(type="text", text=summary)]

        except Exception as e:
            logger.error(f"Erreur lors de l'indexation: {e}")
            logger.error(traceback.format_exc())
            return [
                TextContent(
                    type="text",
                    text=f"❌ Erreur lors de l'indexation: {str(e)}",
                )
            ]

    async def _list_screenshot_indices(
        self, args: dict[str, Any]
    ) -> list[TextContent | ImageContent | EmbeddedResource]:
        """Liste tous les index de screenshots avec leurs statistiques."""
        try:
            available_indices = self.index_selector.get_all_indices()

            if not available_indices:
                return [
                    TextContent(
                        type="text",
                        text="📂 Aucun index de screenshots trouvé.\nUtilisez 'index_screenshots' pour créer des index.",
                    )
                ]

            summary = f"📂 Index de screenshots disponibles ({len(available_indices)})\n\n"

            for index_name in available_indices:
                # Obtenir les statistiques de l'index
                es_instance = ElasticsearchModel(
                    index_name=index_name,
                    es_host=self.es_model.es_host,
                    es_user=self.es_model.es_user,
                    es_password=self.es_model.es_password,
                )

                stats = es_instance.get_stats()
                source_name = index_name.replace("screenshot_", "")

                summary += f"🗂️ {source_name}\n"
                summary += f"   📋 Index: {index_name}\n"
                summary += f"   📊 Documents: {stats.get('document_count', 0):,}\n"

                if stats.get("index_size_bytes"):
                    size_mb = stats["index_size_bytes"] / (1024 * 1024)
                    summary += f"   💾 Taille: {size_mb:.2f} MB\n"

                summary += "\n"
                es_instance.close()

            return [TextContent(type="text", text=summary)]

        except Exception as e:
            logger.error(f"Erreur lors de la récupération des index: {e}")
            return [TextContent(type="text", text=f"❌ Erreur: {str(e)}")]

    async def _delete_screenshot_index(
        self, args: dict[str, Any]
    ) -> list[TextContent | ImageContent | EmbeddedResource]:
        """Supprime un index de screenshots spécifique."""
        index_name = args.get("index_name")
        confirm = args.get("confirm", False)

        # Vérifier que index_name est fourni
        if not index_name:
            return [TextContent(type="text", text="❌ Le paramètre 'index_name' est requis")]

        if not confirm:
            return [
                TextContent(
                    type="text",
                    text="⚠️ Veuillez confirmer la suppression avec confirm=true",
                )
            ]

        if not index_name.startswith("screenshot_"):
            return [
                TextContent(
                    type="text",
                    text="❌ Seuls les index commençant par 'screenshot_' peuvent être supprimés",
                )
            ]

        try:
            # Créer une instance ES pour cet index
            es_instance = ElasticsearchModel(
                index_name=index_name,
                es_host=self.es_model.es_host,
                es_user=self.es_model.es_user,
                es_password=self.es_model.es_password,
            )

            success = es_instance.delete_index()
            es_instance.close()

            if success:
                return [
                    TextContent(
                        type="text",
                        text=f"✅ Index '{index_name}' supprimé avec succès",
                    )
                ]
            else:
                return [
                    TextContent(
                        type="text",
                        text=f"❌ Échec de la suppression de l'index '{index_name}'",
                    )
                ]

        except Exception as e:
            logger.error(f"Erreur lors de la suppression de l'index: {e}")
            return [TextContent(type="text", text=f"❌ Erreur: {str(e)}")]

    async def run(self, read_stream: Any, write_stream: Any) -> None:
        """Lance le serveur MCP."""
        logger.info("🌐 Démarrage du serveur Image RAG...")

        try:
            # Créer des options d'initialisation minimales mais complètes
            init_options = InitializationOptions(
                server_name="colpali-server",
                server_version="0.1.0",
                capabilities=self.server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            )

            await self.server.run(read_stream, write_stream, init_options)

        except Exception as e:
            logger.error(f"❌ Erreur serveur: {e}")
            logger.error(traceback.format_exc())
            raise
        finally:
            # Nettoyage
            logger.info("🧹 Arrêt du serveur...")
            try:
                if hasattr(self, "colpali_model"):
                    self.colpali_model.cleanup()
                if hasattr(self, "es_model"):
                    self.es_model.close()
            except Exception as e:
                logger.warning(f"Avertissement lors du nettoyage: {e}")


async def main() -> None:
    """Point d'entrée principal."""
    import argparse

    parser = argparse.ArgumentParser(description="Serveur MCP pour RAG sur screenshots")
    parser.add_argument("--es-host", help="Hôte Elasticsearch")
    parser.add_argument("--es-user", help="Utilisateur Elasticsearch")
    parser.add_argument("--es-password", help="Mot de passe Elasticsearch")
    parser.add_argument("--model", default="vidore/colqwen2-v1.0", help="Modèle ColPali")

    args = parser.parse_args()

    server = ImageRAGServer(
        es_host=args.es_host,
        es_user=args.es_user,
        es_password=args.es_password,
        colpali_model_path=args.model,
    )

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
