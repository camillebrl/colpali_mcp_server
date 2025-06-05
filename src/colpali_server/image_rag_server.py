"""Serveur MCP pour RAG sur base de données d'images avec ColPali et Elasticsearch."""

import asyncio
import base64
import contextlib
import logging
import os
import re
import shutil
import traceback
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import EmbeddedResource, ImageContent, TextContent, Tool
from pdf2image import convert_from_path
from PIL import Image

from .colpali_model import get_colpali_manager
from .elasticsearch_model import ElasticsearchModel

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class PDFConverter:
    """Convertisseur de PDF en images pour l'indexation."""

    def __init__(self) -> None:
        """Initialise le convertisseur PDF."""
        pass

    def convert_pdf_to_images(
        self,
        pdf_path: str,
        output_dir: str,
        dpi: int = 200,
        image_format: str = "png",
        prefix: str | None = None,
        single_file: bool = False,
    ) -> tuple[int, list[str]]:
        """Convertit un PDF en images.

        Args:
            pdf_path: Chemin vers le fichier PDF
            output_dir: Dossier de sortie pour les images
            dpi: Résolution des images (défaut: 200)
            image_format: Format des images (png, jpg, jpeg)
            prefix: Préfixe pour les noms de fichiers (défaut: nom du PDF)
            single_file: Si True, traite toutes les pages comme un seul document

        Returns:
            Tuple (nombre d'images créées, liste des chemins d'images)
        """
        # Vérifier que le PDF existe
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Fichier PDF introuvable: {pdf_path}")

        # Créer le dossier de sortie
        os.makedirs(output_dir, exist_ok=True)

        # Déterminer le préfixe
        if prefix is None:
            prefix = Path(pdf_path).stem

        # Nettoyer le préfixe
        prefix = re.sub(r"[^a-zA-Z0-9\._-]", "_", prefix)

        logger.info(f"🔄 Conversion du PDF: {pdf_path}")
        logger.info(f"📁 Dossier de sortie: {output_dir}")
        logger.info(f"🔧 DPI: {dpi}, Format: {image_format}")

        try:
            # Convertir le PDF en images
            images = convert_from_path(pdf_path, dpi=dpi)

            image_paths = []

            if single_file:
                # Sauvegarder toutes les pages avec le même nom de base
                for i, image in enumerate(images):
                    filename = f"{prefix}_page_{i+1:03d}.{image_format}"
                    image_path = os.path.join(output_dir, filename)
                    image.save(image_path, image_format.upper())
                    image_paths.append(image_path)
                    logger.info(f"✅ Page {i+1}/{len(images)} sauvegardée: {filename}")

            return len(images), image_paths

        except Exception as e:
            logger.error(f"Erreur lors de la conversion PDF: {e}")
            raise

    def convert_pdfs_directory(
        self,
        input_dir: str,
        output_dir: str,
        dpi: int = 200,
        image_format: str = "png",
        single_file_per_pdf: bool = True,
        clear_output: bool = False,
    ) -> dict[str, int]:
        """Convertit tous les PDFs d'un répertoire en images.

        Args:
            input_dir: Dossier contenant les PDFs
            output_dir: Dossier de sortie pour les images
            dpi: Résolution des images
            image_format: Format des images
            single_file_per_pdf: Si True, groupe toutes les pages d'un PDF
            clear_output: Si True, vide le dossier de sortie avant

        Returns:
            Dictionnaire {nom_pdf: nombre_pages}
        """
        # Vérifier le dossier d'entrée
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Dossier PDF introuvable: {input_dir}")

        # Vider le dossier de sortie si demandé
        if clear_output and os.path.exists(output_dir):
            logger.warning(f"🗑️ Suppression du contenu de: {output_dir}")
            shutil.rmtree(output_dir)

        # Créer le dossier de sortie
        os.makedirs(output_dir, exist_ok=True)

        # Lister tous les PDFs
        pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]

        if not pdf_files:
            logger.warning(f"⚠️ Aucun fichier PDF trouvé dans: {input_dir}")
            return {}

        results = {}

        for pdf_file in pdf_files:
            pdf_path = os.path.join(input_dir, pdf_file)
            pdf_name = Path(pdf_file).stem

            try:
                num_pages, _ = self.convert_pdf_to_images(
                    pdf_path=pdf_path,
                    output_dir=output_dir,
                    dpi=dpi,
                    image_format=image_format,
                    prefix=pdf_name,
                    single_file=single_file_per_pdf,
                )
                results[pdf_file] = num_pages

            except Exception as e:
                logger.error(f"❌ Échec de conversion pour {pdf_file}: {e}")
                results[pdf_file] = 0

        return results


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

    def select_relevant_indices(self, query: str, max_indices: int = 2) -> list[str]:
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
        self.colpali_model_path = colpali_model_path
        self.cleanup_task: asyncio.Task | None = None

        # Initialiser les modèles
        logger.info("🚀 Initialisation du serveur Image RAG...")

        try:
            # Ne PAS charger le modèle ColPali au démarrage
            logger.info("📦 Modèle ColPali configuré (chargement différé)")

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

    async def _periodic_cleanup(self) -> None:
        """Tâche périodique pour vérifier et décharger le modèle si inutilisé."""
        while True:
            try:
                await asyncio.sleep(30)  # Vérifier toutes les 30 secondes
                manager = get_colpali_manager()
                manager.check_and_unload()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Erreur dans la tâche de nettoyage: {e}")

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
                                "default": 2,
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
                                "default": 1,
                                "minimum": 1,
                                "maximum": 10,
                            },
                            "single_index": {
                                "type": "boolean",
                                "description": "Créer un seul index pour toutes les images (défaut: true)",
                                "default": True,
                            },
                            "index_name": {
                                "type": "string",
                                "description": "Nom personnalisé pour l'index (utilisé uniquement si single_index=true)",
                                "default": None,
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
                Tool(
                    name="get_model_status",
                    description="Obtenir l'état actuel du modèle ColPali (chargé/déchargé, mémoire utilisée, etc.)",
                    inputSchema={"type": "object", "properties": {}},
                ),
                Tool(
                    name="convert_pdf_to_images",
                    description="Convertir des fichiers PDF en images pour l'indexation avec ColPali. Peut traiter un PDF unique ou un dossier entier.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "pdf_path": {
                                "type": "string",
                                "description": "Chemin vers un fichier PDF spécifique (utilisé si pdf_directory n'est pas fourni)",
                            },
                            "pdf_directory": {
                                "type": "string",
                                "description": "Dossier contenant des fichiers PDF à convertir",
                            },
                            "output_directory": {
                                "type": "string",
                                "description": "Dossier de sortie pour les images converties",
                                "default": "./screenshots",
                            },
                            "dpi": {
                                "type": "integer",
                                "description": "Résolution DPI pour la conversion (plus élevé = meilleure qualité)",
                                "default": 200,
                                "minimum": 72,
                                "maximum": 600,
                            },
                            "image_format": {
                                "type": "string",
                                "description": "Format des images de sortie",
                                "enum": ["png", "jpg", "jpeg"],
                                "default": "png",
                            },
                            "clear_output": {
                                "type": "boolean",
                                "description": "Vider le dossier de sortie avant la conversion",
                                "default": False,
                            },
                            "single_file_per_pdf": {
                                "type": "boolean",
                                "description": "Grouper toutes les pages d'un PDF sous le même nom de base",
                                "default": True,
                            },
                        },
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
                elif name == "get_model_status":
                    return await self._get_model_status(arguments)
                elif name == "convert_pdf_to_images":
                    return await self._convert_pdf_to_images(arguments)
                else:
                    return [TextContent(type="text", text=f"❌ Outil inconnu: {name}")]

            except Exception as e:
                logger.exception(f"Erreur lors de l'exécution de l'outil {name}")
                return [TextContent(type="text", text=f"❌ Erreur: {str(e)}")]

    def _find_common_prefix(self, strings: list[str]) -> str:
        """Trouve le préfixe commun le plus long dans une liste de chaînes."""
        if not strings:
            return ""

        # Trier pour faciliter la comparaison
        strings = sorted(strings)
        first = strings[0]
        last = strings[-1]

        # Trouver le préfixe commun entre le premier et le dernier
        i = 0
        while i < len(first) and i < len(last) and first[i] == last[i]:
            i += 1

        # Retourner le préfixe commun
        prefix = first[:i]

        # Si le préfixe se termine par un séparateur incomplet, le retirer
        if prefix and prefix[-1] in ["_", "-", "."]:
            prefix = prefix[:-1]

        return prefix

    async def _search_screenshots(self, args: dict[str, Any]) -> list[TextContent | ImageContent | EmbeddedResource]:
        """Recherche des screenshots pertinents dans les index sélectionnés."""
        query = args["query"]
        top_k = args.get("top_k", 5)
        max_indices = args.get("max_indices", 2)

        logger.info(f"🔍 Recherche: '{query}' (top_k={top_k}, max_indices={max_indices})")

        # Acquérir le modèle ColPali pour cette opération
        manager = get_colpali_manager()
        colpali_model = manager.acquire(self.colpali_model_path)

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
            query_embedding = colpali_model.generate_query_embedding(query)

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
        finally:
            # Libérer le modèle
            colpali_model.cleanup()

    async def _index_screenshots(self, args: dict[str, Any]) -> list[TextContent | ImageContent | EmbeddedResource]:
        """Indexe les screenshots depuis le dossier spécifié dans UN SEUL index."""
        screenshots_dir = args.get("screenshots_dir", "./screenshots")
        overwrite_existing = args.get("overwrite_existing", False)
        batch_size = args.get("batch_size", 1)
        custom_index_name = args.get("index_name")

        logger.info(f"📥 Indexation des screenshots depuis: {screenshots_dir}")

        # Acquérir le modèle ColPali pour cette opération
        manager = get_colpali_manager()
        colpali_model = manager.acquire(self.colpali_model_path)

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

            # TOUJOURS regrouper toutes les images sous un seul index
            # Générer le nom de l'index
            if custom_index_name:
                index_base_name = custom_index_name
            else:
                # Utiliser le nom du dossier ou un nom par défaut
                folder_name = Path(screenshots_dir).name

                if folder_name == "screenshots":
                    # Chercher un pattern commun dans les noms de fichiers
                    all_sources = list(screenshots_by_source.keys())

                    if len(all_sources) == 1:
                        # Une seule source détectée, l'utiliser
                        index_base_name = all_sources[0]
                    else:
                        # Plusieurs sources, essayer de trouver un préfixe commun
                        common_prefix = self._find_common_prefix(all_sources)
                        if common_prefix and len(common_prefix) > 2:
                            index_base_name = common_prefix.rstrip("_-")
                        else:
                            # Pas de préfixe commun, utiliser un nom générique avec timestamp
                            from datetime import datetime

                            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                            index_base_name = f"all_screenshots_{timestamp}"
                else:
                    index_base_name = folder_name

            # Nettoyer et normaliser le nom d'index
            index_base_name = re.sub(r"[^a-zA-Z0-9\._-]", "_", index_base_name).lower()
            index_name = f"screenshot_{index_base_name}"

            # Regrouper TOUTES les images dans une seule liste
            all_images_metadata = []
            for source, images_metadata in screenshots_by_source.items():
                for img, metadata in images_metadata:
                    # Ajouter la source originale dans les métadonnées
                    metadata["original_source"] = source
                    all_images_metadata.append((img, metadata))

            logger.info(f"📂 Toutes les images ({len(all_images_metadata)}) seront dans l'index unique '{index_name}'")

            # Statistiques
            total_images = len(all_images_metadata)
            summary = f"📊 Indexation de {total_images} screenshots dans l'index '{index_name}'\n\n"

            # Vérifier si l'index existe
            if self.es_model.es and self.es_model.es.indices.exists(index=index_name):
                if overwrite_existing:
                    logger.info(f"🗑️ Suppression de l'index existant: {index_name}")
                    self.es_model.es.indices.delete(index=index_name)
                else:
                    return [
                        TextContent(
                            type="text",
                            text=f"⚠️ L'index '{index_name}' existe déjà. Utilisez overwrite_existing=true pour le remplacer.",
                        )
                    ]

            # Créer une instance ES pour cet index unique
            es_instance = ElasticsearchModel(
                index_name=index_name,
                es_host=self.es_model.es_host,
                es_user=self.es_model.es_user,
                es_password=self.es_model.es_password,
            )

            # Indexer toutes les images
            indexed_count = 0

            # Extraire toutes les images
            all_images = [img for img, _ in all_images_metadata]
            all_metadata = [metadata for _, metadata in all_images_metadata]

            # Traiter par batch
            for i in range(0, len(all_images), batch_size):
                batch_images = all_images[i : i + batch_size]
                batch_metadata = all_metadata[i : i + batch_size]

                logger.info(
                    f"🧮 Génération des embeddings pour le batch {i//batch_size + 1}/{(len(all_images) + batch_size - 1)//batch_size}"
                )

                try:
                    # Générer tous les embeddings du batch
                    embeddings = colpali_model.generate_embeddings(batch_images)

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
                            "source": metadata.get("original_source", "unknown"),
                        }

                        batch_documents.append(doc)

                    # Indexer ce batch
                    if batch_documents:
                        success = es_instance.bulk_index_documents(batch_documents)
                        if success:
                            indexed_count += len(batch_documents)
                            logger.info(f"💾 Batch indexé: {len(batch_documents)} images")
                        else:
                            logger.error(f"❌ Échec de l'indexation du batch {i//batch_size + 1}")

                except Exception as e:
                    logger.error(f"❌ Erreur lors du traitement du batch {i//batch_size + 1}: {e}")
                    continue

            es_instance.close()

            # Résumé final
            summary += f"✅ Index créé: {index_name}\n"
            summary += f"📷 Images indexées: {indexed_count}/{total_images}\n"

            if indexed_count < total_images:
                summary += f"⚠️ {total_images - indexed_count} images n'ont pas pu être indexées\n"

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
        finally:
            # Libérer le modèle
            colpali_model.cleanup()

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

    async def _get_model_status(self, args: dict[str, Any]) -> list[TextContent | ImageContent | EmbeddedResource]:
        """Obtient l'état actuel du modèle ColPali."""
        try:
            manager = get_colpali_manager()
            info = manager.get_model_info()

            status = "🤖 État du modèle ColPali\n\n"

            if info["loaded"]:
                status += "✅ État: CHARGÉ\n"
                status += f"📦 Modèle: {info['model_path']}\n"
                status += f"🔗 Références actives: {info['reference_count']}\n"

                if info.get("gpu_memory_allocated"):
                    status += f"💾 Mémoire GPU: {info['gpu_memory_allocated']}\n"

                if info["last_used"] is not None:
                    status += f"⏱️ Dernière utilisation: il y a {info['last_used']:.1f} secondes\n"

                status += "\n💡 Le modèle sera automatiquement déchargé après 30s d'inactivité"
            else:
                status += "💤 État: DÉCHARGÉ\n"
                status += "💡 Le modèle sera chargé automatiquement lors de la prochaine utilisation"

            return [TextContent(type="text", text=status)]

        except Exception as e:
            logger.error(f"Erreur lors de la récupération du statut: {e}")
            return [TextContent(type="text", text=f"❌ Erreur: {str(e)}")]

    async def run(self, read_stream: Any, write_stream: Any) -> None:
        """Lance le serveur MCP."""
        logger.info("🌐 Démarrage du serveur Image RAG...")

        try:
            # Démarrer la tâche de nettoyage périodique
            self.cleanup_task = asyncio.create_task(self._periodic_cleanup())

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

            # Annuler la tâche de nettoyage
            if self.cleanup_task:
                self.cleanup_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self.cleanup_task

            try:
                # Forcer le déchargement du modèle s'il est encore chargé
                manager = get_colpali_manager()
                if manager.is_loaded:
                    logger.info("🧹 Déchargement du modèle ColPali...")
                    manager._unload_model()

                if hasattr(self, "es_model"):
                    self.es_model.close()
            except Exception as e:
                logger.warning(f"Avertissement lors du nettoyage: {e}")

    async def _convert_pdf_to_images(
        self, args: dict[str, Any]
    ) -> list[TextContent | ImageContent | EmbeddedResource]:
        """Convertit des PDFs en images."""
        pdf_path = args.get("pdf_path")
        pdf_directory = args.get("pdf_directory")
        output_directory = args.get("output_directory", "./screenshots")
        dpi = args.get("dpi", 200)
        image_format = args.get("image_format", "png")
        clear_output = args.get("clear_output", False)
        single_file_per_pdf = args.get("single_file_per_pdf", True)

        # Vérifier qu'au moins une source est fournie
        if not pdf_path and not pdf_directory:
            return [
                TextContent(
                    type="text",
                    text="❌ Veuillez fournir soit 'pdf_path' pour un fichier unique, soit 'pdf_directory' pour un dossier de PDFs.",
                )
            ]

        try:
            converter = PDFConverter()
            summary = "📄 Conversion PDF vers images\n\n"

            if pdf_directory:
                # Convertir un dossier entier
                logger.info(f"📂 Conversion du dossier: {pdf_directory}")

                results = converter.convert_pdfs_directory(
                    input_dir=pdf_directory,
                    output_dir=output_directory,
                    dpi=dpi,
                    image_format=image_format,
                    single_file_per_pdf=single_file_per_pdf,
                    clear_output=clear_output,
                )

                total_pages = sum(results.values())
                successful_pdfs = sum(1 for pages in results.values() if pages > 0)

                summary += "📊 Statistiques de conversion:\n"
                summary += f"   📁 Dossier source: {pdf_directory}\n"
                summary += f"   📁 Dossier de sortie: {output_directory}\n"
                summary += f"   📄 PDFs traités: {len(results)}\n"
                summary += f"   ✅ Conversions réussies: {successful_pdfs}\n"
                summary += f"   📑 Total de pages converties: {total_pages}\n"
                summary += f"   🎨 Format: {image_format.upper()} ({dpi} DPI)\n\n"

                summary += "📋 Détails par fichier:\n"
                for pdf_name, num_pages in sorted(results.items()):
                    if num_pages > 0:
                        summary += f"   ✅ {pdf_name}: {num_pages} pages\n"
                    else:
                        summary += f"   ❌ {pdf_name}: échec\n"

            else:
                # Convertir un fichier unique
                logger.info(f"📄 Conversion du fichier: {pdf_path}")

                if pdf_path is None:
                    return [
                        TextContent(
                            type="text",
                            text="❌ Le paramètre 'pdf_path' ne peut pas être None",
                        )
                    ]

                num_pages, image_paths = converter.convert_pdf_to_images(
                    pdf_path=pdf_path,
                    output_dir=output_directory,
                    dpi=dpi,
                    image_format=image_format,
                    single_file=single_file_per_pdf,
                )

                summary += "✅ Conversion réussie!\n"
                summary += f"   📄 Fichier source: {pdf_path}\n"
                summary += f"   📁 Dossier de sortie: {output_directory}\n"
                summary += f"   📑 Pages converties: {num_pages}\n"
                summary += f"   🎨 Format: {image_format.upper()} ({dpi} DPI)\n\n"

                if len(image_paths) <= 10:
                    summary += "📷 Images créées:\n"
                    for img_path in image_paths:
                        summary += f"   - {os.path.basename(img_path)}\n"
                else:
                    summary += f"📷 {len(image_paths)} images créées\n"

            summary += (
                f"\n💡 Utilisez 'index_screenshots' avec screenshots_dir='{output_directory}' pour indexer ces images."
            )

            return [TextContent(type="text", text=summary)]

        except FileNotFoundError as e:
            return [
                TextContent(
                    type="text",
                    text=f"❌ Fichier/Dossier introuvable: {str(e)}",
                )
            ]
        except Exception as e:
            logger.error(f"Erreur lors de la conversion PDF: {e}")
            logger.error(traceback.format_exc())
            return [
                TextContent(
                    type="text",
                    text=f"❌ Erreur lors de la conversion: {str(e)}",
                )
            ]


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
