"""Utilitaires pour le serveur MCP Image RAG."""

import logging
import tempfile
from pathlib import Path
from typing import Any

from pdf2image import convert_from_path
from PIL import Image

from .colpali_model import ColPaliModel
from .elasticsearch_model import ElasticsearchModel

logger = logging.getLogger(__name__)


class BatchIndexer:
    """Utilitaire pour indexer des documents en batch."""

    def __init__(
        self,
        es_model: ElasticsearchModel,
        colpali_model: ColPaliModel,
        batch_size: int = 10,
    ):
        self.es_model = es_model
        self.colpali_model = colpali_model
        self.batch_size = batch_size

    async def index_pdf(
        self,
        pdf_path: str,
        output_dir: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Indexe toutes les pages d'un PDF.

        Args:
            pdf_path: Chemin vers le fichier PDF
            output_dir: Dossier pour sauvegarder les images extraites
            metadata: Métadonnées à ajouter à chaque page

        Returns:
            Nombre de pages indexées
        """
        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            raise FileNotFoundError(f"PDF non trouvé: {pdf_path_obj}")

        # Créer le dossier de sortie si nécessaire
        output_dir_obj: Path
        if output_dir:
            output_dir_obj = Path(output_dir)
            output_dir_obj.mkdir(parents=True, exist_ok=True)
        else:
            # Utiliser un dossier temporaire
            output_dir_obj = Path(tempfile.mkdtemp())

        logger.info(f"📄 Extraction des pages du PDF: {pdf_path_obj}")

        try:
            # Convertir le PDF en images
            images = convert_from_path(str(pdf_path_obj))
            logger.info(f"✅ {len(images)} pages extraites")

            # Préparer les données pour l'indexation en batch
            indexed_count = 0

            # Traiter par batch
            for i in range(0, len(images), self.batch_size):
                batch_images = images[i : i + self.batch_size]
                batch_data = []

                # Préparer les métadonnées et sauvegarder les images du batch
                batch_image_objects: list[Image.Image | str] = []
                for j, image in enumerate(batch_images):
                    page_num = i + j + 1

                    # Sauvegarder l'image
                    image_filename = f"{pdf_path_obj.stem}_page_{page_num:03d}.png"
                    image_path = output_dir_obj / image_filename
                    image.save(str(image_path), "PNG")

                    # Préparer les métadonnées
                    page_metadata = metadata.copy() if metadata else {}
                    page_metadata.update(
                        {
                            "total_pages": len(images),
                            "pdf_filename": pdf_path_obj.name,
                            "page_number": page_num,
                        }
                    )

                    batch_image_objects.append(image)
                    batch_data.append(
                        {
                            "image_path": str(image_path),
                            "source_file": pdf_path_obj.name,
                            "page_number": page_num,
                            "metadata": page_metadata,
                        }
                    )

                logger.info(
                    f"🧮 Génération des embeddings pour le batch {i//self.batch_size + 1}/{(len(images) + self.batch_size - 1)//self.batch_size}"
                )

                # Générer tous les embeddings du batch en une fois
                embeddings = self.colpali_model.generate_embeddings(batch_image_objects)

                # Ajouter les embeddings aux données
                for k, embedding in enumerate(embeddings):
                    if k < len(batch_data):
                        batch_data[k]["embedding"] = embedding

                # Indexer le batch
                count = self.es_model.bulk_index_images(batch_data)
                indexed_count += count
                logger.info(f"💾 Batch indexé: {count} pages")

            logger.info(f"✅ Indexation terminée: {indexed_count}/{len(images)} pages")
            return indexed_count

        except Exception as e:
            logger.error(f"❌ Erreur lors de l'indexation du PDF: {e}")
            raise

    async def index_directory(
        self,
        directory: str,
        extensions: list[str] | None = None,
        recursive: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, int]:
        """Indexe tous les fichiers d'un répertoire.

        Args:
            directory: Répertoire à parcourir
            extensions: Extensions de fichiers à traiter
            recursive: Parcourir récursivement
            metadata: Métadonnées à ajouter

        Returns:
            Dictionnaire avec le nombre de fichiers traités par type
        """
        if extensions is None:
            extensions = [".pdf", ".png", ".jpg", ".jpeg"]
        directory_obj = Path(directory)
        if not directory_obj.exists():
            raise FileNotFoundError(f"Répertoire non trouvé: {directory_obj}")

        stats = {ext: 0 for ext in extensions}
        stats["errors"] = 0

        # Trouver tous les fichiers
        pattern = "**/*" if recursive else "*"
        files: list[Path] = []
        for ext in extensions:
            files.extend(directory_obj.glob(f"{pattern}{ext}"))

        logger.info(f"📁 {len(files)} fichiers trouvés dans {directory_obj}")

        # Séparer les PDFs des images pour un traitement optimisé
        pdf_files = [f for f in files if f.suffix.lower() == ".pdf"]
        image_files = [f for f in files if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}]

        # Traiter les PDFs
        for file_path in pdf_files:
            try:
                file_metadata = metadata.copy() if metadata else {}
                file_metadata["directory"] = str(directory_obj)
                file_metadata["relative_path"] = str(file_path.relative_to(directory_obj))

                await self.index_pdf(str(file_path), metadata=file_metadata)
                stats[".pdf"] += 1

            except Exception as e:
                logger.error(f"❌ Erreur avec {file_path}: {e}")
                stats["errors"] += 1

        # Traiter les images par batch
        if image_files:
            for i in range(0, len(image_files), self.batch_size):
                batch_files = image_files[i : i + self.batch_size]

                try:
                    # Charger toutes les images du batch
                    batch_images: list[Image.Image | str] = []
                    batch_metadata = []

                    for file_path in batch_files:
                        with Image.open(file_path) as img:
                            pil_image = img.copy()
                            if pil_image.mode != "RGB":
                                pil_image = pil_image.convert("RGB")

                        file_metadata = metadata.copy() if metadata else {}
                        file_metadata["directory"] = str(directory_obj)
                        file_metadata["relative_path"] = str(file_path.relative_to(directory_obj))

                        batch_images.append(pil_image)
                        batch_metadata.append(
                            {
                                "image_path": str(file_path),
                                "source_file": file_path.name,
                                "metadata": file_metadata,
                                "extension": file_path.suffix.lower(),
                            }
                        )

                    # Générer tous les embeddings du batch
                    embeddings = self.colpali_model.generate_embeddings(batch_images)

                    # Indexer chaque image
                    for j, (file_path, embedding) in enumerate(zip(batch_files, embeddings, strict=False)):
                        if j < len(batch_metadata):
                            doc_id = self.es_model.index_image(
                                image_path=str(file_path),
                                embedding=embedding,
                                source_file=file_path.name,
                                metadata=batch_metadata[j]["metadata"],
                            )

                            if doc_id:
                                ext = file_path.suffix.lower()
                                stats[ext] = stats.get(ext, 0) + 1
                            else:
                                stats["errors"] += 1

                except Exception as e:
                    logger.error(f"❌ Erreur avec le batch {i//self.batch_size + 1}: {e}")
                    stats["errors"] += len(batch_files)

        return stats


class SearchAnalyzer:
    """Analyseur de résultats de recherche."""

    @staticmethod
    def analyze_results(results: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyse les résultats de recherche.

        Args:
            results: Liste des résultats de recherche

        Returns:
            Dictionnaire avec les statistiques
        """
        if not results:
            return {
                "count": 0,
                "avg_score": 0,
                "score_distribution": {},
                "sources": {},
                "pages": {},
            }

        scores = [r["score"] for r in results]
        sources: dict[str, int] = {}
        pages: dict[str, int] = {}

        for result in results:
            # Compter par source
            source = result.get("source_file", "unknown")
            sources[source] = sources.get(source, 0) + 1

            # Distribution des pages
            page = result.get("page_number", 0)
            page_range = f"{(page-1)//10*10+1}-{(page-1)//10*10+10}"
            pages[page_range] = pages.get(page_range, 0) + 1

        # Distribution des scores
        score_distribution = {
            "0.9-1.0": sum(1 for s in scores if 0.9 <= s <= 1.0),
            "0.8-0.9": sum(1 for s in scores if 0.8 <= s < 0.9),
            "0.7-0.8": sum(1 for s in scores if 0.7 <= s < 0.8),
            "0.6-0.7": sum(1 for s in scores if 0.6 <= s < 0.7),
            "<0.6": sum(1 for s in scores if s < 0.6),
        }

        return {
            "count": len(results),
            "avg_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "score_distribution": score_distribution,
            "sources": sources,
            "page_distribution": pages,
        }

    @staticmethod
    def format_analysis(analysis: dict[str, Any]) -> str:
        """Formate l'analyse pour l'affichage."""
        if analysis["count"] == 0:
            return "Aucun résultat à analyser"

        text = "📊 Analyse des résultats\n"
        text += f"{'='*40}\n"
        text += f"Nombre de résultats: {analysis['count']}\n"
        text += f"Score moyen: {analysis['avg_score']:.3f}\n"
        text += f"Score min/max: {analysis['min_score']:.3f} / {analysis['max_score']:.3f}\n\n"

        text += "Distribution des scores:\n"
        for range_str, count in analysis["score_distribution"].items():
            if count > 0:
                text += f"  {range_str}: {'█' * count} ({count})\n"

        text += f"\nSources ({len(analysis['sources'])}):\n"
        for source, count in sorted(analysis["sources"].items(), key=lambda x: x[1], reverse=True)[:5]:
            text += f"  - {source}: {count}\n"

        if len(analysis["sources"]) > 5:
            text += f"  ... et {len(analysis['sources']) - 5} autres\n"

        return text


# Fonction utilitaire pour tester la connexion
async def test_connection(
    es_host: str | None = None,
    es_user: str | None = None,
    es_password: str | None = None,
    index_name: str = "image_embeddings",
) -> bool:
    """Teste la connexion à Elasticsearch.

    Returns:
        True si la connexion est établie
    """
    es_model = None
    try:
        es_model = ElasticsearchModel(
            index_name=index_name,
            es_host=es_host,
            es_user=es_user,
            es_password=es_password,
        )

        stats = es_model.get_stats()

        if stats.get("status") == "connected":
            logger.info("✅ Connexion établie")
            logger.info(f"📊 Index: {stats.get('index_name')}")
            logger.info(f"📚 Documents: {stats.get('document_count', 0)}")
            logger.info(f"🔧 Version ES: {stats.get('elasticsearch_version')}")
            return True
        else:
            logger.error("❌ Connexion échouée")
            return False

    except Exception as e:
        logger.error(f"❌ Erreur de connexion: {e}")
        return False
    finally:
        if es_model is not None:
            es_model.close()
