"""Modèle ColPali pour la génération d'embeddings d'images et de requêtes."""

import logging
from collections.abc import Sequence
from typing import Any
import gc

import torch
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available

# Imports avec annotations de type ignore pour les modules non typés
try:
    from colpali_engine.models import ColQwen2, ColQwen2Processor
except ImportError:
    # Fallback pour les tests ou environnements sans colpali_engine
    ColQwen2 = None
    ColQwen2Processor = None


logger = logging.getLogger(__name__)


class ColPaliModel:
    """Gère le modèle ColPali pour la génération d'embeddings."""

    def __init__(self, model_path: str = "vidore/colqwen2-v1.0", batch_size: int = 1):
        """Initialise le modèle ColPali.

        Args:
            model_path: Chemin vers le modèle préentraîné
            batch_size: Taille du batch pour le traitement
        """
        self.model_path = model_path
        self.batch_size = batch_size

        # Déterminer le device à utiliser
        self.device = self._get_device()
        logger.info(f"🖥️ Utilisation du device: {self.device}")

        # Charger le modèle
        self._load_model()

    def _get_device(self) -> str:
        """Détermine le meilleur device disponible."""
        if torch.cuda.is_available():
            # Vérifier la mémoire GPU disponible
            gpu_props = torch.cuda.get_device_properties(0)
            gpu_mem = gpu_props.total_memory  # type: ignore[attr-defined]
            allocated_mem = torch.cuda.memory_allocated(0)
            free_mem = gpu_mem - allocated_mem
            
            # Nécessite au moins 4GB de mémoire libre
            if free_mem >= 4 * 1024 * 1024 * 1024:
                return "cuda:0"
            else:
                logger.warning("⚠️ Mémoire GPU insuffisante, utilisation du CPU")
                return "cpu"
        else:
            logger.info("ℹ️ CUDA non disponible, utilisation du CPU")
            return "cpu"

    def _load_model(self) -> None:
        """Charge le modèle ColPali."""
        if ColQwen2 is None or ColQwen2Processor is None:
            raise ImportError("colpali_engine n'est pas disponible")

        try:
            logger.info(f"📦 Chargement du modèle {self.model_path}...")

            # Configuration selon le device
            if self.device == "cpu":
                # Chargement CPU
                self.model = ColQwen2.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                ).eval()
            else:
                # Chargement GPU avec optimisations mémoire
                self.model = ColQwen2.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    attn_implementation=("flash_attention_2" if is_flash_attn_2_available() else None),
                    low_cpu_mem_usage=True,  # Ajout pour économiser la mémoire
                    max_memory={0: "4.5GiB"},
                ).eval()

            # Charger le processeur
            self.processor = ColQwen2Processor.from_pretrained(self.model_path)

            # Nettoyage du cache
            if self.device != "cpu" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                # Afficher la mémoire GPU disponible
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                reserved = torch.cuda.memory_reserved(0) / 1024**3
                logger.info(f"📊 Mémoire GPU après chargement: {allocated:.2f}GB alloués, {reserved:.2f}GB réservés")

            logger.info("✅ Modèle ColPali chargé avec succès")

        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement du modèle: {e}")
            raise

    def generate_embeddings(self, images: Sequence[Image.Image | str]) -> list[list[float]]:
        """Génère des embeddings pour une liste d'images.

        Args:
            images: Liste d'images PIL ou de chemins vers des images

        Returns:
            Liste d'embeddings (128 dimensions chacun)
        """
        if not images:
            return []

        # Convertir en liste pour le traitement
        images_list = list(images)

        embeddings = []

        # Traiter par batch
        for i in range(0, len(images_list), self.batch_size):
            batch = images_list[i : i + self.batch_size]

            # Charger et préparer les images du batch
            batch_images = []
            for img in batch:
                if isinstance(img, str):
                    img = Image.open(img)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                batch_images.append(img)

            try:
                # Utiliser le processeur directement
                with torch.no_grad():
                    # Traiter les images du batch
                    batch_doc = self.processor.process_images(batch_images)
                    batch_doc = {k: v.to(self.model.device) for k, v in batch_doc.items()}
                    
                    # Générer les embeddings
                    batch_embeddings = self.model(**batch_doc)

                    # batch_embeddings shape: (batch_size, num_patches, hidden_dim)
                    # Moyenne sur la dimension des patches pour chaque image du batch
                    for j in range(batch_embeddings.shape[0]):
                        mean_embedding = torch.mean(batch_embeddings[j], dim=0).float().cpu().numpy()

                        # Vérifier la dimension
                        if len(mean_embedding) != 128:
                            logger.warning(f"⚠️ Dimension d'embedding inattendue: {len(mean_embedding)}, attendu: 128")

                        embeddings.append(mean_embedding.tolist())
                    
                    # IMPORTANT: Libérer la mémoire GPU après chaque batch
                    del batch_embeddings
                    del batch_doc
                    
                    if self.device != "cpu" and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        # Force garbage collection
                        gc.collect()

            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"❌ Mémoire GPU insuffisante pour le batch {i//self.batch_size + 1}")
                logger.error(f"   Essayez de réduire la taille du batch ou utilisez le CPU")
                # Nettoyer la mémoire GPU
                if self.device != "cpu" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                # Ajouter des embeddings vides pour maintenir la correspondance
                for _ in batch:
                    embeddings.append([0.0] * 128)
                    
            except Exception as e:
                logger.error(
                    f"❌ Erreur lors de la génération de l'embedding pour le batch {i//self.batch_size + 1}: {e}"
                )
                # Ajouter des embeddings vides pour maintenir la correspondance
                for _ in batch:
                    embeddings.append([0.0] * 128)

        return embeddings

    def generate_single_embedding(self, image: Image.Image | str) -> list[float]:
        """Génère un embedding pour une seule image (méthode de compatibilité).

        Args:
            image: Image PIL ou chemin vers une image

        Returns:
            Liste représentant l'embedding (128 dimensions)
        """
        embeddings = self.generate_embeddings([image])
        return embeddings[0] if embeddings else [0.0] * 128

    def generate_query_embedding(self, query: str) -> list[float]:
        """Génère un embedding pour une requête textuelle.

        Args:
            query: Requête textuelle

        Returns:
            Liste représentant l'embedding (128 dimensions)
        """
        try:
            with torch.no_grad():
                batch_doc = self.processor.process_queries([query]).to(self.model.device)
                embeddings = self.model(**batch_doc)

                # embeddings shape: (batch_size, num_tokens, hidden_dim)
                # On fait la moyenne sur la dimension des tokens (dim=1)
                mean_embedding = torch.mean(embeddings, dim=1).float().cpu().numpy()[0]

                # Vérifier la dimension
                if len(mean_embedding) != 128:
                    logger.warning(
                        f"⚠️ Dimension d'embedding de requête inattendue: {len(mean_embedding)}, attendu: 128"
                    )
                
                # Libérer la mémoire
                del embeddings
                del batch_doc
                
                if self.device != "cpu" and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            return mean_embedding.tolist()

        except Exception as e:
            logger.error(f"❌ Erreur lors de la génération de l'embedding de requête: {e}")
            raise

    def batch_generate_embeddings(self, images: Sequence[Image.Image | str]) -> list[list[float]]:
        """Alias pour generate_embeddings (méthode de compatibilité).

        Args:
            images: Liste d'images PIL ou de chemins

        Returns:
            Liste d'embeddings (128 dimensions chacun)
        """
        return self.generate_embeddings(images)

    def cleanup(self) -> None:
        """Nettoie les ressources du modèle."""
        try:
            # Libérer la mémoire GPU si utilisée
            if self.device != "cpu" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("🧹 Cache GPU vidé")

            # Supprimer les références au modèle
            if hasattr(self, "model"):
                del self.model
            if hasattr(self, "processor"):
                del self.processor
            
            # Force garbage collection
            gc.collect()

            logger.info("✅ Nettoyage du modèle terminé")

        except Exception as e:
            logger.warning(f"⚠️ Avertissement lors du nettoyage: {e}")

    def get_model_info(self) -> dict[str, Any]:
        """Retourne des informations sur le modèle."""
        info = {
            "model_path": self.model_path,
            "device": self.device,
            "batch_size": self.batch_size,
            "embedding_dimension": 128,
        }

        if torch.cuda.is_available() and self.device != "cpu":
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_allocated"] = f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB"
            info["gpu_memory_reserved"] = f"{torch.cuda.memory_reserved(0) / 1024**3:.2f} GB"
            info["gpu_memory_total"] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"

        return info