"""Modèle ColPali pour la génération d'embeddings d'images et de requêtes."""

import gc
import logging
import time
from collections.abc import Sequence
from threading import Lock
from typing import Any, Optional

import torch
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available

try:
    from colpali_engine.models import ColQwen2, ColQwen2Processor
except ImportError:
    ColQwen2 = None
    ColQwen2Processor = None


logger = logging.getLogger(__name__)


class ColPaliModelManager:
    """Gestionnaire singleton pour le modèle ColPali avec chargement/déchargement dynamique."""

    _instance: Optional["ColPaliModelManager"] = None
    _lock = Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> "ColPaliModelManager":
        """Crée ou retourne l'instance singleton du gestionnaire ColPali.

        Args:
            *args: Arguments positionnels (ignorés).
            **kwargs: Arguments nommés (ignorés).

        Returns:
            ColPaliModelManager: L'instance unique du gestionnaire.
        """
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialise le gestionnaire ColPali (une seule fois pour le singleton)."""
        # Ne réinitialiser que si c'est la première fois
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self._model: ColQwen2 | None = None
            self._processor: ColQwen2Processor | None = None
            self._model_path: str | None = None
            self._last_used: float = 0
            self._load_lock = Lock()
            self._reference_count = 0
            self._unload_delay = 0  # Délai en secondes avant déchargement

    def acquire(self, model_path: str = "vidore/colqwen2-v1.0") -> "ColPaliModel":
        """Acquiert une référence au modèle et le charge si nécessaire.

        Args:
            model_path (str): Chemin vers le modèle ColPali à charger.

        Returns:
            ColPaliModel: Instance du modèle prête à utiliser.
        """
        with self._load_lock:
            self._reference_count += 1
            self._last_used = time.time()

            # Charger le modèle si pas déjà chargé ou si le chemin a changé
            if self._model is None or self._model_path != model_path:
                self._load_model(model_path)

            return ColPaliModel(self)

    def release(self) -> None:
        """Libère une référence au modèle."""
        with self._load_lock:
            self._reference_count = max(0, self._reference_count - 1)
            self._last_used = time.time()

            # Ne pas décharger immédiatement si d'autres références existent
            if self._reference_count == 0:
                self._unload_model()
                return

            # Programmer le déchargement après le délai
            # Note: Dans une vraie implémentation, on utiliserait un timer thread
            # Pour simplifier, on laisse la responsabilité à check_and_unload()

    def check_and_unload(self) -> None:
        """Vérifie si le modèle doit être déchargé (appelé périodiquement)."""
        with self._load_lock:
            if (
                self._model is not None
                and self._reference_count == 0
                and time.time() - self._last_used > self._unload_delay
            ):
                self._unload_model()

    def _load_model(self, model_path: str) -> None:
        """Charge le modèle en mémoire.

        Args:
            model_path (str): Chemin vers le modèle à charger.

        Raises:
            ImportError: Si colpali_engine n'est pas disponible.
            Exception: Si le chargement échoue.
        """
        if ColQwen2 is None or ColQwen2Processor is None:
            raise ImportError("colpali_engine n'est pas disponible")

        # Décharger l'ancien modèle si nécessaire
        if self._model is not None:
            self._unload_model()

        try:
            logger.info(f"📦 Chargement du modèle {model_path} sur GPU...")
            start_time = time.time()

            # Déterminer le device
            device = self._get_device()

            if device == "cpu":
                # Chargement CPU
                self._model = ColQwen2.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                ).eval()
            else:
                # Chargement GPU avec optimisations mémoire
                self._model = ColQwen2.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    attn_implementation=("flash_attention_2" if is_flash_attn_2_available() else None),
                    low_cpu_mem_usage=True,
                    max_memory={0: "4.5GiB"},
                ).eval()

            # Charger le processeur
            self._processor = ColQwen2Processor.from_pretrained(model_path)
            self._model_path = model_path

            # Nettoyage du cache
            if device != "cpu" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            load_time = time.time() - start_time
            logger.info(f"✅ Modèle chargé en {load_time:.2f}s")

            # Afficher l'utilisation mémoire
            if device != "cpu" and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                logger.info(f"📊 Mémoire GPU utilisée: {allocated:.2f}GB")

        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement du modèle: {e}")
            self._model = None
            self._processor = None
            self._model_path = None
            raise

    def _unload_model(self) -> None:
        """Décharge le modèle de la mémoire GPU directement."""
        if self._model is None:
            return

        try:
            logger.info("🧹 Déchargement du modèle de la GPU...")

            # Supprimer les références
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            self._model_path = None

            # Forcer le garbage collection Python
            logger.info("   -> Garbage collection...")
            gc.collect()

            # Vider le cache CUDA de manière agressive
            if torch.cuda.is_available():
                logger.info("   -> Vidage agressif du cache CUDA...")

                # Plusieurs passes de nettoyage
                for _ in range(3):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                # Essayer de réinitialiser le contexte CUDA si possible
                try:
                    # Réinitialiser l'allocateur de mémoire (PyTorch >= 1.10)
                    if hasattr(torch.cuda, "memory._set_allocator_settings"):
                        torch.cuda.memory._set_allocator_settings("")

                    # Forcer la libération de toute la mémoire réservée
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.reset_accumulated_memory_stats()

                    # Si disponible, forcer un reset plus profond
                    if hasattr(torch.cuda, "cudart"):
                        torch.cuda.cudart().cudaDeviceReset()
                except Exception as e:
                    logger.debug(f"Méthodes de reset CUDA non disponibles: {e}")

                # Afficher la mémoire après nettoyage
                after_allocated = torch.cuda.memory_allocated(0) / 1024**3
                after_reserved = torch.cuda.memory_reserved(0) / 1024**3

                logger.info("✅ Modèle déchargé:")
                logger.info(f"📊 Mémoire GPU finale: Allouée={after_allocated:.2f}GB, Réservée={after_reserved:.2f}GB")

                if after_reserved > 0.5:  # Plus de 500MB encore réservés
                    logger.warning(
                        f"⚠️ {after_reserved:.2f}GB de mémoire encore réservée. "
                        "PyTorch garde la mémoire en cache. Définissez PYTORCH_NO_CUDA_MEMORY_CACHING=1 "
                        "pour forcer la libération complète."
                    )
            else:
                logger.info("✅ Modèle déchargé")

        except Exception as e:
            logger.error(f"❌ Erreur lors du déchargement: {e}")
            # Même en cas d'erreur, s'assurer que les références sont nulles
            self._model = None
            self._processor = None
            self._model_path = None

            # Tenter quand même un nettoyage basique
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def force_unload(self) -> None:
        """Force le déchargement immédiat du modèle, ignore les références."""
        with self._load_lock:
            logger.info("⚠️ Déchargement forcé du modèle...")
            self._reference_count = 0
            self._unload_model()

    def _get_device(self) -> str:
        """Détermine le meilleur device disponible.

        Returns:
            str: Le device à utiliser ("cuda:0" ou "cpu").
        """
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            gpu_mem = gpu_props.total_memory
            allocated_mem = torch.cuda.memory_allocated(0)
            free_mem = gpu_mem - allocated_mem

            if free_mem >= 4 * 1024 * 1024 * 1024:
                return "cuda:0"
            else:
                logger.warning("⚠️ Mémoire GPU insuffisante, utilisation du CPU")
                return "cpu"
        else:
            logger.info("ℹ️ CUDA non disponible, utilisation du CPU")
            return "cpu"

    @property
    def is_loaded(self) -> bool:
        """Vérifie si le modèle est chargé.

        Returns:
            bool: True si le modèle est chargé en mémoire.
        """
        return self._model is not None

    def get_model_info(self) -> dict[str, Any]:
        """Retourne des informations sur le modèle.

        Returns:
            dict[str, Any]: Dictionnaire contenant les informations du modèle.
        """
        info = {
            "loaded": self.is_loaded,
            "model_path": self._model_path,
            "reference_count": self._reference_count,
            "last_used": (time.time() - self._last_used if self._last_used > 0 else None),
        }

        if self.is_loaded and torch.cuda.is_available():
            info["gpu_memory_allocated"] = f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB"

        return info


class ColPaliModel:
    """Interface pour utiliser le modèle ColPali avec gestion automatique du cycle de vie."""

    def __init__(self, manager: ColPaliModelManager):
        """Initialise avec une référence au gestionnaire.

        Args:
            manager (ColPaliModelManager): Le gestionnaire de modèle à utiliser.
        """
        self._manager = manager
        self.batch_size = 1

    def __enter__(self) -> "ColPaliModel":
        """Contexte manager pour acquisition automatique.

        Returns:
            ColPaliModel: L'instance elle-même pour le context manager.
        """
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Libère automatiquement la référence.

        Args:
            exc_type: Type d'exception (si levée).
            exc_val: Valeur de l'exception (si levée).
            exc_tb: Traceback de l'exception (si levée).
        """
        self._manager.release()

    @property
    def device(self) -> str:
        """Retourne le device du modèle.

        Returns:
            str: Le device utilisé par le modèle.
        """
        if self._manager._model is None:
            return "cpu"
        return str(self._manager._model.device)

    def generate_embeddings(self, images: Sequence[Image.Image | str]) -> list[list[float]]:
        """Génère des embeddings pour une liste d'images.

        Args:
            images (Sequence[Image.Image | str]): Liste d'images ou de chemins vers les images.

        Returns:
            list[list[float]]: Liste des embeddings générés.

        Raises:
            RuntimeError: Si le modèle n'est pas chargé.
        """
        if not images:
            return []

        if not self._manager.is_loaded:
            raise RuntimeError("Le modèle n'est pas chargé. Utilisez ColPaliModelManager.acquire()")

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
                with torch.no_grad():
                    # Vérifier que le processeur et le modèle sont chargés
                    if self._manager._processor is None or self._manager._model is None:
                        raise RuntimeError("Le modèle ou le processeur n'est pas chargé")

                    # Traiter les images du batch
                    batch_doc = self._manager._processor.process_images(batch_images)
                    batch_doc = {k: v.to(self._manager._model.device) for k, v in batch_doc.items()}

                    # Générer les embeddings
                    batch_embeddings = self._manager._model(**batch_doc)

                    # Moyenne sur la dimension des patches pour chaque image
                    for j in range(batch_embeddings.shape[0]):
                        mean_embedding = torch.mean(batch_embeddings[j], dim=0).float().cpu().numpy()
                        embeddings.append(mean_embedding.tolist())

                    # Libérer la mémoire GPU après chaque batch
                    del batch_embeddings
                    del batch_doc

                    if self.device != "cpu" and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()

            except torch.cuda.OutOfMemoryError:
                logger.error(f"❌ Mémoire GPU insuffisante pour le batch {i//self.batch_size + 1}")
                if self.device != "cpu" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                # Ajouter des embeddings vides
                for _ in batch:
                    embeddings.append([0.0] * 128)

            except Exception as e:
                logger.error(f"❌ Erreur lors de la génération de l'embedding: {e}")
                for _ in batch:
                    embeddings.append([0.0] * 128)

        return embeddings

    def generate_single_embedding(self, image: Image.Image | str) -> list[float]:
        """Génère un embedding pour une seule image.

        Args:
            image (Image.Image | str): Image ou chemin vers l'image.

        Returns:
            list[float]: Embedding généré pour l'image.
        """
        embeddings = self.generate_embeddings([image])
        return embeddings[0] if embeddings else [0.0] * 128

    def generate_query_embedding(self, query: str) -> list[float]:
        """Génère un embedding pour une requête textuelle.

        Args:
            query (str): Requête textuelle à traiter.

        Returns:
            list[float]: Embedding généré pour la requête.

        Raises:
            RuntimeError: Si le modèle n'est pas chargé.
        """
        if not self._manager.is_loaded:
            raise RuntimeError("Le modèle n'est pas chargé")

        try:
            with torch.no_grad():
                # Vérifier que le processeur et le modèle sont chargés
                if self._manager._processor is None or self._manager._model is None:
                    raise RuntimeError("Le modèle ou le processeur n'est pas chargé")

                batch_doc = self._manager._processor.process_queries([query]).to(self._manager._model.device)
                embeddings = self._manager._model(**batch_doc)

                # Moyenne sur la dimension des tokens
                mean_embedding = torch.mean(embeddings, dim=1).float().cpu().numpy()[0]

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
        """Alias pour generate_embeddings.

        Args:
            images (Sequence[Image.Image | str]): Liste d'images ou de chemins vers les images.

        Returns:
            list[list[float]]: Liste des embeddings générés.
        """
        return self.generate_embeddings(images)

    def cleanup(self) -> None:
        """Libère la référence au modèle."""
        self._manager.release()

    def get_model_info(self) -> dict[str, Any]:
        """Retourne des informations sur le modèle.

        Returns:
            dict[str, Any]: Dictionnaire contenant les informations du modèle.
        """
        info = self._manager.get_model_info()
        info["batch_size"] = self.batch_size
        info["embedding_dimension"] = 128
        return info


# Fonction utilitaire pour obtenir le gestionnaire singleton
def get_colpali_manager() -> ColPaliModelManager:
    """Retourne l'instance singleton du gestionnaire ColPali.

    Returns:
        ColPaliModelManager: L'instance unique du gestionnaire ColPali.
    """
    return ColPaliModelManager()
