"""Modèle Elasticsearch pour le serveur MCP Image RAG."""

import logging
import os
import uuid
from collections.abc import Collection
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

logger = logging.getLogger(__name__)

load_dotenv()


class ElasticsearchModel:
    """Gère les interactions avec Elasticsearch pour l'index d'images."""

    def __init__(
        self,
        index_name: str = "image_embeddings",
        es_host: str | None = None,
        es_user: str | None = None,
        es_password: str | None = None,
    ):
        """Initialise la connexion Elasticsearch.

        Args:
            index_name: Nom de l'index
            es_host: Hôte Elasticsearch (défaut: depuis ES_HOST ou localhost:9200)
            es_user: Utilisateur (défaut: depuis ES_USER ou elastic)
            es_password: Mot de passe (défaut: depuis ES_PASSWORD)
        """
        self.index_name = index_name

        # Configuration de la connexion
        self.es_host = es_host or os.getenv("ES_HOST", "localhost:9200")
        self.es_user = es_user or os.getenv("ES_USER", "elastic")
        self.es_password = es_password or os.getenv("ES_PASSWORD", "")

        # URL de connexion
        if self.es_password:
            host_url = f"https://{self.es_host}"
            auth = (self.es_user, self.es_password)
        else:
            host_url = f"http://{self.es_host}"
            auth = None

        # Construire les arguments explicitement pour éviter les erreurs de type
        es_kwargs: dict[str, Any] = {
            "verify_certs": False,
            "ssl_show_warn": False,
            "request_timeout": 50,
            "headers": {
                "Accept": "application/vnd.elasticsearch+json; compatible-with=8",
                "Content-Type": "application/vnd.elasticsearch+json; compatible-with=8",
            },
        }

        if auth:
            es_kwargs["basic_auth"] = auth

        try:
            # Connexion à Elasticsearch
            self.es: Elasticsearch | None = Elasticsearch(host_url, **es_kwargs)

            # Vérifier la connexion
            if self.es.ping():
                es_info = self.es.info()
                logger.info(f"✅ Connecté à Elasticsearch version {es_info['version']['number']}")

                # Créer l'index si nécessaire
                self._create_index_if_not_exists()
            else:
                logger.error("❌ Impossible de se connecter à Elasticsearch")
                self.es = None

        except Exception as e:
            logger.error(f"❌ Erreur de connexion à Elasticsearch: {e}")
            self.es = None

    def _create_index_if_not_exists(self) -> None:
        """Crée l'index s'il n'existe pas."""
        if not self.es or self.es.indices.exists(index=self.index_name):
            return

        mappings = {
            "mappings": {
                "properties": {
                    "col_pali_vectors": {
                        "type": "dense_vector",
                        "dims": 128,
                        "index": True,
                        "similarity": "cosine",
                    },
                    "metadata": {"type": "object", "enabled": True},
                    "source_file": {"type": "keyword"},
                    "page_number": {"type": "integer"},
                    "image_path": {"type": "keyword"},
                    "image_base64": {"type": "text", "index": False},
                    "indexed_at": {"type": "date"},
                }
            }
        }

        try:
            self.es.indices.create(index=self.index_name, body=mappings)
            logger.info(f"✅ Index '{self.index_name}' créé")
        except Exception as e:
            logger.error(f"❌ Erreur lors de la création de l'index: {e}")

    def search_by_embedding(
        self, query_embedding: list[float], k: int = 5, min_score: float = 0.0
    ) -> list[dict[str, Any]]:
        """Recherche les documents les plus similaires.

        Args:
            query_embedding: Embedding de la requête
            k: Nombre de résultats à retourner
            min_score: Score minimum pour filtrer les résultats

        Returns:
            Liste des documents trouvés avec leurs scores
        """
        if not self.es:
            logger.error("Elasticsearch non disponible")
            return []

        try:
            # Requête KNN
            query = {
                "knn": {
                    "field": "col_pali_vectors",
                    "query_vector": query_embedding,
                    "k": k,
                    "num_candidates": k * 10,
                },
                "_source": {
                    "excludes": ["col_pali_vectors"]  # Exclure les vecteurs de la réponse
                },
            }

            # Exécuter la recherche
            response = self.es.search(index=self.index_name, body=query, size=k)

            # Traiter les résultats
            results = []
            for hit in response["hits"]["hits"]:
                if hit["_score"] >= min_score:
                    result = {
                        "id": hit["_id"],
                        "score": hit["_score"],
                        **hit["_source"],
                    }
                    results.append(result)

            logger.info(f"🔍 {len(results)} résultats trouvés")
            return results

        except Exception as e:
            logger.error(f"❌ Erreur lors de la recherche: {e}")
            return []

    def index_document(self, document: dict[str, Any]) -> str | None:
        """Indexe un document unique.

        Args:
            document: Document à indexer

        Returns:
            ID du document indexé ou None en cas d'erreur
        """
        if not self.es:
            logger.error("Elasticsearch non disponible")
            return None

        try:
            # Ajouter la date d'indexation
            document["indexed_at"] = datetime.utcnow().isoformat()

            # Indexer le document
            response = self.es.index(index=self.index_name, document=document)

            doc_id = response["_id"]
            logger.info(f"✅ Document indexé: {doc_id}")
            return doc_id

        except Exception as e:
            logger.error(f"❌ Erreur lors de l'indexation: {e}")
            return None

    def bulk_index_documents(self, documents: list[dict[str, Any]]) -> bool:
        """Indexe plusieurs documents en une seule opération.

        Args:
            documents: Liste des documents à indexer

        Returns:
            True si succès, False sinon
        """
        if not self.es:
            logger.error("Elasticsearch non disponible")
            return False

        if not documents:
            return True

        try:
            # Préparer les actions pour bulk
            actions = []
            for doc in documents:
                # Ajouter la date d'indexation
                doc["indexed_at"] = datetime.utcnow().isoformat()

                action = {
                    "_index": self.index_name,
                    "_id": str(uuid.uuid4()),
                    "_source": doc,
                }
                actions.append(action)

            # Exécuter l'opération bulk
            success_count, failed_items = bulk(self.es, actions, raise_on_error=False)

            # Traiter la réponse
            if isinstance(failed_items, list) and failed_items:
                logger.error(f"❌ {len(failed_items)} documents ont échoué lors de l'indexation bulk")
                for item in failed_items:
                    logger.error(f"  - {item}")
                return False
            elif isinstance(failed_items, int):
                # Si failed_items est un int, c'est le nombre d'échecs
                failed_count = failed_items
                if failed_count > 0:
                    logger.error(f"❌ {failed_count} documents ont échoué lors de l'indexation bulk")
                    return False

            logger.info(f"✅ {success_count} documents indexés avec succès")
            return True

        except Exception as e:
            logger.error(f"❌ Erreur lors de l'indexation bulk: {e}")
            return False

    def index_image(
        self,
        image_path: str,
        embedding: list[float],
        source_file: str,
        page_number: int = 1,
        metadata: Collection[str] | dict[str, Any] | None = None,
    ) -> str | None:
        """Indexe une image (méthode de compatibilité).

        Args:
            image_path: Chemin vers l'image
            embedding: Embedding de l'image
            source_file: Fichier source
            page_number: Numéro de page
            metadata: Métadonnées additionnelles

        Returns:
            ID du document ou None
        """
        document = {
            "col_pali_vectors": embedding,
            "image_path": image_path,
            "source_file": source_file,
            "page_number": page_number,
            "metadata": metadata or {},
        }

        return self.index_document(document)

    def bulk_index_images(self, images_data: list[dict[str, Any]]) -> int:
        """Indexe plusieurs images en batch (méthode de compatibilité).

        Args:
            images_data: Liste de dictionnaires avec les données des images

        Returns:
            Nombre d'images indexées avec succès
        """
        documents = []
        for data in images_data:
            doc = {
                "col_pali_vectors": data["embedding"],
                "image_path": data.get("image_path", ""),
                "source_file": data.get("source_file", ""),
                "page_number": data.get("page_number", 1),
                "metadata": data.get("metadata", {}),
            }
            documents.append(doc)

        success = self.bulk_index_documents(documents)
        return len(documents) if success else 0

    def get_stats(self) -> dict[str, Any]:
        """Récupère les statistiques de l'index avec gestion d'erreur robuste.

        Returns:
            Dictionnaire avec les statistiques
        """
        if not self.es:
            return {"status": "disconnected", "index_name": self.index_name}

        try:
            # D'abord vérifier la santé du cluster
            cluster_health = self.es.cluster.health()
            if cluster_health["status"] in ["red", "yellow"]:
                logger.warning(f"⚠️ Cluster Elasticsearch en état {cluster_health['status']}")

            # Vérifier si l'index existe
            if self.es.indices.exists(index=self.index_name):
                try:
                    # Nombre de documents sans timeout deprecated
                    count_response = self.es.count(index=self.index_name)
                    doc_count = count_response["count"]

                    # Statistiques de l'index sans timeout deprecated
                    stats_response = self.es.indices.stats(index=self.index_name)
                    index_stats = stats_response["indices"][self.index_name]

                    # Version Elasticsearch
                    es_info = self.es.info()

                    return {
                        "status": "connected",
                        "index_name": self.index_name,
                        "document_count": doc_count,
                        "index_size_bytes": index_stats["total"]["store"]["size_in_bytes"],
                        "elasticsearch_version": es_info["version"]["number"],
                        "cluster_status": cluster_health["status"],
                    }

                except Exception as index_error:
                    logger.warning(f"⚠️ Erreur lors de l'accès à l'index {self.index_name}: {index_error}")

                    # Retourner des infos minimales si l'index pose problème
                    es_info = self.es.info()
                    return {
                        "status": "connected_with_index_issues",
                        "index_name": self.index_name,
                        "document_count": 0,
                        "index_exists": True,
                        "elasticsearch_version": es_info["version"]["number"],
                        "cluster_status": cluster_health["status"],
                        "error": str(index_error),
                    }
            else:
                # Index n'existe pas - c'est normal
                es_info = self.es.info()
                return {
                    "status": "connected",
                    "index_name": self.index_name,
                    "document_count": 0,
                    "index_exists": False,
                    "elasticsearch_version": es_info["version"]["number"],
                    "cluster_status": cluster_health["status"],
                }

        except Exception as e:
            logger.error(f"❌ Erreur lors de la récupération des stats: {e}")
            return {
                "status": "error",
                "error": str(e),
                "index_name": self.index_name,
            }

    def clear_index(self) -> bool:
        """Vide complètement l'index.

        Returns:
            True si succès, False sinon
        """
        if not self.es:
            logger.error("Elasticsearch non disponible")
            return False

        try:
            # Supprimer tous les documents
            response = self.es.delete_by_query(index=self.index_name, body={"query": {"match_all": {}}})

            deleted = response.get("deleted", 0)
            logger.info(f"✅ {deleted} documents supprimés de l'index '{self.index_name}'")
            return True

        except Exception as e:
            logger.error(f"❌ Erreur lors du vidage de l'index: {e}")
            return False

    def delete_index(self) -> bool:
        """Supprime complètement l'index.

        Returns:
            True si succès, False sinon
        """
        if not self.es:
            logger.error("Elasticsearch non disponible")
            return False

        try:
            if self.es.indices.exists(index=self.index_name):
                self.es.indices.delete(index=self.index_name)
                logger.info(f"✅ Index '{self.index_name}' supprimé")
                return True
            else:
                logger.warning(f"⚠️ L'index '{self.index_name}' n'existe pas")
                return True

        except Exception as e:
            logger.error(f"❌ Erreur lors de la suppression de l'index: {e}")
            return False

    def close(self) -> None:
        """Ferme la connexion Elasticsearch."""
        if self.es:
            self.es.close()
            logger.info("🔌 Connexion Elasticsearch fermée")
