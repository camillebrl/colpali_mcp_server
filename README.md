# Colpali MCP Server - Python

Un serveur MCP (Model Context Protocol) pour effectuer de la recherche (retrieval) sur une base de données d'images en utilisant ColPali et Elasticsearch.

## 🚀 Fonctionnalités

- **Recherche sémantique d'images** : Trouvez des images pertinentes en utilisant des requêtes en langage naturel
- **Indexation d'images et de PDFs** : Indexez automatiquement des images individuelles ou des PDFs complets
- **Embeddings multimodaux** : Utilise [ColPali](https://huggingface.co/vidore/colqwen2-v1.0) pour générer des embeddings riches combinant vision et langage
- **Stockage scalable** : Elasticsearch pour un stockage et une recherche efficaces
- **API MCP standard** : Compatible avec tout client MCP

## 📋 Prérequis

- Python 3.10+
- Elasticsearch 8.0+
- GPU recommandé pour ColPali (fonctionne aussi sur CPU)

## 🔧 Installation

```bash
git clone https://github.com/camillebrl/colpali_server
cd colpali_server
make install
```

## 🎯 Utilisation

Ajoute à ton mcp.json de ton client MCP:
```json
{
  "mcpServers": {
    "colpali-server": {
          "command": "path to colpali_server/start_server.sh"
          "env": {
            "ES_HOST": "localhost:9200", # adapt to your elasticsearch hostname
            "ES_USER": "elastic", # adapt to your elasticsearch username
            "ES_PASSWORD": "mot_de_passe_elasticsearch" # adapt to your elasticsearch password
          }
    }
  }
}
```

### Utiliser avec un client MCP

Le serveur expose les outils suivants :

#### 🔍 `search_images`
Recherche des images pertinentes dans la base de données.

```json
{
  "tool": "search_images",
  "arguments": {
    "query": "schéma d'architecture réseau",
    "top_k": 5
  }
}
```

#### 📥 `index_image`
Indexe une nouvelle image dans la base de données.
Note que les images d'un folder sont indexées 1 par 1 (batch size de 1). Si vous avez accès à une meilleure GPU, tester un batch_size plus grand.
```json
{
  "tool": "index_image",
  "arguments": {
    "image_path": "/path/to/image.png",
    "source_file": "architecture.pdf",
    "page_number": 3,
    "metadata": {
      "author": "John Doe",
      "category": "network"
    }
  }
}
```

#### 📊 `get_index_stats`
Obtient les statistiques de l'index.

```json
{
  "tool": "get_index_stats",
  "arguments": {}
}
```

#### 🗑️ `clear_index`
Vide complètement l'index (attention : irréversible).

```json
{
  "tool": "clear_index",
  "arguments": {
    "confirm": true
  }
}
```

#### Debugging
Pour afficher les logs du server:
```bash
tail -n 1000 /tmp/colpali_mcp_startup.log
```

## 📚 Ressources

- [Documentation MCP](https://modelcontextprotocol.io)
- [ColPali Paper](https://arxiv.org/abs/2407.01449)
- [Elasticsearch Guide](https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html)

## NOTE
Si vous avez plus de capacité GPU, modifiez le maximul top_k dans le image_rag_server.py pour le tool search_screenshots à + (en fonction de vos ressources). Car en effet, c'est le nombre max d'images que prendra le vlm mcp server tool search en entrée.