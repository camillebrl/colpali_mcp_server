"""Simple unit tests for Image RAG Server without starting full server."""

import os
import sys
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestImageRAGServerUnit:
    """Unit tests for ImageRAGServer class."""

    @pytest.mark.asyncio
    async def test_server_initialization(self) -> None:
        """Test that the server initializes correctly with mocks."""
        # Mock all dependencies - Fixed: Combined nested with statements
        with (
            patch("colpali_server.image_rag_server.ColPaliModel") as mock_colpali,
            patch("colpali_server.image_rag_server.ElasticsearchModel") as mock_es,
            patch("colpali_server.image_rag_server.Server") as mock_server,
        ):
            # Setup mocks
            mock_colpali_instance = MagicMock()
            mock_colpali.return_value = mock_colpali_instance

            mock_es_instance = MagicMock()
            mock_es_instance.get_stats.return_value = {"status": "connected"}
            mock_es_instance.es = MagicMock()
            mock_es.return_value = mock_es_instance

            mock_server_instance = MagicMock()
            mock_server.return_value = mock_server_instance

            # Import and create server
            from colpali_server.image_rag_server import ImageRAGServer

            server = ImageRAGServer()

            # Verify initialization
            assert server.colpali_model == mock_colpali_instance
            assert server.es_model == mock_es_instance
            assert server.server == mock_server_instance

    @pytest.mark.asyncio
    async def test_list_tools_handler(self) -> None:
        """Test the list_tools handler returns correct tools."""
        # Fixed: Combined nested with statements
        with (
            patch("colpali_server.image_rag_server.ColPaliModel"),
            patch("colpali_server.image_rag_server.ElasticsearchModel") as mock_es,
            patch("colpali_server.image_rag_server.Server") as mock_server,
        ):
            # Setup mocks
            mock_es_instance = MagicMock()
            mock_es_instance.get_stats.return_value = {"status": "connected"}
            mock_es_instance.es = MagicMock()
            mock_es.return_value = mock_es_instance

            # Create a list to capture the decorated function
            list_tools_handler: Callable[[], Any] | None = None

            def capture_list_tools() -> Callable[[Callable[[], Any]], Callable[[], Any]]:
                def decorator(func: Callable[[], Any]) -> Callable[[], Any]:
                    nonlocal list_tools_handler
                    list_tools_handler = func
                    return func

                return decorator

            mock_server_instance = MagicMock()
            mock_server_instance.list_tools = capture_list_tools
            mock_server.return_value = mock_server_instance

            # Import and create server
            from colpali_server.image_rag_server import ImageRAGServer

            ImageRAGServer()

            # Call the captured handler
            if list_tools_handler:
                tools = await list_tools_handler()

                # Verify tools
                assert len(tools) == 4
                tool_names = [tool.name for tool in tools]
                assert "search_screenshots" in tool_names
                assert "index_screenshots" in tool_names
                assert "list_screenshot_indices" in tool_names
                assert "delete_screenshot_index" in tool_names

    def test_screenshot_processor(self) -> None:
        """Test ScreenshotProcessor functionality."""
        from colpali_server.image_rag_server import ScreenshotProcessor

        # Test source extraction from filename
        processor = ScreenshotProcessor.__new__(ScreenshotProcessor)

        test_cases = [
            ("site_example.com_page1.png", "example.com"),
            ("document_test_page1.png", "document"),
            ("example.com_homepage.png", "example.com"),
            ("my_file.png", "my_file"),
        ]

        for filename, expected_source in test_cases:
            source = processor.get_source_from_filename(filename)
            assert expected_source in source or source in expected_source

    def test_index_selector(self) -> None:
        """Test IndexSelector functionality."""
        from colpali_server.image_rag_server import IndexSelector

        # Mock ES model
        mock_es = MagicMock()
        mock_es.es = MagicMock()
        mock_es.es.cat.indices.return_value = [
            {"index": "screenshot_example_com"},
            {"index": "screenshot_test_site"},
            {"index": "screenshot_demo_app"},
            {"index": ".kibana"},  # Should be filtered out
        ]

        selector = IndexSelector(mock_es)

        # Test get all indices
        indices = selector.get_all_indices()
        assert len(indices) == 3
        assert "screenshot_example_com" in indices
        assert "screenshot_test_site" in indices
        assert "screenshot_demo_app" in indices
        assert ".kibana" not in indices

        # Test selection based on query
        selected = selector.select_relevant_indices("example.com homepage", max_indices=2)
        assert "screenshot_example_com" in selected
        assert len(selected) <= 2

    @pytest.mark.asyncio
    async def test_search_tool_handler(self) -> None:
        """Test search_screenshots tool functionality."""
        # Fixed: Combined nested with statements
        with (
            patch("colpali_server.image_rag_server.ColPaliModel") as mock_colpali,
            patch("colpali_server.image_rag_server.ElasticsearchModel") as mock_es,
            patch("colpali_server.image_rag_server.Server"),
        ):
            # Setup mocks
            mock_colpali_instance = MagicMock()
            mock_colpali_instance.generate_query_embedding.return_value = [0.1] * 1024
            mock_colpali.return_value = mock_colpali_instance

            mock_es_instance = MagicMock()
            mock_es_instance.get_stats.return_value = {"status": "connected"}
            mock_es_instance.es = MagicMock()
            mock_es_instance.es.cat.indices.return_value = []
            mock_es_instance.search_by_embedding.return_value = []
            mock_es.return_value = mock_es_instance

            from colpali_server.image_rag_server import ImageRAGServer

            server = ImageRAGServer()

            # Test search with no indices
            result = await server._search_screenshots({"query": "test query", "top_k": 5, "max_indices": 3})

            assert len(result) > 0
            assert result[0].type == "text"
            assert "Aucun index" in result[0].text


class TestElasticsearchModel:
    """Unit tests for ElasticsearchModel."""

    def test_initialization(self) -> None:
        """Test ElasticsearchModel initialization."""
        with patch("colpali_server.elasticsearch_model.Elasticsearch") as mock_es_class:
            mock_es = MagicMock()
            mock_es.ping.return_value = True
            mock_es.info.return_value = {"version": {"number": "8.0.0"}}
            mock_es.indices.exists.return_value = False
            mock_es_class.return_value = mock_es

            from colpali_server.elasticsearch_model import ElasticsearchModel

            model = ElasticsearchModel(
                index_name="test_index", es_host="localhost:9200", es_user="elastic", es_password="test"
            )

            # Verify connection was attempted
            mock_es_class.assert_called_once()
            mock_es.ping.assert_called_once()
            assert model.es == mock_es

    def test_search_by_embedding(self) -> None:
        """Test search functionality."""
        with patch("colpali_server.elasticsearch_model.Elasticsearch") as mock_es_class:
            mock_es = MagicMock()
            mock_es.ping.return_value = True
            mock_es.info.return_value = {"version": {"number": "8.0.0"}}
            mock_es.indices.exists.return_value = True
            mock_es.search.return_value = {
                "hits": {"hits": [{"_id": "1", "_score": 0.95, "_source": {"filename": "test.png", "source": "test"}}]}
            }
            mock_es_class.return_value = mock_es

            from colpali_server.elasticsearch_model import ElasticsearchModel

            model = ElasticsearchModel(index_name="test_index")
            model.es = mock_es  # Force set the mock

            # Test search
            results = model.search_by_embedding([0.1] * 1024, k=5)

            assert len(results) == 1
            assert results[0]["score"] == 0.95
            assert results[0]["filename"] == "test.png"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
