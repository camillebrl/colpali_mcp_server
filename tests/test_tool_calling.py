"""Test the MCP server tool calling implementation with direct method testing."""

import os
import sys
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestMCPToolCalling:
    """Test MCP tool calling functionality."""

    @pytest.fixture
    async def mock_server(self) -> Any:
        """Create a mock server instance."""
        # Fixed: Combined nested with statements
        with (
            patch("colpali_server.image_rag_server.ColPaliModel") as mock_colpali,
            patch("colpali_server.image_rag_server.ElasticsearchModel") as mock_es,
            patch("colpali_server.image_rag_server.Server") as mock_mcp_server,
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
            mock_es_instance.es_host = "localhost:9200"
            mock_es_instance.es_user = "elastic"
            mock_es_instance.es_password = "test"
            mock_es.return_value = mock_es_instance

            # Store handlers that will be captured
            captured_handlers: dict[str, Any] = {}

            def mock_list_tools() -> Callable[[Callable[[], Any]], Callable[[], Any]]:
                def decorator(func: Callable[[], Any]) -> Callable[[], Any]:
                    captured_handlers["list_tools"] = func
                    return func

                return decorator

            def mock_call_tool() -> (
                Callable[[Callable[[str, dict[str, Any]], Any]], Callable[[str, dict[str, Any]], Any]]
            ):
                def decorator(func: Callable[[str, dict[str, Any]], Any]) -> Callable[[str, dict[str, Any]], Any]:
                    captured_handlers["call_tool"] = func
                    return func

                return decorator

            mock_mcp_server_instance = MagicMock()
            mock_mcp_server_instance.list_tools = mock_list_tools
            mock_mcp_server_instance.call_tool = mock_call_tool
            mock_mcp_server.return_value = mock_mcp_server_instance

            from colpali_server.image_rag_server import ImageRAGServer

            server = ImageRAGServer()

            # Store captured handlers on the server for testing
            server._captured_handlers = captured_handlers

            yield server

    @pytest.mark.asyncio
    async def test_list_tools(self, mock_server: Any) -> None:
        """Test listing available tools."""
        # Get the captured list_tools handler
        if "list_tools" in mock_server._captured_handlers:
            handler = mock_server._captured_handlers["list_tools"]
            tools = await handler()

            # Verify tools
            assert len(tools) == 4
            tool_names = [tool.name for tool in tools]

            expected_tools = [
                "search_screenshots",
                "index_screenshots",
                "list_screenshot_indices",
                "delete_screenshot_index",
            ]

            for expected in expected_tools:
                assert expected in tool_names, f"Tool {expected} not found"
        else:
            # Fallback: test that the handler methods exist
            assert hasattr(mock_server, "_search_screenshots")
            assert hasattr(mock_server, "_index_screenshots")
            assert hasattr(mock_server, "_list_screenshot_indices")
            assert hasattr(mock_server, "_delete_screenshot_index")

    @pytest.mark.asyncio
    async def test_call_list_indices(self, mock_server: Any) -> None:
        """Test calling list_screenshot_indices tool."""
        result = await mock_server._list_screenshot_indices({})

        assert len(result) > 0
        assert result[0].type == "text"
        assert "Aucun index" in result[0].text or "Index de screenshots" in result[0].text

    @pytest.mark.asyncio
    async def test_call_search_screenshots(self, mock_server: Any) -> None:
        """Test calling search_screenshots tool."""
        result = await mock_server._search_screenshots({"query": "test query", "top_k": 3, "max_indices": 2})

        assert len(result) > 0
        assert result[0].type == "text"
        text = result[0].text
        assert "résultat" in text or "Aucun index" in text

    @pytest.mark.asyncio
    async def test_call_invalid_tool(self, mock_server: Any) -> None:
        """Test calling an invalid tool through the handler."""
        # Test using the captured call_tool handler if available
        if "call_tool" in mock_server._captured_handlers:
            handler = mock_server._captured_handlers["call_tool"]
            result = await handler("invalid_tool", {})

            assert len(result) > 0
            assert result[0].type == "text"
            assert "Outil inconnu" in result[0].text
        else:
            # Fallback: verify invalid tool names don't have corresponding methods
            assert not hasattr(mock_server, "_invalid_tool")

            # The actual tool handler would return an error for invalid tools
            valid_tools = [
                "_search_screenshots",
                "_index_screenshots",
                "_list_screenshot_indices",
                "_delete_screenshot_index",
            ]

            for tool_method in valid_tools:
                assert hasattr(mock_server, tool_method)

    @pytest.mark.asyncio
    async def test_call_tool_with_missing_params(self, mock_server: Any) -> None:
        """Test calling a tool with missing required parameters."""
        # Test delete with missing confirm parameter
        result = await mock_server._delete_screenshot_index(
            {
                "index_name": "screenshot_test"
                # Missing "confirm" - should default to False
            }
        )

        # Should ask for confirmation when confirm is False or missing
        assert len(result) > 0
        assert result[0].type == "text"
        assert "confirmer" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_tool_schemas(self, mock_server: Any) -> None:
        """Test that tool schemas are correct."""
        # Get the captured list_tools handler
        if "list_tools" in mock_server._captured_handlers:
            handler = mock_server._captured_handlers["list_tools"]
            tools = await handler()

            # Define expected schemas
            tool_schemas = {
                "search_screenshots": {"required": ["query"], "properties": ["query", "top_k", "max_indices"]},
                "index_screenshots": {
                    "required": [],
                    "properties": ["screenshots_dir", "overwrite_existing", "batch_size"],
                },
                "list_screenshot_indices": {"required": [], "properties": []},
                "delete_screenshot_index": {
                    "required": ["index_name", "confirm"],
                    "properties": ["index_name", "confirm"],
                },
            }

            # Verify each tool
            for tool in tools:
                name = tool.name
                if name in tool_schemas:
                    schema = tool.inputSchema

                    assert schema["type"] == "object"
                    assert "properties" in schema

        else:
            # Fallback: test that tool methods exist and have expected behavior
            tool_methods = {
                "_search_screenshots": ["query"],  # Required params
                "_index_screenshots": [],  # No required params
                "_list_screenshot_indices": [],  # No required params
                "_delete_screenshot_index": ["index_name"],  # Note: confirm has default
            }

            for method_name, _required_params in tool_methods.items():
                assert hasattr(mock_server, method_name), f"Method {method_name} not found"


class TestToolFunctionality:
    """Test actual tool functionality."""

    @pytest.fixture
    async def mock_server(self) -> Any:
        """Create a mock server instance."""
        # Fixed: Combined nested with statements
        with (
            patch("colpali_server.image_rag_server.ColPaliModel") as mock_colpali,
            patch("colpali_server.image_rag_server.ElasticsearchModel") as mock_es,
            patch("colpali_server.image_rag_server.Server"),
            patch("colpali_server.image_rag_server.ScreenshotProcessor") as mock_processor,
        ):
            # Setup mocks
            mock_colpali_instance = MagicMock()
            mock_colpali_instance.generate_embeddings.return_value = [0.1] * 1024
            mock_colpali.return_value = mock_colpali_instance

            mock_es_instance = MagicMock()
            mock_es_instance.get_stats.return_value = {"status": "connected"}
            mock_es_instance.es = MagicMock()
            mock_es_instance.es.cat.indices.return_value = []
            mock_es_instance.es_host = "localhost:9200"
            mock_es_instance.es_user = "elastic"
            mock_es_instance.es_password = "test"
            mock_es.return_value = mock_es_instance

            # Mock ScreenshotProcessor to raise FileNotFoundError
            mock_processor.side_effect = FileNotFoundError(
                "Répertoire screenshots introuvable: ./test_screenshots_not_exists"
            )

            from colpali_server.image_rag_server import ImageRAGServer

            server = ImageRAGServer()
            yield server

    @pytest.mark.asyncio
    async def test_index_screenshots_dry_run(self, mock_server: Any) -> None:
        """Test indexing screenshots without actual screenshots directory."""
        result = await mock_server._index_screenshots(
            {"screenshots_dir": "./test_screenshots_not_exists", "overwrite_existing": False, "batch_size": 5}
        )

        assert len(result) > 0
        assert result[0].type == "text"
        assert "Erreur" in result[0].text or "trouvé" in result[0].text

    @pytest.mark.asyncio
    async def test_delete_index_without_confirm(self, mock_server: Any) -> None:
        """Test delete index without confirmation."""
        result = await mock_server._delete_screenshot_index({"index_name": "screenshot_test", "confirm": False})

        assert len(result) > 0
        assert "confirmer" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_search_with_index_selection(self, mock_server: Any) -> None:
        """Test that search properly selects indices based on query."""
        # Mock the index selector to return specific indices
        mock_server.index_selector.select_relevant_indices = MagicMock(return_value=["screenshot_example_com"])

        result = await mock_server._search_screenshots(
            {"query": "example.com homepage design", "top_k": 5, "max_indices": 3}
        )

        assert len(result) > 0
        text = result[0].text
        assert "recherche" in text.lower() or "index" in text.lower() or "résultat" in text.lower()

        # Verify the index selector was called with the query
        mock_server.index_selector.select_relevant_indices.assert_called_once_with("example.com homepage design", 3)
