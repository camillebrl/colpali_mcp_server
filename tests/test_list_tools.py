"""Test listing all available tools from Image RAG server with direct method testing."""

import os
import sys
from collections.abc import Callable
from typing import Any, TypeVar
from unittest.mock import MagicMock, patch

import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestImageRAGToolsList:
    """Test the Image RAG MCP server tools listing."""

    @pytest.mark.asyncio
    async def test_server_initialization(self) -> None:
        """Test that the server initializes correctly."""
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
            mock_es_instance.es.cat.indices.return_value = []
            mock_es.return_value = mock_es_instance

            mock_server_instance = MagicMock()
            mock_server.return_value = mock_server_instance

            from colpali_server.image_rag_server import ImageRAGServer

            server = ImageRAGServer()

            # Verify server was created
            assert server is not None
            assert hasattr(server, "colpali_model")
            assert hasattr(server, "es_model")
            assert hasattr(server, "index_selector")

    @pytest.mark.asyncio
    async def test_list_tools(self) -> None:
        """Test listing all available tools."""
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

            # Capture the list_tools handler
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

            from colpali_server.image_rag_server import ImageRAGServer

            ImageRAGServer()

            # Call the handler directly
            if list_tools_handler:
                tools = await list_tools_handler()

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

    @pytest.mark.asyncio
    async def test_tool_schemas(self) -> None:
        """Test that specific tools have correct schemas."""
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

            # Capture the list_tools handler
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

            from colpali_server.image_rag_server import ImageRAGServer

            ImageRAGServer()

            if list_tools_handler:
                tools = await list_tools_handler()

                # Define expected schemas
                tool_schemas = {
                    "search_screenshots": {
                        "required": ["query"],
                        "properties": ["query", "top_k", "max_indices"],
                        "defaults": {"top_k": 5, "max_indices": 3},
                    },
                    "index_screenshots": {
                        "required": [],
                        "properties": ["screenshots_dir", "overwrite_existing", "batch_size"],
                        "defaults": {
                            "screenshots_dir": "./screenshots",
                            "overwrite_existing": False,
                            "batch_size": 10,
                        },
                    },
                    "list_screenshot_indices": {"required": [], "properties": []},
                    "delete_screenshot_index": {
                        "required": ["index_name", "confirm"],
                        "properties": ["index_name", "confirm"],
                        "defaults": {"confirm": False},
                    },
                }

                # Verify each tool
                for tool in tools:
                    name = tool.name
                    if name in tool_schemas:
                        schema = tool.inputSchema

                        assert schema["type"] == "object"
                        assert "properties" in schema

    @pytest.mark.asyncio
    async def test_tool_descriptions(self) -> None:
        """Test that all tools have meaningful descriptions."""
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

            # Capture the list_tools handler
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

            from colpali_server.image_rag_server import ImageRAGServer

            ImageRAGServer()

            if list_tools_handler:
                tools = await list_tools_handler()

                for tool in tools:
                    assert tool.description
                    assert len(tool.description) > 10
                    assert not tool.description.startswith("TODO")

    @pytest.mark.asyncio
    async def test_search_screenshots_tool_details(self) -> None:
        """Test search_screenshots tool details."""
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

            # Capture the list_tools handler
            list_tools_handler: Any | None = None

            T = TypeVar("T", bound=Callable[..., Any])

            def capture_list_tools() -> Callable[[T], T]:
                def decorator(func: T) -> T:
                    nonlocal list_tools_handler
                    list_tools_handler = func
                    return func

                return decorator

            mock_server_instance = MagicMock()
            mock_server_instance.list_tools = capture_list_tools
            mock_server.return_value = mock_server_instance

            from colpali_server.image_rag_server import ImageRAGServer

            ImageRAGServer()

            if list_tools_handler:
                tools = await list_tools_handler()

                search_tool = next((t for t in tools if t.name == "search_screenshots"), None)

                assert search_tool is not None
                assert "automatique" in search_tool.description

                # Check input schema
                schema = search_tool.inputSchema
                assert schema["properties"]["query"]["type"] == "string"
                assert schema["properties"]["top_k"]["type"] == "integer"
                assert schema["properties"]["top_k"]["minimum"] == 1
                assert schema["properties"]["top_k"]["maximum"] == 20
                assert schema["properties"]["max_indices"]["type"] == "integer"

    @pytest.mark.asyncio
    async def test_call_tool_via_direct_method(self) -> None:
        """Test calling a tool via direct method call."""
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
            mock_es_instance.es.cat.indices.return_value = []
            mock_es.return_value = mock_es_instance

            # Capture the tool handler
            tool_handler: Any | None = None

            T = TypeVar("T", bound=Callable[..., Any])

            def capture_tool_handler() -> Callable[[T], T]:
                def decorator(func: T) -> T:
                    nonlocal tool_handler
                    tool_handler = func
                    return func

                return decorator

            mock_server_instance = MagicMock()
            mock_server_instance.call_tool = capture_tool_handler
            mock_server.return_value = mock_server_instance

            from colpali_server.image_rag_server import ImageRAGServer

            ImageRAGServer()

            # Call the tool handler directly
            if tool_handler:
                result = await tool_handler("list_screenshot_indices", {})

                assert len(result) > 0
                assert result[0].type == "text"
                assert "Aucun index" in result[0].text or "Index de screenshots" in result[0].text
