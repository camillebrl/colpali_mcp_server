"""Integration tests with mocked MCP server."""

from typing import Any

import pytest
from mcp.types import TextContent, Tool


class MockMCPServer:
    """Mock MCP server that simulates the real server behavior."""

    def __init__(self) -> None:
        self.tools = [
            Tool(
                name="search_screenshots",
                description="Rechercher des screenshots pertinents dans tous les index en utilisant une requête textuelle. L'outil sélectionne automatiquement les index les plus pertinents.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Requête de recherche en langage naturel"},
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
                        "confirm": {"type": "boolean", "description": "Confirmer la suppression", "default": False},
                    },
                    "required": ["index_name", "confirm"],
                },
            ),
        ]

        self.indices: list[str] = []  # Track created indices

    async def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle JSON-RPC requests."""
        method = request.get("method")
        request_id = request.get("id", 1)

        if method == "initialize":
            return {"jsonrpc": "2.0", "id": request_id, "result": {"capabilities": {"tools": True}}}

        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": [
                        {"name": tool.name, "description": tool.description, "inputSchema": tool.inputSchema}
                        for tool in self.tools
                    ]
                },
            }

        elif method == "tools/call":
            params = request.get("params", {})
            tool_name = params.get("name")
            arguments = params.get("arguments", {})

            if tool_name == "list_screenshot_indices":
                if not self.indices:
                    content = [
                        TextContent(
                            type="text",
                            text="📂 Aucun index de screenshots trouvé.\nUtilisez 'index_screenshots' pour créer des index.",
                        )
                    ]
                else:
                    text = f"📂 Index de screenshots disponibles ({len(self.indices)})\n\n"
                    for idx in self.indices:
                        text += f"🗂️ {idx}\n"
                    content = [TextContent(type="text", text=text)]

                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"content": [{"type": c.type, "text": c.text} for c in content]},
                }

            elif tool_name == "search_screenshots":
                query = arguments.get("query", "")
                if not self.indices:
                    content = [
                        TextContent(
                            type="text",
                            text="❌ Aucun index de screenshots trouvé. Utilisez 'index_screenshots' pour créer des index.",
                        )
                    ]
                else:
                    content = [
                        TextContent(
                            type="text",
                            text=f"📚 0 résultat(s) trouvé(s) pour: '{query}'\n🎯 Index recherchés: {', '.join(self.indices[:3])}\n",
                        )
                    ]

                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"content": [{"type": c.type, "text": c.text} for c in content]},
                }

            elif tool_name == "index_screenshots":
                content = [
                    TextContent(
                        type="text",
                        text="❌ Erreur lors de l'indexation: Répertoire screenshots introuvable: ./test_screenshots_not_exists",
                    )
                ]

                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"content": [{"type": c.type, "text": c.text} for c in content]},
                }

            elif tool_name == "delete_screenshot_index":
                if not arguments.get("confirm", False):
                    content = [TextContent(type="text", text="⚠️ Veuillez confirmer la suppression avec confirm=true")]
                else:
                    content = [TextContent(type="text", text="✅ Index 'screenshot_test' supprimé avec succès")]

                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"content": [{"type": c.type, "text": c.text} for c in content]},
                }

            else:
                content = [TextContent(type="text", text=f"❌ Outil inconnu: {tool_name}")]

                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"content": [{"type": c.type, "text": c.text} for c in content]},
                }

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }


class TestMCPIntegration:
    """Test MCP integration with mocked server."""

    @pytest.fixture
    def mock_server(self) -> MockMCPServer:
        """Create a mock MCP server."""
        return MockMCPServer()

    @pytest.mark.asyncio
    async def test_initialize(self, mock_server: MockMCPServer) -> None:
        """Test server initialization."""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"},
            },
        }

        response = await mock_server.handle_request(request)

        assert response["id"] == 1
        assert "result" in response
        assert "capabilities" in response["result"]

    @pytest.mark.asyncio
    async def test_list_tools(self, mock_server: MockMCPServer) -> None:
        """Test listing available tools."""
        request = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}

        response = await mock_server.handle_request(request)

        assert "result" in response
        assert "tools" in response["result"]
        tools = response["result"]["tools"]

        assert len(tools) == 4
        tool_names = [tool["name"] for tool in tools]
        assert "search_screenshots" in tool_names
        assert "index_screenshots" in tool_names
        assert "list_screenshot_indices" in tool_names
        assert "delete_screenshot_index" in tool_names

    @pytest.mark.asyncio
    async def test_call_list_indices_empty(self, mock_server: MockMCPServer) -> None:
        """Test calling list_screenshot_indices with no indices."""
        request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "list_screenshot_indices", "arguments": {}},
        }

        response = await mock_server.handle_request(request)

        assert "result" in response
        assert "content" in response["result"]
        content = response["result"]["content"]

        assert len(content) > 0
        assert content[0]["type"] == "text"
        assert "Aucun index" in content[0]["text"]

    @pytest.mark.asyncio
    async def test_call_search_no_indices(self, mock_server: MockMCPServer) -> None:
        """Test search with no indices."""
        request = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "search_screenshots", "arguments": {"query": "test query", "top_k": 5}},
        }

        response = await mock_server.handle_request(request)

        assert "result" in response
        content = response["result"]["content"]
        assert "Aucun index" in content[0]["text"]

    @pytest.mark.asyncio
    async def test_call_invalid_tool(self, mock_server: MockMCPServer) -> None:
        """Test calling invalid tool."""
        request = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": "invalid_tool", "arguments": {}},
        }

        response = await mock_server.handle_request(request)

        assert "result" in response
        content = response["result"]["content"]
        assert "Outil inconnu" in content[0]["text"]

    @pytest.mark.asyncio
    async def test_tool_schemas(self, mock_server: MockMCPServer) -> None:
        """Test tool schemas are correct."""
        request = {"jsonrpc": "2.0", "id": 6, "method": "tools/list"}

        response = await mock_server.handle_request(request)
        tools = response["result"]["tools"]

        # Verify search_screenshots schema
        search_tool = next(t for t in tools if t["name"] == "search_screenshots")
        schema = search_tool["inputSchema"]

        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert schema["properties"]["query"]["type"] == "string"
        assert "query" in schema["required"]

        # Verify delete_screenshot_index schema
        delete_tool = next(t for t in tools if t["name"] == "delete_screenshot_index")
        schema = delete_tool["inputSchema"]

        assert "index_name" in schema["required"]
        assert "confirm" in schema["required"]

    @pytest.mark.asyncio
    async def test_delete_without_confirm(self, mock_server: MockMCPServer) -> None:
        """Test delete without confirmation."""
        request = {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {
                "name": "delete_screenshot_index",
                "arguments": {"index_name": "screenshot_test", "confirm": False},
            },
        }

        response = await mock_server.handle_request(request)

        assert "result" in response
        content = response["result"]["content"]
        assert "confirmer" in content[0]["text"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
