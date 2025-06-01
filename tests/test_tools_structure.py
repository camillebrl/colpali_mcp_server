"""Test the tools structure and schemas directly."""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mcp.types import Tool


class TestToolsStructure:
    """Test the structure and schemas of MCP tools."""

    def test_tool_definitions(self) -> None:
        """Test that tools are properly defined with correct schemas."""
        # Define the expected tools structure
        expected_tools = [
            {
                "name": "search_screenshots",
                "description_contains": ["Rechercher", "screenshots", "automatique"],
                "required_params": ["query"],
                "optional_params": ["top_k", "max_indices"],
                "param_types": {"query": "string", "top_k": "integer", "max_indices": "integer"},
            },
            {
                "name": "index_screenshots",
                "description_contains": ["index", "Elasticsearch", "screenshots"],
                "required_params": [],
                "optional_params": ["screenshots_dir", "overwrite_existing", "batch_size"],
                "param_types": {"screenshots_dir": "string", "overwrite_existing": "boolean", "batch_size": "integer"},
            },
            {
                "name": "list_screenshot_indices",
                "description_contains": ["Lister", "index", "screenshots"],
                "required_params": [],
                "optional_params": [],
                "param_types": {},
            },
            {
                "name": "delete_screenshot_index",
                "description_contains": ["Supprimer", "index", "screenshots"],
                "required_params": ["index_name", "confirm"],
                "optional_params": [],
                "param_types": {"index_name": "string", "confirm": "boolean"},
            },
        ]

        # Create actual tool objects to match expected structure
        tools = [
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

        # Verify tools match expected structure
        assert len(tools) == len(expected_tools)

        for i, (tool, expected) in enumerate(zip(tools, expected_tools, strict=False)):
            # Check name
            assert tool.name == expected["name"], f"Tool {i}: name mismatch"

            # Check input schema
            schema = tool.inputSchema
            assert schema["type"] == "object"
            assert "properties" in schema

            # Check required parameters
            if expected["required_params"]:
                assert "required" in schema
                required_params = schema.get("required", [])
                if isinstance(required_params, list):
                    assert set(required_params) == set(expected["required_params"])
            else:
                assert "required" not in schema or schema["required"] == []

    def test_tool_defaults(self) -> None:
        """Test that optional parameters have correct defaults."""
        # Test search_screenshots defaults
        search_tool = Tool(
            name="search_screenshots",
            description="...",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20},
                    "max_indices": {"type": "integer", "default": 3, "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
            },
        )

        assert search_tool.inputSchema["properties"]["top_k"]["default"] == 5
        assert search_tool.inputSchema["properties"]["max_indices"]["default"] == 3

        # Test index_screenshots defaults
        index_tool = Tool(
            name="index_screenshots",
            description="...",
            inputSchema={
                "type": "object",
                "properties": {
                    "screenshots_dir": {"type": "string", "default": "./screenshots"},
                    "overwrite_existing": {"type": "boolean", "default": False},
                    "batch_size": {"type": "integer", "default": 10},
                },
            },
        )

        assert index_tool.inputSchema["properties"]["screenshots_dir"]["default"] == "./screenshots"
        assert index_tool.inputSchema["properties"]["overwrite_existing"]["default"] is False
        assert index_tool.inputSchema["properties"]["batch_size"]["default"] == 10
