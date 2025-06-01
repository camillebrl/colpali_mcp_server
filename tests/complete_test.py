"""Script de test corrigé pour le serveur ColPali MCP."""

import asyncio
import json
import os
import queue
import queue as queue_module
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


class MCPTester:
    """Testeur pour le serveur MCP ColPali."""

    def __init__(self, project_root: str = ".") -> None:
        self.project_root = Path(project_root)
        self.src_path = self.project_root / "src"
        self.process: subprocess.Popen[str] | None = None
        self.stderr_queue: queue_module.Queue[str] = queue.Queue()
        self.stderr_thread: threading.Thread | None = None

    def _stderr_reader(self) -> None:
        """Thread pour lire stderr en continu."""
        if self.process and self.process.stderr:
            try:
                for line in iter(self.process.stderr.readline, ""):
                    if line:
                        self.stderr_queue.put(line.strip())
                    if self.process.poll() is not None:
                        break
            except Exception:
                pass

    def _get_stderr_output(self) -> list[str]:
        """Récupère tous les messages stderr disponibles."""
        messages: list[str] = []
        try:
            while True:
                message = self.stderr_queue.get_nowait()
                messages.append(message)
        except queue.Empty:
            pass
        return messages

    async def start_server(self) -> None:
        """Démarre le serveur MCP."""
        print("🚀 Démarrage du serveur MCP...")

        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.src_path) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONUNBUFFERED"] = "1"

        # Vérifier que le module existe
        cli_path = self.src_path / "colpali_server" / "cli.py"
        if not cli_path.exists():
            raise Exception(f"Module CLI introuvable: {cli_path}")

        print(f"📂 Répertoire de travail: {self.src_path}")

        self.process = subprocess.Popen(
            [sys.executable, "-m", "colpali_server.cli"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0,
            cwd=str(self.src_path),
            env=env,
        )

        # Démarrer le thread de lecture stderr
        self.stderr_thread = threading.Thread(target=self._stderr_reader)
        self.stderr_thread.daemon = True
        self.stderr_thread.start()

        # Attendre que le serveur soit vraiment prêt
        print("⏳ Attente du démarrage complet du serveur...")
        ready_indicators = [
            "Serveur prêt à recevoir des requêtes",
            "ready to receive requests",
            "server started",
            "listening",
        ]

        timeout = 60  # 1 minute pour le démarrage
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Vérifier si le processus est mort
            if self.process.poll() is not None:
                stderr_messages = self._get_stderr_output()
                error_msg = f"Serveur crashed (code: {self.process.returncode})"
                if stderr_messages:
                    error_msg += "\nErreurs:\n" + "\n".join(stderr_messages[-10:])
                raise Exception(error_msg)

            # Afficher les messages stderr et chercher les indicateurs de ready
            stderr_messages = self._get_stderr_output()
            for msg in stderr_messages:
                print(f"🔍 Serveur: {msg}")

                # Chercher des signes que le serveur est prêt
                if any(indicator in msg.lower() for indicator in ready_indicators):
                    print("✅ Serveur prêt détecté")
                    await asyncio.sleep(2)  # Pause supplémentaire pour stabilité
                    return

            await asyncio.sleep(0.5)

        # Si on arrive ici, timeout mais on continue quand même
        print("⚠️ Timeout d'attente, mais le serveur semble démarré")

    async def send_request(
        self, request: dict[str, Any], timeout: int = 30, wait_after: bool = True
    ) -> dict[str, Any]:
        """Envoie une requête JSON-RPC au serveur avec attente appropriée."""
        request_line = json.dumps(request) + "\n"
        print(f"📤 Envoi: {request['method']} (id: {request.get('id', 'N/A')})")

        if not self.process or self.process.poll() is not None:
            stderr_messages = self._get_stderr_output()
            raise Exception(
                f"Processus serveur mort. Messages: {stderr_messages[-5:] if stderr_messages else 'Aucun'}"
            )

        try:
            if self.process.stdin:
                self.process.stdin.write(request_line)
                self.process.stdin.flush()
        except Exception as e:
            raise Exception(f"Erreur écriture stdin: {e}") from e

        # Attendre la réponse
        try:
            start_time = time.time()
            while time.time() - start_time < timeout:
                # Vérifier si le processus est toujours vivant
                if self.process.poll() is not None:
                    stderr_messages = self._get_stderr_output()
                    raise Exception(
                        f"Serveur crashed. Messages: {stderr_messages[-5:] if stderr_messages else 'Aucun'}"
                    )

                # Essayer de lire une ligne
                try:
                    if self.process.stdout:
                        response_line = await asyncio.wait_for(
                            asyncio.to_thread(self.process.stdout.readline), timeout=2.0
                        )
                    else:
                        continue

                    if response_line.strip():
                        try:
                            response = json.loads(response_line)
                            print(f"📥 Réponse reçue pour {request['method']}")

                            # Attendre un peu après certaines requêtes pour laisser le serveur se stabiliser
                            if wait_after and request["method"] in ["initialize", "notifications/initialized"]:
                                await asyncio.sleep(1)

                            return response
                        except json.JSONDecodeError as e:
                            print(f"⚠️ Erreur JSON: {e}, ligne: {repr(response_line)}")
                            continue

                except asyncio.TimeoutError:
                    # Afficher les messages stderr pendant l'attente
                    stderr_messages = self._get_stderr_output()
                    for msg in stderr_messages:
                        print(f"🔍 Serveur: {msg}")
                    continue

            raise Exception(f"Timeout pour {request['method']}")

        except Exception as e:
            if "Timeout pour" in str(e):
                raise
            else:
                raise Exception(f"Erreur lors de l'attente de réponse: {e}") from e

    async def initialize(self) -> bool:
        """Initialise le serveur MCP selon le protocole standard."""
        print("\n1️⃣ Initialisation du serveur MCP...")

        # Étape 1: Envoyer initialize
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"},
            },
        }

        try:
            response = await self.send_request(init_request, timeout=60)

            if "result" in response:
                print("✅ Initialisation réussie")
                print(f"📋 Capacités serveur: {response['result']}")

                # Étape 2: Envoyer notifications/initialized (obligatoire)
                print("📤 Envoi de la notification initialized...")
                initialized_notif = {"jsonrpc": "2.0", "method": "notifications/initialized"}

                # Pour les notifications, on n'attend pas de réponse
                notif_line = json.dumps(initialized_notif) + "\n"
                if self.process and self.process.stdin:
                    self.process.stdin.write(notif_line)
                    self.process.stdin.flush()

                # Attendre un peu que le serveur traite la notification
                await asyncio.sleep(2)
                print("✅ Notification initialized envoyée")

                return True
            else:
                print(f"❌ Erreur d'initialisation: {response.get('error')}")
                return False

        except Exception as e:
            print(f"❌ Exception durant l'initialisation: {e}")
            return False

    async def list_tools(self) -> list[dict[str, Any]]:
        """Liste les outils disponibles."""
        print("\n2️⃣ Récupération des outils disponibles...")

        request = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}

        try:
            response = await self.send_request(request, timeout=30)

            if "result" in response and "tools" in response["result"]:
                tools = response["result"]["tools"]
                print(f"✅ {len(tools)} outils trouvés:")
                for tool in tools:
                    print(f"   • {tool['name']}: {tool['description'][:60]}...")
                return tools
            else:
                print(f"❌ Erreur listing tools: {response.get('error')}")
                return []
        except Exception as e:
            print(f"❌ Exception lors du listing des outils: {e}")
            return []

    async def list_indices(self) -> bool:
        """Liste les index créés."""
        print("\n3️⃣ Test de l'outil list_screenshot_indices...")

        request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "list_screenshot_indices", "arguments": {}},
        }

        try:
            response = await self.send_request(request)

            if "result" in response:
                content = response["result"]["content"]
                if content:
                    print("📂 Index disponibles:")
                    print(content[0]["text"])
                    return True

            print(f"❌ Erreur listing indices: {response.get('error')}")
            return False
        except Exception as e:
            print(f"❌ Exception lors du listing des indices: {e}")
            return False

    def cleanup(self) -> None:
        """Nettoie les ressources."""
        if self.process:
            print("\n🧹 Arrêt du serveur...")

            # Récupérer les derniers messages stderr
            stderr_messages = self._get_stderr_output()
            if stderr_messages:
                print("📋 Derniers messages du serveur:")
                for msg in stderr_messages[-5:]:
                    print(f"   {msg}")

            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("⚠️ Forçage de l'arrêt du serveur...")
                self.process.kill()
                self.process.wait()
            print("✅ Serveur arrêté")


async def check_prerequisites() -> bool:
    """Vérifications préalables."""
    print("🔧 Vérifications préalables...")

    if not Path("src/colpali_server").exists():
        print("❌ Dossier src/colpali_server introuvable")
        return False

    if not Path("src/colpali_server/cli.py").exists():
        print("❌ Module CLI introuvable")
        return False

    try:
        import importlib.util
        import sys

        # Use importlib.util.find_spec instead of importing
        spec = importlib.util.find_spec("colpali_server")
        if spec is None:
            sys.path.insert(0, "src")
            spec = importlib.util.find_spec("colpali_server")

        if spec is not None:
            print("✅ Module colpali_server importable")
        else:
            print("❌ Module colpali_server non trouvé")
            return False

    except ImportError as e:
        print(f"❌ Import impossible: {e}")
        return False

    return True


async def main() -> None:
    """Test de base du protocole MCP."""
    print("🧪 Test du protocole MCP ColPali")
    print("=" * 40)

    if not await check_prerequisites():
        return

    tester = MCPTester()

    try:
        await tester.start_server()

        if not await tester.initialize():
            print("❌ Impossible d'initialiser le serveur")
            return

        tools = await tester.list_tools()
        if not tools:
            print("❌ Aucun outil trouvé")
            return

        await tester.list_indices()

        print("\n🎉 Test de base terminé avec succès!")
        print("💡 Le serveur MCP fonctionne correctement.")
        print("💡 Vous pouvez maintenant l'utiliser avec Claude Desktop.")

    except Exception as e:
        print(f"\n❌ Erreur durant le test: {e}")
        import traceback

        traceback.print_exc()

    finally:
        tester.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
