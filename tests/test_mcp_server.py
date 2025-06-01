"""Simple test to check MCP server functionality."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path


async def test_mcp_server() -> None:
    """Test basic MCP server functionality."""
    # Setup paths - FIX: Go up one level to project root
    project_root = Path(__file__).parent.parent  # Go up from tests/ to project root
    src_path = project_root / "src"

    # Verify paths exist
    print(f"Project root: {project_root}")
    print(f"Source path: {src_path}")
    print(f"Source exists: {src_path.exists()}")

    if not src_path.exists():
        print(f"ERROR: Source directory does not exist at {src_path}")
        return

    # Check if the CLI module exists
    cli_module = src_path / "colpali_server" / "cli.py"
    print(f"CLI module: {cli_module}")
    print(f"CLI exists: {cli_module.exists()}")

    if not cli_module.exists():
        print(f"ERROR: CLI module does not exist at {cli_module}")
        return

    # Setup environment
    env = os.environ.copy()
    env["PYTHONPATH"] = str(src_path) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"

    # Start server
    print("Starting server...")
    cmd = [sys.executable, "-m", "colpali_server.cli"]
    print(f"Command: {' '.join(cmd)}")
    print(f"Working directory: {src_path}")

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=0,
        cwd=str(src_path),
        env=env,
    )

    # Wait for startup with better diagnostics
    await asyncio.sleep(3)

    # Check if process is running
    if process.poll() is not None:
        print("ERROR: Server died!")
        stdout_stream = process.stdout
        stderr_stream = process.stderr
        if stdout_stream is not None:
            stdout_data = stdout_stream.read()
            print(f"STDOUT: {stdout_data}")
        if stderr_stream is not None:
            stderr_data = stderr_stream.read()
            print(f"STDERR: {stderr_data}")
        print(f"Return code: {process.returncode}")
        return

    print("Server is running, sending initialize request...")

    # Send initialize request
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

    request_line = json.dumps(request) + "\n"
    print(f"Sending: {request_line.strip()}")

    stdin_stream = process.stdin
    stdout_stream = process.stdout
    if stdin_stream is not None:
        stdin_stream.write(request_line)
        stdin_stream.flush()

    # Read response with timeout
    try:
        if stdout_stream is not None:
            response_line = await asyncio.wait_for(asyncio.to_thread(stdout_stream.readline), timeout=10.0)
        else:
            response_line = None

        if response_line:
            print(f"Got response: {response_line}")
            response = json.loads(response_line)

            if "result" in response:
                print("✓ Server initialized successfully!")

                # Send initialized notification
                notif = {"jsonrpc": "2.0", "method": "notifications/initialized"}
                if stdin_stream is not None:
                    stdin_stream.write(json.dumps(notif) + "\n")
                    stdin_stream.flush()

                await asyncio.sleep(0.5)

                # Now try to list tools
                print("\nListing tools...")
                tools_request = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}

                if stdin_stream is not None:
                    stdin_stream.write(json.dumps(tools_request) + "\n")
                    stdin_stream.flush()

                if stdout_stream is not None:
                    tools_response_line = await asyncio.wait_for(
                        asyncio.to_thread(stdout_stream.readline), timeout=5.0
                    )
                else:
                    tools_response_line = None

                if tools_response_line:
                    tools_response = json.loads(tools_response_line)
                    if "result" in tools_response and "tools" in tools_response["result"]:
                        tools = tools_response["result"]["tools"]
                        print(f"✓ Found {len(tools)} tools:")
                        for tool in tools:
                            print(f"  - {tool['name']}: {tool['description'][:60]}...")
                    else:
                        print("✗ No tools found in response")
                        print(f"Tools response: {tools_response}")
                else:
                    print("✗ No response to tools/list")

            else:
                print("✗ Server returned error:", response.get("error"))
        else:
            print("✗ No response from server")

    except asyncio.TimeoutError:
        print("✗ Timeout waiting for response")
        # Try to read any error output
        try:
            stderr_stream = process.stderr
            if stderr_stream is not None:
                stderr_stream_with_timeout = stderr_stream  # Type narrowing
                # Can't use settimeout on pipe, so just try to read
                stderr = stderr_stream_with_timeout.read()
                if stderr:
                    print("STDERR:", stderr)
        except Exception:
            pass

    # Cleanup
    print("\nTerminating server...")
    try:
        process.terminate()
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


if __name__ == "__main__":
    asyncio.run(test_mcp_server())
