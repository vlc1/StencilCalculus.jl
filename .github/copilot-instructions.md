@./AGENTS.md

## MCP Servers

This project uses the [julia-mcp](https://github.com/aplavin/julia-mcp) MCP server
for efficient Julia code execution (persistent sessions, no TTFX per call).

### VS Code Copilot — add to `.vscode/settings.json`

```json
{
  "mcp": {
    "servers": {
      "julia": {
        "command": "uv",
        "args": ["run", "--directory", "/path/to/julia-mcp", "python", "server.py"]
      }
    }
  }
}
```

Replace `/path/to/julia-mcp` with the directory where you cloned
<https://github.com/aplavin/julia-mcp>.

### Verify

Open the Copilot Chat panel and confirm the `julia_eval` tool is listed under
available MCP tools before starting work.
