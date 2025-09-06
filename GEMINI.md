# Listen Extension — Event / Audio Queue for Gemini-CLI

This extension adds a Listen MCP server and `/listen…` commands for **hands-free input**. It lets Gemini subscribe to events (like webhooks) or audio files, enqueue them, and respond without you typing.

## Commands

- **/listen** — quick help (this doc in short form)
- **/listen:start [key=value …]** — start the listener
- **/listen:status** — show if it’s running
- **/listen:stop** — stop the listener
- **/listen:logs [lines=N]** — view recent log lines
- **/listen:clear_logs** — clear logs
- **/listen:health** — status + queue depth
- **/listen:next [timeout_ms=N]** — pull the next queued item (blocking with timeout)
- **/enqueue_audio path.wav** — enqueue an audio file manually

### Common /listen:start keys

- `http_enable` (bool, default `true`) — expose HTTP webhook endpoint
- `http_port` (int, default `8765`) — port for inbound HTTP events/audio
- `queue_dir` (path, default `~/.listen_queue`) — directory for queued items
- `language` (str, default `en`) — language for Whisper transcription
- `model` (str, default `base`) — Whisper model (tiny|base|small|medium|large)
- `delete_after` (bool, default `false`) — whether to delete audio after processing
- `poll_ms` (int, default `500`) — polling interval in ms
- `timeout_ms` (int, default `900000`) — max time per transcription

## Quick Start

1. `/listen:start http_enable=true http_port=8765`
2. Send an event or audio file to the listener (HTTP POST or `/enqueue_audio file.wav`)
3. Gemini will pick it up via `/listen:next` and respond
4. `/listen:stop` to exit

## Event Loop Behavior

- When running, the listener keeps polling for queued items.
- Each item → Gemini treats as if you typed it.  
- If item has `.text`, that is passed directly.  
- If item has `.file` (audio), it is transcribed first, then passed.  
- Gemini replies concisely to each item.  
- If TTS is available, it may also synthesize a spoken reply.

## Defaults

- Listener starts in **HTTP mode**, serving `/event` on the given port.  
- Audio transcriptions use Whisper with `fp16=false` for CPU portability.  
- If nothing arrives for ~5 minutes, the loop ends politely.  
- If the inbound text is exactly `stop | quit | goodbye`, the loop ends politely.

## Platform Notes

- **macOS**: check firewall if using HTTP listener.  
- **Linux**: make sure the chosen `http_port` is not reserved.  
- **Windows**: ensure Python has permissions to bind the port.

## Troubleshooting

- **“Connection closed”** → check venv deps: install `mcp`, `fastmcp`, `whisper`, `flask` (or aiohttp) into the extension venv.  
- **No events picked up** → confirm POSTs to `http://localhost:8765/event` (if `http_enable=true`).  
- **No transcription** → make sure Whisper CLI or `openai-whisper` Python package is installed in the venv.

## Examples

- Start listener with default options:
