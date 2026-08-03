# README demo. Everything on screen is a real session against a real GGUF model:
# fox is started, loads llama-3.2-1b, and answers both requests. Nothing is mocked.
# Regenerate with scripts/record_demo.sh.
#
# JSON bodies are written in backtick-quoted strings: VHS strings are quoted, and the
# payloads contain both ' and ", so either other quoting style fails to parse.
Output "/assets/demo.gif"

Set Shell bash
Set FontSize 16
Set Width 1280
Set Height 340
Set Padding 20
Set Theme "Catppuccin Mocha"
Set TypingSpeed 35ms

# Server start is hidden rather than faked: the recording container has no GPU, so its
# startup banner would advertise a CPU backend and undersell the thing. The requests
# below are the point anyway, and they are genuinely answered by that server.
Hide
Type "cd /demo && export LD_LIBRARY_PATH=/demo PATH=/demo:$PATH" Enter
Type "fox serve --model-path /demo/llama-3.2-1b.gguf > /tmp/fox.log 2>&1 &" Enter
Type "until curl -sf localhost:8080/health >/dev/null; do sleep 1; done; clear" Enter
Show

Sleep 800ms
Type "# One server. Two APIs. Same port." Sleep 200ms Enter
Sleep 1s

Type `curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \` Enter
Type `  -d '{"model":"llama-3.2-1b","messages":[{"role":"user","content":"Write a haiku about a fox."}]}' \` Enter
Type `  | jq -r '.choices[0].message.content'` Sleep 400ms Enter
Sleep 6s

Type "# Same port, Ollama's API. Existing clients need no code change." Sleep 200ms Enter
Sleep 1s

Type `curl -s localhost:8080/api/generate -d '{"model":"llama-3.2-1b","prompt":"A fox is","stream":false}' | jq -r .response` Sleep 400ms Enter
Sleep 12s
