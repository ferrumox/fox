# Integrations

fox works out of the box with any tool that supports the OpenAI or Ollama API. This page covers the most common integrations with configuration examples.

---

## OpenAI Python SDK

Install:

```bash
pip install openai
```

Configure the client to point at fox:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="none"  # any non-empty string
)
```

That is the only change needed. All existing code that uses the OpenAI SDK works as-is.

### Chat completion

```python
response = client.chat.completions.create(
    model="llama3.2",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    max_tokens=256,
    temperature=0.7
)
print(response.choices[0].message.content)
```

### Streaming

```python
with client.chat.completions.stream(
    model="llama3.2",
    messages=[{"role": "user", "content": "Tell me a story about a robot"}],
    max_tokens=512
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
print()
```

### Embeddings

```python
response = client.embeddings.create(
    model="llama3.2",
    input=["Hello world", "Another sentence"]
)
for item in response.data:
    print(f"Index {item.index}: {len(item.embedding)} dimensions")
```

### Function calling

```python
import json

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"]
        }
    }
}]

response = client.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "What's the weather in Madrid?"}],
    tools=tools,
    tool_choice="auto"
)

choice = response.choices[0]
if choice.finish_reason == "tool_calls":
    tool_call = choice.message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    print(f"Tool: {tool_call.function.name}")
    print(f"Args: {args}")
```

---

## OpenAI JavaScript / TypeScript SDK

Install:

```bash
npm install openai
```

```typescript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:8080/v1",
  apiKey: "none",
});

// Non-streaming
const response = await client.chat.completions.create({
  model: "llama3.2",
  messages: [{ role: "user", content: "Hello!" }],
});
console.log(response.choices[0].message.content);

// Streaming
const stream = await client.chat.completions.create({
  model: "llama3.2",
  messages: [{ role: "user", content: "Count to 10" }],
  stream: true,
});
for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0]?.delta?.content ?? "");
}
```

---

## LangChain (Python)

Install:

```bash
pip install langchain langchain-openai
```

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

llm = ChatOpenAI(
    base_url="http://localhost:8080/v1",
    api_key="none",
    model="llama3.2"
)

embeddings = OpenAIEmbeddings(
    base_url="http://localhost:8080/v1",
    api_key="none",
    model="llama3.2"
)
```

### Basic chat

```python
from langchain_core.messages import SystemMessage, HumanMessage

response = llm.invoke([
    SystemMessage(content="You are a concise assistant."),
    HumanMessage(content="What is Rust?")
])
print(response.content)
```

### Streaming

```python
for chunk in llm.stream("Explain ownership in Rust, step by step"):
    print(chunk.content, end="", flush=True)
```

### Chains with prompt templates

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a patient tutor."),
    ("user", "Explain {topic} to a beginner in three paragraphs.")
])

chain = prompt | llm | StrOutputParser()
result = chain.invoke({"topic": "async programming"})
print(result)
```

### RAG with embeddings

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import numpy as np

embeddings = OpenAIEmbeddings(
    base_url="http://localhost:8080/v1",
    api_key="none",
    model="llama3.2"
)

documents = [
    Document(page_content="Rust uses ownership to manage memory without a garbage collector."),
    Document(page_content="Python uses reference counting and a cyclic garbage collector."),
    Document(page_content="Go uses a mark-and-sweep garbage collector with very short pause times."),
]

vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

rag_prompt = ChatPromptTemplate.from_template(
    "Answer based on the context:\n\n{context}\n\nQuestion: {question}"
)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("How does Rust manage memory?")
print(answer)
```

### Structured output

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

parser = JsonOutputParser()

prompt = PromptTemplate(
    template="Return a JSON object with keys 'language', 'year_created', 'creator'.\n{format_instructions}\nLanguage: {language}",
    input_variables=["language"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | llm | parser
result = chain.invoke({"language": "Rust"})
print(result)  # {'language': 'Rust', 'year_created': 2010, 'creator': 'Graydon Hoare'}
```

---

## LangChain.js

Install:

```bash
npm install @langchain/openai @langchain/core
```

```typescript
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";

const llm = new ChatOpenAI({
  configuration: { baseURL: "http://localhost:8080/v1" },
  apiKey: "none",
  model: "llama3.2",
  temperature: 0.7,
});

const response = await llm.invoke([
  new SystemMessage("You are a helpful assistant."),
  new HumanMessage("What is TypeScript?"),
]);
console.log(response.content);

// Streaming
const stream = await llm.stream("Tell me about Rust");
for await (const chunk of stream) {
  process.stdout.write(chunk.content as string);
}
```

---

## Open WebUI

Open WebUI is a browser-based chat interface. It supports both OpenAI and Ollama backends.

### Docker (quickest)

```bash
# Start fox first
fox serve --max-models 3

# Start Open WebUI
docker run -d \
  --name open-webui \
  -p 3000:8080 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:8080 \
  --add-host=host.docker.internal:host-gateway \
  ghcr.io/open-webui/open-webui:main
```

Open `http://localhost:3000` in your browser.

### Docker Compose

```yaml
services:
  fox:
    image: ferrumox/fox:latest
    ports:
      - "8080:8080"
    volumes:
      - ~/.cache/ferrumox:/root/.cache/ferrumox
    environment:
      FOX_MAX_MODELS: "3"

  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    ports:
      - "3000:8080"
    environment:
      OLLAMA_BASE_URL: "http://fox:8080"
    depends_on:
      - fox
```

### What works

| Feature | Status |
|---------|--------|
| Model list | ✓ |
| Chat (streaming) | ✓ |
| RAG / Embeddings | ✓ |
| Function calling | ✓ (via OpenAI endpoint) |
| Image upload | — (text models only) |
| Model pulling via UI | ✓ |

Models appear in the model selector as their filename stems. Use [aliases](./configuration.md#aliases) to give them shorter names.

---

## Continue.dev

[Continue](https://continue.dev/) is a VS Code and JetBrains extension for AI-assisted coding. It supports the Ollama backend.

Edit `~/.continue/config.json`:

```json
{
  "models": [
    {
      "title": "fox / llama3.2",
      "provider": "ollama",
      "model": "llama3.2",
      "apiBase": "http://localhost:8080"
    }
  ],
  "tabAutocompleteModel": {
    "title": "fox / qwen2.5-coder",
    "provider": "ollama",
    "model": "qwen2.5-coder",
    "apiBase": "http://localhost:8080"
  }
}
```

---

## LlamaIndex (Python)

Install:

```bash
pip install llama-index-llms-openai llama-index-embeddings-openai
```

```python
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

Settings.llm = OpenAI(
    model="llama3.2",
    api_base="http://localhost:8080/v1",
    api_key="none"
)

Settings.embed_model = OpenAIEmbedding(
    model="llama3.2",
    api_base="http://localhost:8080/v1",
    api_key="none"
)
```

---

## Cursor

In Cursor's settings, add a custom model:

1. Open **Settings → Models**
2. Add a new model with the OpenAI base URL: `http://localhost:8080/v1`
3. Set any non-empty API key
4. Set the model name to match your alias or filename stem

---

## Ollama CLI

The Ollama CLI can point at any Ollama-compatible server:

```bash
export OLLAMA_HOST=http://localhost:8080

# Chat
ollama run llama3.2

# List models
ollama list

# Pull a model (triggers fox's /api/pull)
ollama pull llama3.2
```

---

## curl

A quick reference for testing endpoints directly:

```bash
HOST=http://localhost:8080
MODEL=llama3.2

# Health check
curl $HOST/health

# List models
curl $HOST/v1/models

# Chat (non-streaming)
curl $HOST/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}"

# Chat (streaming)
curl $HOST/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"stream\":true}"

# Embeddings
curl $HOST/v1/embeddings \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"input\":\"Hello world\"}"

# Ollama chat
curl $HOST/api/chat \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"stream\":false}"

# Prometheus metrics
curl $HOST/metrics
```
