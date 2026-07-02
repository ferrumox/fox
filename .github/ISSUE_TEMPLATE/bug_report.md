---
name: Bug report
about: Something isn't working
title: ''
labels: bug
assignees: ''
---

**What happened**
A clear description of the bug.

**To reproduce**
Steps or the exact command / request:
```
fox serve --model-path ...
curl ...
```

**Expected vs actual**
What you expected, and what happened instead (include error output).

**Model & environment**
- fox version (`fox --version`):
- Model (name / quantization, e.g. `llama-3.2-1b-instruct-q8_0`):
- OS & architecture:
- GPU / backend (CUDA / ROCm / Vulkan / CPU):

**`fox probe <model>` output** (if the issue involves a specific model)
```
(paste here — it shows the model's resolved facts and any contradictions)
```
