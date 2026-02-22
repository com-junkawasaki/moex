# MoEx: Distributed Mixture-of-Experts Inference on Consumer Devices via WebGPU

**Browser-Based Expert FFN Disaggregation with Hedged Dispatch and Binary Transport**

## Overview

MoEx disaggregates MoE language models (e.g., Qwen3-30B-A3B) into:

- **Hub** — executes attention, layer norms, router, embeddings, LM head (~6B shared params)
- **Workers** — browser tabs running WebGPU compute shaders for expert FFN blocks (~24B expert params, INT4 quantized)

Communication uses a binary transport protocol over WebSocket with **hedged dispatch** for tail-latency mitigation.

## Project Structure

```
moex/
├── main.tex            # arXiv paper (LaTeX)
├── references.bib      # Bibliography
├── Makefile             # Build paper: make
├── docs/               # GitHub Pages site
│   └── index.html
└── src/                # Reference implementation
    ├── protocol.ts     # Binary transport protocol
    ├── hub.ts          # Hub coordination engine
    ├── worker.ts       # WebGPU worker
    └── worker.html     # Worker browser UI
```

## Build Paper

```bash
make        # requires pdflatex + bibtex
```

## Citation

```bibtex
@article{kawasaki2026moex,
  title   = {MoEx: Distributed Mixture-of-Experts Inference
             on Consumer Devices via WebGPU},
  author  = {Kawasaki, Jun},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026},
}
```

## License

MIT
