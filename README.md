# MoEx: Distributed Mixture-of-Experts Inference on Consumer Devices via WebGPU

[![DOI](https://zenodo.org/badge/1163965029.svg)](https://doi.org/10.5281/zenodo.18732132)

**Browser-Based Expert FFN Disaggregation with Hedged Dispatch and Binary Transport**

<p align="center">
  <img src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEgqO627KWWvMzIDJ6I90cqH6nuJ63Td6Xytcz2mocTAgroQE2gVMVQo9W-uaPdWueWr9DDgOo2pFJX_-CkSijopo2H6nBAzfc-oZUXKsiqXRiG1-OMkycLme_JGXTlzyPxj5d2rrfW8g1k/s450/job_maid_meido_kissa.png" alt="MoEx" width="200">
</p>

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
  doi     = {10.5281/zenodo.18732132},
}
```

## License

MIT
