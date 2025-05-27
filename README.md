# AI-Powered-Biosensing


### Description


### Getting Started
1. Install uv e.g. `pip install uv`
1.a if "uv is not recognized ..." error: ensure correct <python-installation>/Scripts path is added to PATH. Use `pip show uv` to show location, then add that path without the last folder but with "Scripts" to PATH.

2. Execute 
`uv sync`
to create a virtual environment and install dependencies, then 
`pip install -e .`
to enable running the notebooks.

3. Clone and initialize SAM2 Repository (instructions from the official SAM2 repo)
`git clone https://github.com/facebookresearch/sam2.git`
`cd sam2`
`pip install -e .`

4. Optional: download SAM2 checkpoints:
```cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

5. Optional: initialize training scripts
```
cd training && \
pip install -e ".[dev]"
```
