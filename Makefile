# The loom website lives under docs/ ( VitePress ), with its own package.json, so the project root
# stays a Python package root. `npm install` is idempotent, so there is no separate step to remember.

.PHONY: help site site-build anim kernels wheels

help:
	@echo "  make site        the website, live reloading   ( docs/, VitePress )"
	@echo "  make site-build  the website into docs/.vitepress/dist"
	@echo "  make kernels     measure the choices ( src/sdot/choices/ ) and build the kernels ( src/sdot/precompiled-kernels/ ), on the hardware here"
	@echo "  make wheels      the loom + sdot wheels under dist/, validated ( scripts/wheels.py --publish URL uploads them )"
	@echo "  make anim        regenerate the tutorials' animations ( docs/public/anim/*.gif ) -- needs jax + matplotlib"

site: docs/node_modules
	cd docs && npm run dev

site-build: docs/node_modules
	cd docs && npm run build

docs/node_modules: docs/package.json
	cd docs && npm install
	@touch $@


# ── the wheels ( for the OS / CPU family of this machine ) ──
#   make kernels   measure the choices + compile the kernels for every architecture the hardware here can be ( sdot.catalogue )
#   make wheels    build + validate the wheels into dist/ ( scripts/wheels.py -h for the platform tag and publishing )

PYTHON ?= python

kernels:
	$(PYTHON) -m sdot.catalogue

wheels:
	$(PYTHON) scripts/wheels.py

publish:
	$(PYTHON) scripts/wheels.py --publish https://upload.pypi.org/legacy/
