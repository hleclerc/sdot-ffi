# The loom website lives under docs/ ( VitePress ), with its own package.json, so the project root
# stays a Python package root. `npm install` is idempotent, so there is no separate step to remember.

.PHONY: help site site-build anim local_kernels wheels local_wheels fetch_linux publish

help:
	@echo "  make site        the website, live reloading   ( docs/, VitePress )"
	@echo "  make site-build  the website into docs/.vitepress/dist"
	@echo "  make local_kernels measure the choices ( src/sdot/choices/ ) and build the kernels ( src/sdot/precompiled-kernels/ ), on the hardware here"
	@echo "  make wheels      the loom + sdot wheels under dist/, validated ( scripts/wheels.py --publish URL uploads them )"
	@echo "  make local_wheels  the wheels of the machine / environment this runs in, no upload -- fan out with: errand --env=jax,lmo-jax,win-jax -- make local_wheels"
	@echo "  make fetch_linux   the manylinux wheel built on $(LINUX_HOST) ( by local_wheels there ) into dist/"
	@echo "  make publish     the same, then upload everything in dist/ ( TWINE_PASSWORD = your PyPI token )"
	@echo "  make anim        regenerate the tutorials' animations ( docs/public/anim/*.gif ) -- needs jax + matplotlib"

site: docs/node_modules
	cd docs && npm run dev

site-build: docs/node_modules
	cd docs && npm run build

docs/node_modules: docs/package.json
	cd docs && npm install
	@touch $@


# ── the wheels ( for the OS / CPU family of this machine ) ──
#   make local_kernels  measure the choices + compile the kernels for every architecture the hardware here can be ( sdot.catalogue )
#   make wheels    build + validate the wheels into dist/ ( scripts/wheels.py -h for the platform tag and publishing )

PYTHON ?= python

local_kernels:
	$(PYTHON) -m sdot.catalogue

wheels:
	$(PYTHON) scripts/wheels.py

# What one environment can build, by itself: its kernels ( make local_kernels ), then its wheels. errand runs it
# in as many places as asked, each on its own copy of the tree:
#   errand --env=jax,lmo-jax,win-jax -- make local_wheels
# On Linux it is the manylinux build ( scripts/linux_wheels.sh, in the image ); the wheel is then in dist/ OF THAT MACHINE:
# `make fetch_linux` brings it here.
ifeq ($(shell uname),Linux)
local_wheels:
	bash scripts/linux_wheels.sh
else
local_wheels: local_kernels wheels
endif

LINUX_HOST ?= lmo
LINUX_DIST ?= Projects-rsync/sdot-ffi/dist
fetch_linux:
	mkdir -p dist
	rsync -av --include='*manylinux*.whl' --exclude='*' $(LINUX_HOST):$(LINUX_DIST)/ dist/

publish:
	$(PYTHON) scripts/wheels.py --publish https://upload.pypi.org/legacy/
