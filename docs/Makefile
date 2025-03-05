# Minimal makefile for Sphinx documentation
#
# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = moabb
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: copy-readme

copy-readme:
	cp ../README.md source/README.md

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@python prepare_summary_tables.py ../moabb/datasets $(BUILDDIR)
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

apidoc: copy-readme
	sphinx-apidoc -o "$(SOURCEDIR)" ../"$(SPHINXPROJ)" --module-first -d 1

html-noplot: copy-readme
	@$(SPHINXBUILD) -D plot_gallery=0 -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."
