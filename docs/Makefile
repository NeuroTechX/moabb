# Minimal Makefile for Sphinx documentation
#
# Variables (can be overridden from the command line)
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = moabb
SOURCEDIR     = source
BUILDDIR      = build

.PHONY: all copy-readme help apidoc html html-noplot

# Default target: run copy-readme then build HTML.
all: copy-readme html

# Copy target: copy README.md from the project root (../)
# into docs/source.
copy-readme:
	@echo "Copying README.md into $(SOURCEDIR)..."
	@cp ../README.md $(SOURCEDIR)/README.md

clean:
	-rm -rf build source/auto_examples source/generated source/results

# Help target.
help: copy-readme
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Sphinx apidoc target.
apidoc: copy-readme
	sphinx-apidoc -o "$(SOURCEDIR)" ../"$(SPHINXPROJ)" --module-first -d 1

# HTML build target.
html: copy-readme
	@python prepare_summary_tables.py ../moabb/datasets $(BUILDDIR)
	@python prepare_paper_results_tables.py ../results $(SOURCEDIR)/results
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# HTML build target without plot gallery.
html-noplot: copy-readme
	@python prepare_summary_tables.py ../moabb/datasets $(BUILDDIR)
	@python prepare_paper_results_tables.py ../results $(SOURCEDIR)/results
	@$(SPHINXBUILD) -D plot_gallery=0 -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."
