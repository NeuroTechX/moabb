(() => {
  const macroLabels = {
    "motor-imagery-datasets": "Imagery",
    "erp-p300-datasets": "ERP/P300",
    "ssvep-datasets": "SSVEP",
    "c-vep-datasets": "c-VEP",
    "resting-state-datasets": "Resting State",
    "compound-datasets": "Compound",
  };

  function collectDatasetLinks(heading) {
    const links = [];
    const seen = new Set();
    let cursor = heading.nextElementSibling;

    while (cursor) {
      if (cursor.matches("h2, h3")) {
        break;
      }

      if (cursor.matches("table.autosummary")) {
        const datasetAnchors = cursor.querySelectorAll(
          "tbody tr td:first-child a.reference.internal",
        );
        datasetAnchors.forEach((anchor) => {
          const href = anchor.getAttribute("href");
          const name = anchor.textContent.trim();
          if (!href || !name || seen.has(href)) {
            return;
          }
          seen.add(href);
          links.push({ href, name });
        });
      }

      cursor = cursor.nextElementSibling;
    }

    return links;
  }

  function addDatasetGroupToToc(sectionId, links) {
    const tocLink = document.querySelector(`.bd-toc nav a[href="#${sectionId}"]`);
    if (!tocLink) {
      return;
    }

    const macroLabel = macroLabels[sectionId];
    if (macroLabel) {
      tocLink.textContent = macroLabel;
    }

    const tocItem = tocLink.closest("li");
    if (!tocItem) {
      return;
    }

    const existingGroup = Array.from(tocItem.children).find(
      (child) =>
        child.tagName === "UL" &&
        Array.from(child.children).some((li) => li.classList?.contains("toc-dataset-group")),
    );
    if (existingGroup) {
      return;
    }

    let nested = Array.from(tocItem.children).find((child) => child.tagName === "UL");
    if (!nested) {
      nested = document.createElement("ul");
      nested.className = "nav section-nav flex-column";
      tocItem.appendChild(nested);
    }

    const groupItem = document.createElement("li");
    groupItem.className = "toc-h4 nav-item toc-entry toc-dataset-group";

    const groupLabel = document.createElement("span");
    groupLabel.className = "nav-link toc-dataset-group-label";
    groupLabel.textContent = "Datasets";
    groupItem.appendChild(groupLabel);

    const datasetList = document.createElement("ul");
    datasetList.className = "nav section-nav flex-column toc-dataset-list";

    links.forEach(({ href, name }) => {
      const datasetItem = document.createElement("li");
      datasetItem.className = "toc-h5 nav-item toc-entry";

      const datasetLink = document.createElement("a");
      datasetLink.className = "reference internal nav-link";
      datasetLink.href = href;
      datasetLink.textContent = name;

      datasetItem.appendChild(datasetLink);
      datasetList.appendChild(datasetItem);
    });

    groupItem.appendChild(datasetList);
    nested.appendChild(groupItem);
  }

  function init() {
    if (!document.querySelector("#api-and-main-concepts")) {
      return;
    }

    Object.keys(macroLabels).forEach((sectionId) => {
      const heading = document.getElementById(sectionId);
      if (!heading) {
        return;
      }
      const datasetLinks = collectDatasetLinks(heading);
      if (datasetLinks.length) {
        addDatasetGroupToToc(sectionId, datasetLinks);
      }
    });
  }

  function stripDatasetPrefix(text) {
    return text
      .replace(/^moabb\.datasets\.compound_dataset\./, "")
      .replace(/^moabb\.datasets\./, "");
  }

  function initGeneratedDatasetSidebarHierarchy() {
    const pagename =
      (window.DOCUMENTATION_OPTIONS && window.DOCUMENTATION_OPTIONS.pagename) || "";
    const path = (window.location && window.location.pathname) || "";
    const isGeneratedDatasetPage =
      pagename.startsWith("generated/moabb.datasets.") ||
      /\/generated\/moabb\.datasets\..+\.html$/i.test(path);
    if (!isGeneratedDatasetPage) {
      return;
    }

    const navContainer = document.querySelector(
      'nav.bd-docs-nav[aria-label="Section Navigation"] .bd-toc-item.navbar-nav',
    );
    if (!navContainer) {
      return;
    }

    const hasGroupedRoot = Array.from(navContainer.children).some(
      (child) => child.tagName === "UL" && child.classList.contains("toc-datasets-root"),
    );
    if (hasGroupedRoot) {
      return;
    }

    const groups = Array.from(navContainer.children).filter(
      (child) =>
        child.tagName === "UL" &&
        child.classList.contains("nav") &&
        child.classList.contains("bd-sidenav"),
    );
    if (groups.length < 6) {
      return;
    }

    const datasetGroupLabels = [
      "Imagery",
      "ERP/P300",
      "SSVEP",
      "c-VEP",
      "Resting State",
      "Compound",
    ];

    const rootList = document.createElement("ul");
    rootList.className = "nav bd-sidenav toc-datasets-root";

    const datasetsLi = document.createElement("li");
    datasetsLi.className = "toctree-l1 toc-group-datasets";
    const datasetsLabel = document.createElement("span");
    datasetsLabel.className = "reference internal toc-group-label";
    datasetsLabel.textContent = "Datasets";
    datasetsLi.appendChild(datasetsLabel);

    const subgroupList = document.createElement("ul");
    subgroupList.className = "nav bd-sidenav";

    groups.slice(0, 6).forEach((group, index) => {
      const subgroupLi = document.createElement("li");
      subgroupLi.className = "toctree-l2 toc-group-paradigm";

      const subgroupLabel = document.createElement("span");
      subgroupLabel.className = "reference internal toc-subgroup-label";
      subgroupLabel.textContent = datasetGroupLabels[index];
      subgroupLi.appendChild(subgroupLabel);

      const datasetList = document.createElement("ul");
      datasetList.className = "nav bd-sidenav";

      Array.from(group.children).forEach((item) => {
        if (!(item instanceof HTMLLIElement)) {
          return;
        }
        item.classList.remove("toctree-l1");
        item.classList.add("toctree-l3", "toc-dataset-item");
        const link = item.querySelector("a.reference.internal");
        if (link) {
          link.textContent = stripDatasetPrefix(link.textContent.trim());
        }
        datasetList.appendChild(item);
      });

      subgroupLi.appendChild(datasetList);
      subgroupList.appendChild(subgroupLi);
      group.remove();
    });

    datasetsLi.appendChild(subgroupList);
    rootList.appendChild(datasetsLi);
    navContainer.insertBefore(rootList, navContainer.firstChild);
  }

  function runEnhancers() {
    try {
      init();
      initGeneratedDatasetSidebarHierarchy();
    } catch (error) {
      console.error("[section-nav-hierarchy] failed:", error);
    }
  }

  function boot() {
    runEnhancers();
    setTimeout(runEnhancers, 150);
    setTimeout(runEnhancers, 500);
    setTimeout(runEnhancers, 1200);

    // PyData theme can reshape sidebars after load; keep retrying briefly.
    const startedAt = Date.now();
    const timer = setInterval(() => {
      runEnhancers();
      const grouped = document.querySelector("nav.bd-docs-nav ul.toc-datasets-root");
      if (grouped || Date.now() - startedAt > 8000) {
        clearInterval(timer);
      }
    }, 250);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot, { once: true });
  } else {
    boot();
  }
})();
