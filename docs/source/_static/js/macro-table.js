/**
 * MOABB Macro Table — DataTables initialization.
 *
 * Uses DataTables extensions: Buttons (CSV, ColVis, Print),
 * SearchPanes, Select, FixedHeader.
 */
document.addEventListener("DOMContentLoaded", function () {
  var $table = $("#moabb-macro-table");
  if (!$table.length) return;

  // --- Column config matching Python _TABLE_COLUMNS order ---
  var columns = [
    // Visible by default
    /* 0  */ { name: "Dataset",       visible: true,  type: "string" },
    /* 1  */ { name: "Paradigm",      visible: true,  type: "string",  pane: true },
    /* 2  */ { name: "#Subj",         visible: true,  type: "num" },
    /* 3  */ { name: "#Chan",         visible: true,  type: "num" },
    /* 4  */ { name: "#EEG",          visible: true,  type: "num" },
    /* 5  */ { name: "#Classes",      visible: true,  type: "num" },
    /* 6  */ { name: "Freq",          visible: true,  type: "num" },
    /* 7  */ { name: "Trial",         visible: true,  type: "num" },
    /* 8  */ { name: "#Sess",         visible: true,  type: "num" },
    /* 9  */ { name: "#Runs",         visible: true,  type: "num" },
    /* 10 */ { name: "Health",        visible: true,  type: "string",  pane: true },
    /* 11 */ { name: "#Trials",       visible: true,  type: "string" },
    /* 12 */ { name: "Country",       visible: true,  type: "string",  pane: true },
    /* 13 */ { name: "Year",          visible: true,  type: "num" },
    /* 14 */ { name: "DOI",           visible: true,  type: "string" },
    // Hidden by default
    /* 15 */ { name: "Class Labels",  visible: false, type: "string" },
    /* 16 */ { name: "Stimulus",      visible: false, type: "string",  pane: true },
    /* 17 */ { name: "Modality",      visible: false, type: "string",  pane: true },
    /* 18 */ { name: "Feedback",      visible: false, type: "string",  pane: true },
    /* 19 */ { name: "Sync",          visible: false, type: "string",  pane: true },
    /* 20 */ { name: "Mode",          visible: false, type: "string",  pane: true },
    /* 21 */ { name: "Study Design",  visible: false, type: "string" },
    /* 22 */ { name: "Hardware",      visible: false, type: "string",  pane: true },
    /* 23 */ { name: "Reference",     visible: false, type: "string" },
    /* 24 */ { name: "Sensor Type",   visible: false, type: "string",  pane: true },
    /* 25 */ { name: "Montage",       visible: false, type: "string" },
    /* 26 */ { name: "Line Freq",     visible: false, type: "num" },
    /* 27 */ { name: "Filters",       visible: false, type: "string" },
    /* 28 */ { name: "Cap Mfr",       visible: false, type: "string" },
    /* 29 */ { name: "Software",      visible: false, type: "string" },
    /* 30 */ { name: "Clinical Pop.", visible: false, type: "string" },
    /* 31 */ { name: "Age Mean",      visible: false, type: "num" },
    /* 32 */ { name: "Age Min",       visible: false, type: "num" },
    /* 33 */ { name: "Age Max",       visible: false, type: "num" },
    /* 34 */ { name: "Gender",        visible: false, type: "string" },
    /* 35 */ { name: "Handedness",    visible: false, type: "string" },
    /* 36 */ { name: "BCI Exp.",      visible: false, type: "string",  pane: true },
    /* 37 */ { name: "License",       visible: false, type: "string",  pane: true },
    /* 38 */ { name: "Institution",   visible: false, type: "string" },
    /* 39 */ { name: "Repository",    visible: false, type: "string",  pane: true },
    /* 40 */ { name: "Duration (h)",  visible: false, type: "num" },
    /* 41 */ { name: "Author",        visible: false, type: "string" },
    /* 42 */ { name: "Data URL",      visible: false, type: "string" },
    /* 43 */ { name: "Pathology Tags",visible: false, type: "string",  pane: true },
    /* 44 */ { name: "Modality Tags", visible: false, type: "string",  pane: true },
    /* 45 */ { name: "Type Tags",     visible: false, type: "string",  pane: true },
    /* 46 */ { name: "File Format",   visible: false, type: "string",  pane: true },
    /* 47 */ { name: "EOG",           visible: false, type: "string" },
    /* 48 */ { name: "EMG",           visible: false, type: "string" },
    /* 49 */ { name: "#Blocks",       visible: false, type: "num" },
    /* 50 */ { name: "Trials Context",visible: false, type: "string" },
    /* 51 */ { name: "Stim. Freqs",   visible: false, type: "string" },
    /* 52 */ { name: "Code Type",     visible: false, type: "string",  pane: true },
    /* 53 */ { name: "#Targets",      visible: false, type: "num" },
    /* 54 */ { name: "#Repetitions",  visible: false, type: "num" },
    /* 55 */ { name: "ISI (ms)",      visible: false, type: "num" },
    /* 56 */ { name: "SOA (ms)",      visible: false, type: "num" },
    /* 57 */ { name: "MI Tasks",      visible: false, type: "string" },
  ];

  var COL_PARADIGM = 1;
  var COL_SUBJ = 2;
  var COL_HEALTH = 10;
  var COL_COUNTRY = 12;

  // Build column indices dynamically
  var hiddenCols = [];
  var numericCols = [];
  var paneCols = [];

  columns.forEach(function (col, i) {
    if (!col.visible) hiddenCols.push(i);
    if (col.type === "num") numericCols.push(i);
    if (col.pane) paneCols.push(i);
  });

  // Columns containing HTML tags (need stripping for sort/filter)
  var htmlCols = [COL_PARADIGM, COL_HEALTH];

  function stripHtml(html) {
    var tmp = document.createElement("div");
    tmp.innerHTML = html;
    return (tmp.textContent || tmp.innerText || "").trim();
  }

  var columnDefs = [
    { targets: hiddenCols, visible: false },
    { targets: numericCols, type: "num" },
    {
      targets: htmlCols,
      render: function (data, type) {
        if (type === "display") return data;
        return stripHtml(data);
      },
    },
    { targets: "_all", searchPanes: { show: false } },
    { targets: paneCols, searchPanes: { show: true } },
  ];

  var table = $table.DataTable({
    dom: "Blfrtip",
    paging: false,
    ordering: true,
    orderMulti: true,
    order: [[0, "asc"]],
    fixedHeader: true,
    columnDefs: columnDefs,
    buttons: [
      {
        extend: "searchPanes",
        text: "Filter",
        config: {
          cascadePanes: true,
          viewTotal: true,
          layout: "columns-4",
          orderable: false,
          columns: paneCols,
        },
      },
      {
        extend: "colvis",
        text: "Columns",
        columns: ":gt(0)",
      },
      {
        extend: "csv",
        text: "CSV",
        exportOptions: { orthogonal: "export" },
      },
      {
        extend: "print",
        text: "Print",
        exportOptions: { orthogonal: "export" },
      },
    ],
    language: {
      searchPanes: {
        title: { _: "%d Filters Active", 0: "" },
      },
    },
  });

  // ---- Paradigm bar: click to filter ----

  document.querySelectorAll(".mt-bar-seg").forEach(function (seg) {
    seg.style.cursor = "pointer";
    seg.addEventListener("click", function () {
      var title = seg.getAttribute("title") || "";
      // Extract paradigm label from title like "Motor Imagery: 53 datasets (36%)"
      var label = title.split(":")[0].trim();
      if (label) {
        table.column(COL_PARADIGM).search(label).draw();
      }
    });
  });

  document.querySelectorAll(".mt-bar-legend-item").forEach(function (item) {
    item.style.cursor = "pointer";
    item.addEventListener("click", function () {
      var text = item.textContent.trim();
      // Extract label from "Motor Imagery (53)"
      var label = text.replace(/\s*\(\d+\)$/, "").trim();
      if (label) {
        table.column(COL_PARADIGM).search(label).draw();
      }
    });
  });

  // ---- Paradigm tags in table: click to filter ----

  $table.on("click", ".mt-tag[data-paradigm]", function () {
    var paradigm = stripHtml(this.innerHTML);
    table.column(COL_PARADIGM).search(paradigm).draw();
  });

  // Make paradigm tags look clickable
  $table.find(".mt-tag[data-paradigm]").css("cursor", "pointer");

  // ---- Dynamic summary card + bar updates on filter ----

  function updateSummaryCards() {
    var filteredData = table.rows({ search: "applied" }).data();
    var totalDatasets = filteredData.length;
    var totalSubjects = 0;
    var paradigms = {};
    var countries = {};

    filteredData.each(function (row) {
      totalSubjects += parseFloat(row[COL_SUBJ]) || 0;

      var p = stripHtml(row[COL_PARADIGM]);
      if (p) paradigms[p] = (paradigms[p] || 0) + 1;

      var c = (row[COL_COUNTRY] || "").trim();
      if (c) countries[c] = true;
    });

    // Update cards
    var cards = document.querySelectorAll("#mt-cards .mt-card");
    if (cards.length >= 4) {
      cards[0].querySelector(".mt-card-value").textContent = totalDatasets;
      cards[1].querySelector(".mt-card-value").textContent =
        totalSubjects.toLocaleString();
      cards[2].querySelector(".mt-card-value").textContent =
        Object.keys(paradigms).length;
      cards[3].querySelector(".mt-card-value").textContent =
        Object.keys(countries).length;
    }

    // Update paradigm bar
    var barSegs = document.querySelectorAll(".mt-bar-seg");
    var legendItems = document.querySelectorAll(".mt-bar-legend-item");
    if (barSegs.length > 0) {
      barSegs.forEach(function (seg) {
        var title = seg.getAttribute("title") || "";
        var label = title.split(":")[0].trim();
        var count = paradigms[label] || 0;
        var pct = totalDatasets > 0 ? (count / totalDatasets) * 100 : 0;
        seg.style.width = pct.toFixed(1) + "%";
        seg.setAttribute(
          "title",
          label + ": " + count + " datasets (" + Math.round(pct) + "%)"
        );
      });
    }
    legendItems.forEach(function (item) {
      var text = item.textContent.trim();
      var label = text.replace(/\s*\(\d+\)$/, "").trim();
      var count = paradigms[label] || 0;
      // Update the text node after the dot span
      var dot = item.querySelector(".mt-bar-dot");
      if (dot) {
        item.textContent = "";
        item.appendChild(dot);
        item.appendChild(document.createTextNode(" " + label + " (" + count + ")"));
      }
    });
  }

  table.on("search.dt", updateSummaryCards);

  // ---- Reset filter on double-click bar/legend ----

  document.querySelectorAll(".mt-bar-seg, .mt-bar-legend-item").forEach(function (el) {
    el.addEventListener("dblclick", function () {
      table.column(COL_PARADIGM).search("").draw();
    });
  });
});
