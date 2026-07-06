/*
 * sortable-numeric.js
 *
 * DataTables custom column type: "num-varies".
 *
 * The paradigm summary tables on dataset_summary.html are made sortable by a
 * single shared `$('.sortable').DataTable()` call with no per-column config, so
 * DataTables has to auto-detect each column's type. Its numeric detector is
 * all-or-nothing: one non-numeric cell downgrades the whole column to string
 * (lexicographic) sorting. Several numeric columns ("Total_trials",
 * "#Trials / class", "#Runs") carry the sentinel "varies" for datasets whose
 * trial count is not fixed, which silently broke numeric sorting for them
 * (e.g. Total_trials ordered 11000, 1114, 11496, 1200, ...).
 *
 * This plugin registers a column type that treats a column as numeric when
 * every cell is EITHER a number OR a known sentinel, ordering the sentinel
 * rows as -Infinity so they sink to the bottom of a largest-first sort while
 * still DISPLAYING their original text ("varies") to the reader.
 */
(function () {
  "use strict";

  var DataTable = jQuery.fn.dataTable;

  // Tokens that stand in for "no single numeric value". Matched case-folded and
  // trimmed; the empty string counts too, so one blank cell can't silently
  // re-trigger the string-sort bug.
  var SENTINELS = ["varies", "n/a", "na", "-", "—", ""];

  // Strip any HTML DataTables may hand us and normalise for comparison.
  function clean(value) {
    return String(value).replace(/<[^>]*>/g, "").trim();
  }

  function isSentinel(text) {
    return SENTINELS.indexOf(text.toLowerCase()) !== -1;
  }

  function asNumber(text) {
    return parseFloat(text.replace(/,/g, ""));
  }

  DataTable.type("num-varies", {
    // Claim a column only if EVERY cell is a number or a known sentinel.
    // A named custom type is always checked before the built-in "string"
    // fallback, so a mixed number/"varies" column lands here rather than being
    // sorted lexicographically.
    detect: function (value) {
      var text = clean(value);
      return isSentinel(text) || !isNaN(asNumber(text));
    },
    // Ordering key: real numbers sort normally; sentinels pin to the bottom.
    order: {
      pre: function (value) {
        var text = clean(value);
        if (isSentinel(text)) {
          return -Infinity;
        }
        var n = asNumber(text);
        return isNaN(n) ? -Infinity : n;
      },
    },
  });
})();
