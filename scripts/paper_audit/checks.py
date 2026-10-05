"""Deterministic extractors for the paper audit.

Each ``find_*`` function takes the full source text and returns a list of
``(value, quote, locator)`` tuples. ``quote`` is a verbatim excerpt of the text
(so it can be re-verified by substring search) and ``locator`` is ``pN:lM``
(page/line, pages delimited by form feeds from ``pdftotext``) or ``lM``.
"""

from __future__ import annotations

import re


_UNITS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_WORD_NUM = r"(?:(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?|one hundred|hundred|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen)"
_NUM = rf"(?:\d{{1,3}}(?:,\d{{3}})+|\d+|{_WORD_NUM})"


def parse_number(token: str) -> int | None:
    """Parse ``"64"``, ``"12,500"`` or ``"sixty-four"`` into an int."""
    t = token.strip().lower().replace(",", "")
    if t.isdigit():
        return int(t)
    if t in ("one hundred", "hundred"):
        return 100
    parts = re.split(r"[- ]", t)
    total = 0
    for p in parts:
        if p in _UNITS:
            total += _UNITS[p]
        elif p in _TENS:
            total += _TENS[p]
        else:
            return None
    return total


def locate(text: str, pos: int) -> str:
    """Return ``pN:lM`` (pages from form feeds) or ``lM`` for offset ``pos``."""
    before = text[:pos]
    if "\x0c" in text:
        page = before.count("\x0c") + 1
        line = before.rsplit("\x0c", 1)[-1].count("\n") + 1
        return f"p{page}:l{line}"
    return f"l{before.count(chr(10)) + 1}"


def _quote(text: str, start: int, end: int, pad: int = 45) -> str:
    """Verbatim excerpt around ``[start, end)`` trimmed to word boundaries."""
    a = max(0, start - pad)
    b = min(len(text), end + pad)
    if a > 0:
        m = re.search(r"\s", text[a:start])
        if m:
            a += m.end()
    if b < len(text):
        m = re.search(r"\s\S*$", text[end:b])
        if m:
            b = end + m.start()
    return re.sub(r"\s+", " ", text[a:b]).strip()


def _scan(text: str, pattern: re.Pattern, value_fn):
    hits = []
    for m in pattern.finditer(text):
        try:
            value = value_fn(m)
        except (ValueError, TypeError):
            continue
        if value is None:
            continue
        hits.append((value, _quote(text, m.start(), m.end()), locate(text, m.start())))
    return hits


def _dedupe(hits):
    out, seen = [], set()
    for h in hits:
        key = (h[0], h[1])
        if key not in seen:
            seen.add(key)
            out.append(h)
    return out


# ---------------------------------------------------------------------------
# Participants
# ---------------------------------------------------------------------------
_SUBJ_NOUN = r"(?:healthy\s+|right-handed\s+|able-bodied\s+|naive\s+|naïve\s+|human\s+|adult\s+|male\s+|female\s+|volunteer\s+|BCI[- ]naive\s+)*(?:subjects?|participants?|volunteers?|individuals|persons|people|students|patients)"
_SUBJECTS_RE = re.compile(
    rf"(?<![\w.])(?:(?:n|N)\s*=\s*)?({_NUM})\s+(?:\(\s*\d+\s*\)\s+)?{_SUBJ_NOUN}\b",
    re.IGNORECASE,
)
_SUBJECTS_ALT_RE = re.compile(
    rf"\b(?:subjects?|participants?|volunteers?)\s*(?:\(|:)?\s*(?:n|N)\s*=\s*({_NUM})",
    re.IGNORECASE,
)


def find_subjects(text: str):
    def val(m):
        n = parse_number(m.group(1))
        if n is None or not 1 <= n <= 5000:
            return None
        # "one subject", "for each of 1 participant" are not cohort sizes.
        if n == 1:
            return None
        ctx = text[max(0, m.start() - 12) : m.start()].lower()
        if re.search(r"\b(each|per|every|single|any|another|remaining|other)\s*$", ctx):
            return None
        return n

    hits = _scan(text, _SUBJECTS_RE, val) + _scan(text, _SUBJECTS_ALT_RE, val)
    hits.sort(key=lambda h: h[2])
    return _dedupe(hits)


# ---------------------------------------------------------------------------
# Sampling rate
# ---------------------------------------------------------------------------
_SR_CONTEXT = r"(?:sampl\w*|digiti[sz]\w*|acqui\w*|recorded|record\w*|sample\s+rate|sampling\s+(?:rate|frequency)|Fs|fs)"
_SR_RE = re.compile(
    rf"{_SR_CONTEXT}[^.;]{{0,60}}?(?<![\d.])(\d+(?:[.,]\d+)?)\s*(k?)\s?Hz\b",
    re.IGNORECASE,
)
_SR_RE2 = re.compile(
    r"(?<![\d.])(\d+(?:[.,]\d+)?)\s*(k?)\s?Hz\s+(?:sampling|sample)\s+(?:rate|frequency)",
    re.IGNORECASE,
)


def _hz(m) -> float | None:
    raw = (
        m.group(1).replace(",", ".")
        if "," in m.group(1) and len(m.group(1).split(",")[-1]) != 3
        else m.group(1).replace(",", "")
    )
    v = float(raw)
    if m.group(2).lower() == "k":
        v *= 1000.0
    return v if 16 <= v <= 100000 else None


def find_sampling_rate(text: str):
    hits = _scan(text, _SR_RE, _hz) + _scan(text, _SR_RE2, _hz)
    # Drop obvious filter-band contexts.
    keep = []
    for v, q, loc in hits:
        low = q.lower()
        if re.search(
            r"(band-?pass|high-?pass|low-?pass|filter|notch|between\s+\d)", low
        ) and not re.search(r"sampl|digiti", low):
            continue
        keep.append((v, q, loc))
    keep.sort(key=lambda h: h[2])
    return _dedupe(keep)


# ---------------------------------------------------------------------------
# Channels / sessions / runs / trials
# ---------------------------------------------------------------------------
_CH_RE = re.compile(
    rf"(?<![\w.])({_NUM})[-\s]?(?:(?:EEG|scalp|active|passive|wet|dry|Ag/AgCl|gel-based|monopolar|bipolar|recording)[-\s])*(?:channels?|electrodes?)\b(?!\s+(?:were|was)\s+(?:removed|excluded|rejected|discarded))",
    re.IGNORECASE,
)


def find_channels(text: str):
    def val(m):
        n = parse_number(m.group(1))
        return n if n is not None and 1 <= n <= 1024 else None

    return _dedupe(_scan(text, _CH_RE, val))


_SESS_RE = re.compile(
    rf"(?<![\w.])({_NUM})\s+(?:(?:separate|different|recording|experimental|independent|daily|distinct|consecutive)\s+)*sessions?\b",
    re.IGNORECASE,
)


def find_sessions(text: str):
    def val(m):
        n = parse_number(m.group(1))
        return n if n is not None and 1 <= n <= 200 else None

    return _dedupe(_scan(text, _SESS_RE, val))


_RUNS_RE = re.compile(
    rf"(?<![\w.])({_NUM})\s+(?:(?:separate|different|recording|experimental|consecutive)\s+)*runs?\b",
    re.IGNORECASE,
)


def find_runs(text: str):
    def val(m):
        n = parse_number(m.group(1))
        return n if n is not None and 1 <= n <= 500 else None

    return _dedupe(_scan(text, _RUNS_RE, val))


_TRIALS_RE = re.compile(
    rf"(?<![\w.])({_NUM})\s+(?:(?:labelled|labeled|artifact-free|valid|correct|motor imagery|MI|imagery|training|test|testing|calibration|total)\s+)*trials\b",
    re.IGNORECASE,
)


def find_trials(text: str):
    def val(m):
        n = parse_number(m.group(1))
        return n if n is not None and 1 <= n <= 1_000_000 else None

    return _dedupe(_scan(text, _TRIALS_RE, val))


# ---------------------------------------------------------------------------
# Reference / ground / hardware
# ---------------------------------------------------------------------------
_REF_SITES = r"(?:left|right|linked|both|averaged?)?\s*(?:mastoids?|ear ?lobes?|ears?|nose|nasion|Cz|CPz|FCz|AFz|Fz|Pz|Oz|Fpz|A1|A2|TP9|TP10|M1|M2|CMS|DRL|common average|average|vertex|forehead)"
_REF_RE = re.compile(
    rf"(?:referenc\w+\s+(?:to|at|against|with|on|was|were|electrode(?:s)?(?:\s+(?:was|were))?)?(?:\s+the)?\s+((?:the\s+)?{_REF_SITES}(?:\s+(?:and|/)\s+(?:the\s+)?{_REF_SITES})?)"  # codespell:ignore referenc
    rf"|((?:the\s+)?{_REF_SITES})\s+(?:serv\w+|used|was used|acting|acted)\s+as\s+(?:the\s+)?reference"
    rf"|((?:the\s+)?{_REF_SITES})\s+(?:as|was the)\s+(?:the\s+)?reference(?:\s+electrode)?"
    rf"|reference(?:\s+electrode)?(?:\s+(?:was|were|placed|located|positioned|set))?\s+(?:at|on|to)\s+(?:the\s+)?({_REF_SITES}))",
    re.IGNORECASE,
)


def _first_group(m):
    for g in m.groups():
        if g:
            return re.sub(r"\s+", " ", g).strip().lower().removeprefix("the ")
    return None


def find_reference(text: str):
    return _dedupe(_scan(text, _REF_RE, _first_group))


_GND_RE = re.compile(
    rf"(?:ground(?:ed)?(?:\s+electrode)?(?:\s+(?:was|were|placed|located|positioned|set|at|on|to))*\s+(?:at|on|to|the)?\s*((?:the\s+)?{_REF_SITES})"
    rf"|((?:the\s+)?{_REF_SITES})\s+(?:serv\w+|used|was used|acting|acted)\s+as\s+(?:the\s+)?ground"
    rf"|((?:the\s+)?{_REF_SITES})\s+as\s+(?:the\s+)?ground)",
    re.IGNORECASE,
)


def find_ground(text: str):
    return _dedupe(_scan(text, _GND_RE, _first_group))


HARDWARE_TERMS = (
    "BrainAmp",
    "actiCHamp",
    "LiveAmp",
    "Brain Products",
    "BrainVision",
    "g.USBamp",
    "g.HIamp",
    "g.Nautilus",
    "g.tec",
    "g.GAMMAsys",
    "g.LADYbird",
    "BioSemi",
    "ActiveTwo",
    "Neuroscan",
    "SynAmps",
    "NuAmps",
    "Quik-Cap",
    "Compumedics",
    "ANT Neuro",
    "eego",
    "asalab",
    "EGI",
    "Geodesic",
    "Emotiv",
    "EPOC",
    "OpenBCI",
    "Cyton",
    "NeurOne",
    "Nihon Kohden",
    "Micromed",
    "TMSi",
    "Porti",
    "Enobio",
    "Neuroelectrics",
    "Unicorn",
    "Muse",
    "Biopac",
    "NeuroScan",
    "Grass",
    "NeuroSky",
    "Mindo",
    "Cognionics",
    "mBrainTrain",
    "Smarting",
    "Wearable Sensing",
    "DSI-24",
    "Neuracle",
    "NeuSen",
    "SynAmps2",
    "Cerebus",
    "Blackrock",
    "Ripple",
    "Elekta",
    "CTF",
    "Nexstim",
    "Bittium",
    "NeurOne",
    "NIRScout",
    "NIRx",
    "Artinis",
    "Hitachi",
    "Shimadzu",
    "Deymed",
    "Mitsar",
    "Encephalan",
    "NVX",
    "Medicom",
    "BE Plus",
    "EB Neuro",
    "Galileo",
    "Natus",
    "Nicolet",
    "XLTEK",
    "Twente Medical",
    "Refa",
    "SAGA",
    "Mobita",
    "Waveguard",
    "Easycap",
    "EasyCap",
    "actiCAP",
    "ActiCap",
)
_HW_RE = re.compile(
    r"\b("
    + "|".join(re.escape(t) for t in sorted(HARDWARE_TERMS, key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)


def find_hardware(text: str):
    return _dedupe(_scan(text, _HW_RE, lambda m: m.group(1)))


# ---------------------------------------------------------------------------
# Filters / line frequency / license
# ---------------------------------------------------------------------------
_FILT_RE = re.compile(
    r"(?:band-?pass(?:ed)?|filter\w*|pass-?band)[^.;]{0,60}?(?<![\d.])(\d+(?:\.\d+)?)\s*(?:Hz)?\s*(?:-|–|—|to|and)\s*(\d+(?:\.\d+)?)\s*Hz",
    re.IGNORECASE,
)
_FILT_RE2 = re.compile(
    r"(?<![\d.])(\d+(?:\.\d+)?)\s*(?:Hz)?\s*(?:-|–|—|to)\s*(\d+(?:\.\d+)?)\s*Hz\s+(?:band-?pass|filter)",
    re.IGNORECASE,
)


def find_filters(text: str):
    def val(m):
        lo, hi = float(m.group(1)), float(m.group(2))
        return (lo, hi) if lo < hi <= 20000 else None

    hits = _scan(text, _FILT_RE, val) + _scan(text, _FILT_RE2, val)
    hits.sort(key=lambda h: h[2])
    return _dedupe(hits)


_LINE_RE = re.compile(
    r"(?<![\d.])(50|60)\s?Hz\s+(?:notch|line|power(?:-|\s)?line|mains|band-?stop|power)|(?:notch|line noise|power(?:-|\s)?line|mains)[^.;]{0,40}?(?<![\d.])(50|60)\s?Hz",
    re.IGNORECASE,
)


def find_line_freq(text: str):
    return _dedupe(_scan(text, _LINE_RE, lambda m: float(m.group(1) or m.group(2))))


_LIC_RE = re.compile(
    r"\b(CC[- ]?BY(?:[- ]?(?:NC|ND|SA))*(?:[- ]?\d\.\d)?|CC0(?:[ -]?1\.0)?|Creative Commons(?: Attribution)?(?:[- ](?:Non[- ]?Commercial|No[- ]?Derivatives|Share[- ]?Alike))*(?: \d\.\d)?(?: International)?|ODC[- ](?:BY|ODbL|PDDL)|PDDL|Open Data Commons \w+|MIT License|GPL(?:v\d)?|Apache(?: License)?(?: 2\.0)?|BSD(?:-\d)?(?: license)?|PhysioNet Credentialed Health Data License[^\n]{0,10})\b",
    re.IGNORECASE,
)


def normalize_license(s: str | None) -> str | None:
    if not s:
        return None
    t = s.lower().replace("_", "-")
    t = t.replace("creative commons attribution", "cc-by").replace(
        "creative commons", "cc"
    )
    t = t.replace("non-commercial", "nc").replace("noncommercial", "nc")
    t = (
        t.replace("no-derivatives", "nd")
        .replace("noderivatives", "nd")
        .replace("no derivatives", "nd")
    )
    t = (
        t.replace("share-alike", "sa")
        .replace("sharealike", "sa")
        .replace("share alike", "sa")
    )
    t = t.replace(" international", "").replace(" license", "")
    t = re.sub(r"[\s_]+", "-", t.strip())
    t = re.sub(r"-+", "-", t)
    t = t.replace("cc-by-", "cc-by-").replace("ccby", "cc-by")
    m = re.match(r"^(cc-by(?:-(?:nc|nd|sa))*)-?(\d\.\d)?$", t)
    if m:
        return m.group(1) + (f"-{m.group(2)}" if m.group(2) else "")
    if t.startswith("cc0"):
        return "cc0"
    return t


def find_license(text: str):
    return _dedupe(_scan(text, _LIC_RE, lambda m: normalize_license(m.group(1))))


# ---------------------------------------------------------------------------
# Paradigm / class labels / free-text presence
# ---------------------------------------------------------------------------
PARADIGM_KEYWORDS = {
    "imagery": r"motor imagery|motor imagination|imagined movement|imagin\w+ (?:movement|motor|of|the)|kinesthetic imagery|imagery task|\bMI\b",  # codespell:ignore imagin
    "p300": r"P300|P3b\b|oddball|speller|event-related potential",
    "ssvep": r"SSVEP|steady[- ]state visual",
    "cvep": r"c-?VEP|code[- ]modulated",
    "rstate": r"resting[- ]state|eyes (?:open|closed)",
    "erp": r"event-related potential|\bERPs?\b|P300|N400|N170|\bMMN\b|N2pc",
    "movement": r"motor execution|executed movement|actual movement|movement execution|reach\w*|grasp\w*",
}

LABEL_SYNONYMS = {
    "left_hand": r"left[- ]hand|left hand|left fist",
    "right_hand": r"right[- ]hand|right hand|right fist",
    "hands": r"both hands|two hands|both fists",
    "feet": r"\bfeet\b|\bfoot\b|both feet",
    "tongue": r"\btongue\b",
    "rest": r"\brest\b|resting|idle",
    "navigation": r"navigation",
    "subtraction": r"subtraction|arithmetic",
    "word_ass": r"word association",
    "Target": r"\btarget",
    "NonTarget": r"non-?target|nontarget|standard",
    "target": r"\btarget",
    "nontarget": r"non-?target|nontarget|standard",
}


def find_keyword(text: str, pattern: str):
    rx = re.compile(pattern, re.IGNORECASE)
    return _dedupe(_scan(text, rx, lambda m: m.group(0)))


def find_phrase(text: str, phrase: str):
    """Case-insensitive search for a literal phrase (institution, hardware...)."""
    if not phrase or len(phrase.strip()) < 3:
        return []
    rx = re.compile(r"\s+".join(re.escape(w) for w in phrase.split()), re.IGNORECASE)
    return _dedupe(_scan(text, rx, lambda m: m.group(0)))


EXTRACTORS = {
    "n_subjects": find_subjects,
    "sampling_rate": find_sampling_rate,
    "channel_types.eeg": find_channels,
    "sessions_per_subject": find_sessions,
    "runs_per_session": find_runs,
    "n_trials": find_trials,
    "reference": find_reference,
    "ground": find_ground,
    "hardware": find_hardware,
    "filters": find_filters,
    "line_freq": find_line_freq,
    "license": find_license,
}
