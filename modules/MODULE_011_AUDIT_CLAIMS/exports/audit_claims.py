"""Scientific claim auditing with 7-section canonical reports.

Audits a text claim against a corpus of artifacts using keyword overlap,
channel-aware evidence extraction, and fail-closed verdict policy.

Verdict scale: SUPPORTED / PARTIALLY SUPPORTED / WEAKLY SUPPORTED / UNSUPPORTED / UNCERTAIN
Confidence: 0–10 heuristic score (not a calibrated probability).

All outputs: LOCAL_LAB_ONLY — not external proof.
"""
from __future__ import annotations

import csv
import json
import math
import re
import statistics
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VERDICTS = ["SUPPORTED", "PARTIALLY SUPPORTED", "WEAKLY SUPPORTED", "UNSUPPORTED", "UNCERTAIN"]

WEAKNESS_TYPES = [
    "Logical Contradiction",
    "Observational Tension",
    "Assumption Dependence",
    "Explanatory Limit",
]
STATUSES = ["Established", "Contested", "Speculative"]
SEVERITIES = ["High", "Medium", "Low"]
CAUSES = ["Data", "Measurement", "Modeling", "Parameters", "New Physics"]

STRICT_SECTIONS = [
    "1. Title",
    "2. Executive Summary",
    "3. Detailed Analysis",
    "4. Weakness Taxonomy",
    "5. Limitations",
    "6. Recommendations",
    "7. Sources",
]

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "is", "are", "was", "were", "be",
    "been", "being", "de", "la", "le", "les", "des", "du", "et", "ou", "mais", "si",
    "est", "sont", "être", "dans", "to", "of", "for", "with", "on", "in", "by",
    "that", "this", "these", "those", "as", "at", "from", "un", "une", "au", "aux",
    "ce", "cet", "cette", "ces", "sur", "par", "pour", "sans", "plus", "moins",
}

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SourceRecord:
    label: str
    path: Optional[str]
    source_type: str  # artifact / conversation / derived
    stable: bool
    excerpt: Optional[str] = None

    def render(self) -> str:
        tail = f" — excerpt: {self.excerpt}" if self.excerpt else ""
        if self.path:
            return f"- {self.label} ({self.source_type}, {'stable' if self.stable else 'unstabilized'}) — {self.path}{tail}"
        return f"- {self.label} ({self.source_type}, {'stable' if self.stable else 'unstabilized'}){tail}"


@dataclass
class EvidenceSlice:
    source_label: str
    channel: str  # observation / theory / modeling / interpretation / inference
    text: str
    score: float = 0.0


@dataclass
class Weakness:
    type: str
    status: str
    severity: str
    probable_cause: str
    discriminant_test: str
    basis: Optional[str] = None

    def validate(self) -> None:
        if self.type not in WEAKNESS_TYPES:
            raise ValueError(f"Invalid weakness type: {self.type}")
        if self.status not in STATUSES:
            raise ValueError(f"Invalid status: {self.status}")
        if self.severity not in SEVERITIES:
            raise ValueError(f"Invalid severity: {self.severity}")
        if self.probable_cause not in CAUSES:
            raise ValueError(f"Invalid cause: {self.probable_cause}")
        if not self.discriminant_test.strip():
            raise ValueError("Discriminant Test is empty")

    def render(self, idx: int) -> str:
        self.validate()
        parts = [
            f"Weakness {idx}", "",
            f"Type: {self.type}", "",
            f"Status: {self.status}", "",
            f"Severity: {self.severity}", "",
            f"Probable Cause: {self.probable_cause}", "",
            f"Discriminant Test: {self.discriminant_test.strip()}",
        ]
        if self.basis:
            parts += ["", self.basis.strip()]
        return "\n".join(parts)


@dataclass
class PropositionAssessment:
    proposition: str
    support_score: float
    status: str  # supported / mixed / unsupported
    supporting: List[str] = field(default_factory=list)
    opposing: List[str] = field(default_factory=list)


@dataclass
class AuditReport:
    title: str
    reformulated_question: str
    verdict: str
    confidence: float
    one_sentence_justification: str
    propositions: List[str]
    supporting_arguments: List[str]
    opposing_arguments: List[str]
    weaknesses: List[Weakness]
    internal_logical_contradictions: str
    tensions_with_observations: str
    assumption_dependence: str
    explanatory_limits: str
    limitations: List[str]
    recommendations_tests: List[str]
    recommendations_alternatives: List[str]
    recommendations_next_questions: List[str]
    sources: List[SourceRecord]

    def validate(self) -> None:
        if self.verdict not in VERDICTS:
            raise ValueError(f"Invalid verdict: {self.verdict}")
        if not (0.0 <= self.confidence <= 10.0):
            raise ValueError("Confidence out of [0,10]")
        for w in self.weaknesses:
            w.validate()

    def render_markdown(self) -> str:
        self.validate()
        prop_text = "\n".join(f"{i+1}. {p}" for i, p in enumerate(self.propositions)) or "No testable propositions extracted."
        supp_text = "\n".join(f"- {x}" for x in self.supporting_arguments) or "- No sufficiently supported favorable arguments."
        opp_text = "\n".join(f"- {x}" for x in self.opposing_arguments) or "- No strong objections isolated."
        weak_text = "\n\n".join(w.render(i+1) for i, w in enumerate(self.weaknesses)) or "No individualized weaknesses."
        limitations = "\n".join(f"- {x}" for x in self.limitations) or "- Limits not specified."
        rec_tests = "\n".join(f"- {x}" for x in self.recommendations_tests) or "- No tests proposed."
        rec_alt = "\n".join(f"- {x}" for x in self.recommendations_alternatives) or "- No alternatives proposed."
        rec_next = "\n".join(f"- {x}" for x in self.recommendations_next_questions) or "- No next questions proposed."
        src_text = "\n".join(s.render() for s in self.sources) or "- conversational source, not stabilized as external artifact"

        return f"""1. Title

{self.title}

2. Executive Summary

- Reformulated question/hypothesis
{self.reformulated_question}

- Verdict: {self.verdict}

- Confidence score (0–10)
{self.confidence:.1f}/10

- One-sentence justification based on evidence quality, theoretical coherence, and limits
{self.one_sentence_justification}

3. Detailed Analysis

- Decomposition into propositions
{prop_text}

- Supporting arguments
{supp_text}

- Opposing arguments
{opp_text}

- Weakness analysis using micro-syntax

{weak_text}

4. Weakness Taxonomy

- Internal logical contradictions
{self.internal_logical_contradictions}

- Tensions with observations
{self.tensions_with_observations}

- Strong dependence on assumptions or parameterization
{self.assumption_dependence}

- Explanatory limits or unknown ontology
{self.explanatory_limits}

5. Limitations

{limitations}

6. Recommendations

- tests
{rec_tests}

- alternatives
{rec_alt}

- next questions
{rec_next}

7. Sources

{src_text}
"""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["claim_ceiling"] = "LOCAL_LAB_ONLY_NOT_PUBLIC_PROOF"
        return d


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_json(path: Path) -> str:
    obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _read_csv(path: Path, max_rows: int = 100) -> str:
    rows: List[List[str]] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        for i, row in enumerate(csv.reader(f)):
            rows.append(row)
            if i + 1 >= max_rows:
                break
    return "\n".join(",".join(cell[:200] for cell in row) for row in rows)


def _compact_excerpt(text: str, limit: int = 180) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit] + ("…" if len(text) > limit else "")


def load_artifacts(
    inputs: Sequence[str],
    max_chars_per_file: int = 25_000,
) -> Tuple[List[Tuple[Path, str]], List[SourceRecord]]:
    """Load text content from files/directories. Returns (loaded_pairs, sources)."""
    loaded: List[Tuple[Path, str]] = []
    sources: List[SourceRecord] = []

    def _iter_paths() -> Iterable[Path]:
        for item in inputs:
            p = Path(item).expanduser().resolve()
            if p.is_file():
                yield p
            elif p.is_dir():
                for f in p.rglob("*"):
                    if f.is_file():
                        yield f

    for path in _iter_paths():
        suffix = path.suffix.lower()
        try:
            if suffix in {".txt", ".md", ".py"}:
                text = _read_text(path)
            elif suffix == ".json":
                text = _read_json(path)
            elif suffix == ".csv":
                text = _read_csv(path)
            else:
                continue
            text = text[:max_chars_per_file]
            loaded.append((path, text))
            sources.append(SourceRecord(
                label=path.name, path=str(path),
                source_type="artifact", stable=True,
                excerpt=_compact_excerpt(text),
            ))
        except Exception as e:
            sources.append(SourceRecord(
                label=path.name, path=str(path),
                source_type="artifact", stable=True,
                excerpt=f"read error: {e}",
            ))
    return loaded, sources


# ---------------------------------------------------------------------------
# Text analysis utilities
# ---------------------------------------------------------------------------

def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> List[str]:
    toks = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9_]+", text.lower())
    return [t for t in toks if t not in _STOPWORDS and len(t) > 2]


def _keyword_overlap(a: str, b: str) -> float:
    sa, sb = set(_tokenize(a)), set(_tokenize(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def _split_sentences(text: str) -> List[str]:
    return [_normalize_space(c) for c in re.split(r"(?<=[\.\?\!])\s+|\n{2,}", text) if _normalize_space(c)]


def _infer_channel(sentence: str) -> str:
    low = sentence.lower()
    if any(k in low for k in ["observe", "observed", "mesure", "measured", "present", "contains", "states", "indique"]):
        return "observation"
    if any(k in low for k in ["model", "framework", "procedure", "architecture"]):
        return "modeling"
    if any(k in low for k in ["infer", "therefore", "thus", "donc", "supports", "soutient"]):
        return "inference"
    if any(k in low for k in ["theory", "theorem", "axiom", "principe"]):
        return "theory"
    if any(k in low for k in ["interpret", "suggests", "suggère"]):
        return "interpretation"
    return "observation"


# ---------------------------------------------------------------------------
# Core audit functions
# ---------------------------------------------------------------------------

def extract_evidence(
    claim: str,
    artifacts: List[Tuple[Path, str]],
    top_k: int = 20,
) -> List[EvidenceSlice]:
    """Extract and rank evidence sentences from artifacts against a claim."""
    slices: List[EvidenceSlice] = []
    for path, text in artifacts:
        for sent in _split_sentences(text):
            score = _keyword_overlap(sent, claim)
            if any(t in sent.lower() for t in ["must", "required", "explicit", "strict", "define"]):
                score = min(1.0, score + 0.05)
            if score > 0:
                slices.append(EvidenceSlice(
                    source_label=path.name,
                    channel=_infer_channel(sent),
                    text=sent,
                    score=score,
                ))
    slices.sort(key=lambda x: x.score, reverse=True)
    return slices[:top_k]


def decompose_claim(claim: str) -> List[str]:
    """Decompose a compound claim into testable propositions."""
    claim_clean = _normalize_space(claim)
    parts = re.split(r"\b(?:and|et|,|;)\b", claim_clean)
    parts = [_normalize_space(p) for p in parts if _normalize_space(p)]
    if len(parts) <= 1:
        qualifiers = [
            "coherent", "non-ambiguous", "operational", "canonical",
            "reliable", "strictly",
        ]
        found = [q for q in qualifiers if q in claim_clean.lower()]
        if found:
            return [f"The audited object satisfies the property: {q}." for q in found]
        return [claim_clean]
    subject = parts[0]
    props = []
    if len(parts) > 1 and len(subject.split()) >= 2:
        props.append(subject)
        for p in parts[1:]:
            if len(p.split()) < 3:
                props.append(f"{subject} satisfies the property: {p}.")
            else:
                props.append(p)
    else:
        props = parts
    out: List[str] = []
    seen: set = set()
    for p in props:
        p = p.strip(" .")
        if not p:
            continue
        if not p.endswith("."):
            p += "."
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out[:12]


def assess_proposition(
    prop: str,
    evidence: List[EvidenceSlice],
) -> PropositionAssessment:
    """Assess support level for one proposition against evidence slices."""
    support = [e for e in evidence if _keyword_overlap(e.text, prop) >= 0.08]
    positive: List[str] = []
    negative: List[str] = []

    _pos = ["must", "required", "explicit", "defined", "clear", "strict", "canonical", "define", "template", "verdict"]
    _neg = ["not", "absence", "missing", "pending", "future work", "heuristic", "not yet", "unproven", "limit"]

    total = 0.0
    for e in support:
        low = e.text.lower()
        sign = 0.0
        if any(m in low for m in _pos):
            sign += 0.18
            positive.append(f"{e.source_label}: {_compact_excerpt(e.text, 220)}")
        if any(m in low for m in _neg):
            sign -= 0.18
            negative.append(f"{e.source_label}: {_compact_excerpt(e.text, 220)}")
        sign += (e.score - 0.1)
        total += sign

    if not support:
        return PropositionAssessment(prop, 0.0, "unsupported", [], [])

    score = max(0.0, min(1.0, 0.5 + total / max(3.0, len(support) * 1.2)))
    if score >= 0.67:
        status = "supported"
    elif score >= 0.35:
        status = "mixed"
    else:
        status = "unsupported"

    if not positive and support:
        positive = [f"{support[0].source_label}: {_compact_excerpt(support[0].text, 220)}"]
    if not negative:
        implied = [e for e in support if any(x in e.text.lower() for x in ["pending", "future", "limit", "uncertain", "insufficient"])]
        negative = [f"{e.source_label}: {_compact_excerpt(e.text, 220)}" for e in implied[:2]]

    return PropositionAssessment(prop, score, status, positive[:4], negative[:4])


def detect_logical_contradictions(texts: List[str]) -> List[str]:
    """Scan combined text for lexical contradiction pairs."""
    pairs = [
        ("always", "never"),
        ("strictly canonical", "not canonical"),
        ("deployment-ready", "not yet validated"),
    ]
    combined = " \n".join(t.lower() for t in texts)
    return [
        f"Lexical tension detected between '{a}' and '{b}'."
        for a, b in pairs
        if a in combined and b in combined
    ]


def _build_weaknesses(
    assessments: List[PropositionAssessment],
    claim: str,
    evidence: List[EvidenceSlice],
    contradictions: List[str],
) -> List[Weakness]:
    weaknesses: List[Weakness] = []
    unsupported = [a for a in assessments if a.status == "unsupported"]
    mixed = [a for a in assessments if a.status == "mixed"]

    if contradictions:
        weaknesses.append(Weakness(
            type="Observational Tension", status="Established", severity="Medium",
            probable_cause="Modeling",
            discriminant_test="Systematically compare passages claiming strong readiness against passages listing benchmark/calibration/validation as future work; if both coexist without a clear condition, maintain the observational tension.",
            basis=contradictions[0],
        ))

    if unsupported:
        names = "; ".join(a.proposition for a in unsupported[:3])
        weaknesses.append(Weakness(
            type="Explanatory Limit", status="Established", severity="Medium",
            probable_cause="Modeling",
            discriminant_test="Isolate each unsupported sub-claim and request a specific artifact or measure for each; if no direct evidence is provided, the sub-claim remains unestablished.",
            basis=f"Unsupported sub-claims: {names}",
        ))

    if mixed or unsupported:
        weaknesses.append(Weakness(
            type="Assumption Dependence", status="Established",
            severity="Medium" if unsupported else "Low",
            probable_cause="Modeling",
            discriminant_test="Apply the same protocol to multiple independent evaluators on the same corpus and measure variance in verdicts, confidence scores, and weakness types.",
            basis="Validity of some conclusions depends partly on interpretation of the claim scope and granularity chosen for decomposition.",
        ))

    if not weaknesses:
        weaknesses.append(Weakness(
            type="Explanatory Limit", status="Contested", severity="Low",
            probable_cause="Data",
            discriminant_test="Extend the corpus or provide a contradictory benchmark to verify the verdict remains stable outside currently available artifacts.",
            basis="No major weaknesses extracted automatically, but absence of external benchmark limits scope.",
        ))

    return weaknesses[:4]


def _verdict_from_assessments(
    assessments: List[PropositionAssessment],
    sources: List[SourceRecord],
) -> Tuple[str, float]:
    if not assessments:
        return "UNCERTAIN", 3.0
    scores = [a.support_score for a in assessments]
    mean_score = statistics.mean(scores)
    unsupported = sum(1 for a in assessments if a.status == "unsupported")
    mixed = sum(1 for a in assessments if a.status == "mixed")
    source_bonus = min(0.6, sum(1 for s in sources if s.source_type == "artifact" and s.stable) * 0.08)
    confidence = max(0.0, min(10.0, mean_score * 8.8 + source_bonus))

    if unsupported == len(assessments):
        return "UNSUPPORTED", max(1.5, confidence - 2.2)
    if unsupported > 0 or mixed > 0:
        if mean_score >= 0.60:
            return "PARTIALLY SUPPORTED", confidence
        if mean_score >= 0.35:
            return "WEAKLY SUPPORTED", confidence
        return "UNCERTAIN", max(2.5, confidence - 1.5)
    if mean_score >= 0.80:
        return "SUPPORTED", confidence
    return "PARTIALLY SUPPORTED", confidence


def _dedupe(items: List[str]) -> List[str]:
    out: List[str] = []
    seen: set = set()
    for x in items:
        if x not in seen and x.strip():
            seen.add(x)
            out.append(x)
    return out


def _one_sentence_justification(verdict: str, assessments: List[PropositionAssessment]) -> str:
    supported = sum(1 for a in assessments if a.status == "supported")
    mixed = sum(1 for a in assessments if a.status == "mixed")
    unsupported = sum(1 for a in assessments if a.status == "unsupported")
    if verdict == "SUPPORTED":
        return f"The corpus strongly supports the central propositions ({supported} supported), with no major structural weakness other than explicitly bounded scope limits."
    if verdict == "PARTIALLY SUPPORTED":
        return f"The corpus supports several central propositions ({supported} supported, {mixed} mixed), but at least one sub-claim remains insufficiently established."
    if verdict == "WEAKLY SUPPORTED":
        return f"Textual support remains fragmented ({unsupported} sub-claims insufficiently supported) and the conclusion remains sensitive to scope and artifact quality."
    if verdict == "UNSUPPORTED":
        return "Available artifacts do not directly support the central claim and cannot close the evidence gap."
    return "Available evidence is too incomplete or ambiguous to support a robust conclusion without overinterpretation."


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def audit_claim(
    claim: str,
    title: str,
    inputs: Sequence[str] = (),
    *,
    add_conversation_source: bool = False,
) -> AuditReport:
    """Audit a claim against a corpus of artifacts. Returns AuditReport.

    Args:
        claim: The claim text to audit.
        title: Title for the report.
        inputs: List of file or directory paths to load as corpus.
        add_conversation_source: If True, add the claim itself as an unstable source.
    """
    artifacts, sources = load_artifacts(inputs)
    if add_conversation_source:
        sources.append(SourceRecord(
            label="Claim provided as input",
            path=None, source_type="conversation", stable=False,
            excerpt=_compact_excerpt(claim),
        ))

    evidence = extract_evidence(claim, artifacts)
    propositions = decompose_claim(claim)
    assessments = [assess_proposition(p, evidence) for p in propositions]
    contradictions = detect_logical_contradictions([text for _, text in artifacts])
    weaknesses = _build_weaknesses(assessments, claim, evidence, contradictions)
    verdict, confidence = _verdict_from_assessments(assessments, sources)

    internal = ("A logical contradiction has been retained and must be checked line by line."
                if any(w.type == "Logical Contradiction" for w in weaknesses)
                else ("No demonstrable logical contradiction; opposing formulations exist but constitute observational tension rather than formal internal incompatibility."
                      if contradictions else "No clear internal logical contradiction established."))
    tension = ("At least one observational tension is established between description, claimed readiness, and available artifact support."
               if any(w.type == "Observational Tension" for w in weaknesses)
               else "No major observational tension extracted.")
    assumption = ("Validity of some conclusions depends on the scope of the claim, sub-claim decomposition, and evaluator interpretation."
                  if any(w.type == "Assumption Dependence" for w in weaknesses)
                  else "Assumption dependence does not appear dominant.")
    explanatory = ("The framework or corpus does not always suffice to establish all sub-claims; some conclusions remain better specified than demonstrated."
                   if any(w.type == "Explanatory Limit" for w in weaknesses)
                   else "No strong explanatory limit retained.")

    limitations = _dedupe([
        "This analysis depends exclusively on the provided input artifacts.",
        "Conclusions apply to the material actually read, not to undocumented intentions.",
    ] + (["No exploitable artifacts loaded; verdict must remain weak or uncertain."] if not artifacts else [])
      + (["Volume of relevant textual evidence is limited; verdict scope should be restricted."] if len(evidence) < 5 else [])
      + ["The confidence score is a heuristic audit score, not a calibrated probability."])

    rec_tests = _dedupe([w.discriminant_test for w in weaknesses])[:5]
    rec_alt = [
        "Explicitly separate specification, procedure, and empirical validation verdicts.",
        "Compare the protocol against a simpler structured review checklist to measure real gain.",
    ]
    rec_next = [
        "Which sub-claims require external benchmark rather than textual inspection alone?",
        "Does the verdict remain stable if the claim is reformulated without strong qualifiers?",
    ]

    return AuditReport(
        title=title,
        reformulated_question=claim.strip(),
        verdict=verdict,
        confidence=confidence,
        one_sentence_justification=_one_sentence_justification(verdict, assessments),
        propositions=[a.proposition for a in assessments],
        supporting_arguments=_dedupe([
            f"Proposition: {a.proposition} — support: {a.status}."
            for a in assessments if a.status in {"supported", "mixed"}
        ] + [x for a in assessments if a.status in {"supported", "mixed"} for x in a.supporting[:2]])[:8],
        opposing_arguments=_dedupe([
            f"Proposition: {a.proposition} — insufficient support: {a.status}."
            for a in assessments if a.status in {"mixed", "unsupported"}
        ] + [x for a in assessments if a.status in {"mixed", "unsupported"} for x in a.opposing[:2]])[:8],
        weaknesses=weaknesses,
        internal_logical_contradictions=internal,
        tensions_with_observations=tension,
        assumption_dependence=assumption,
        explanatory_limits=explanatory,
        limitations=limitations,
        recommendations_tests=rec_tests,
        recommendations_alternatives=rec_alt,
        recommendations_next_questions=rec_next,
        sources=sources or [SourceRecord(
            label="conversational source, not stabilized as external artifact",
            path=None, source_type="conversation", stable=False,
        )],
    )


def inspect_document(title: str, inputs: Sequence[str] = ()) -> AuditReport:
    """Inspect a document corpus without an explicit claim. Returns AuditReport."""
    claim = "The provided document is coherent, traceable, canonically structured, and without major category confusion."
    return audit_claim(claim, title, inputs)


def validate_canonical_report(report_text: str) -> Tuple[bool, List[str]]:
    """Check that a rendered markdown report contains all 7 required sections."""
    errors = [f"Missing section: {s}" for s in STRICT_SECTIONS if s not in report_text]
    for block in re.findall(r"Weakness\s+\d+.*?(?=(?:\nWeakness\s+\d+)|\n4\.|\Z)", report_text, flags=re.DOTALL):
        idx = int(re.search(r"Weakness\s+(\d+)", block).group(1))
        for field_name in ["Type:", "Status:", "Severity:", "Probable Cause:", "Discriminant Test:"]:
            if field_name not in block:
                errors.append(f"Weakness {idx}: missing field {field_name}")
    return len(errors) == 0, errors


def batch_audit(
    claims: List[Dict[str, Any]],
    inputs: Sequence[str] = (),
) -> Dict[str, Any]:
    """Run audit_claim on a list of {title, claim} dicts. Returns summary dict."""
    rows = []
    verdicts_list: List[str] = []
    confidences_list: List[float] = []
    for item in claims:
        report = audit_claim(
            claim=item["claim"],
            title=item.get("title", f"Audit: {item['claim'][:50]}"),
            inputs=inputs,
        )
        rows.append({
            "title": report.title,
            "claim": report.reformulated_question,
            "verdict": report.verdict,
            "confidence": report.confidence,
            "weakness_count": len(report.weaknesses),
        })
        verdicts_list.append(report.verdict)
        confidences_list.append(report.confidence)

    return {
        "n_claims": len(rows),
        "rows": rows,
        "verdict_distribution": {v: verdicts_list.count(v) for v in VERDICTS if verdicts_list.count(v)},
        "confidence_mean": round(statistics.mean(confidences_list), 3) if confidences_list else None,
        "confidence_stdev": round(statistics.pstdev(confidences_list), 3) if len(confidences_list) > 1 else 0.0,
        "claim_ceiling": "LOCAL_LAB_ONLY_NOT_PUBLIC_PROOF",
        "note": "Internal protocol benchmark. No external benchmark claimed.",
    }
