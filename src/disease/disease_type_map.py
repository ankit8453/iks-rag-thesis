"""Map every source dataset's class name to a canonical DISEASE TYPE.

Why
---
Our datasets (PlantVillage, PlantDoc, Paddy Doctor, and Dr. Pandey's Brazilian
multi-crop set) between them have ~200 *crop x disease* classes, most with far
too few images to train. But the underlying *diseases* — rust, powdery mildew,
leaf spot, blight, mosaic — recur across crops. Regrouping by disease TYPE pools
those images into a handful of well-populated, crop-agnostic classes, which is
both trainable and aligned with the symptom-based direction of the system.

This module is the mapping. It works by keyword, in priority order, over a
normalised class string, so it handles English and Portuguese labels uniformly
and is easy to audit and extend. The first rule that matches wins — so more
specific rules (``late blight``, ``bacterial spot``) must precede generic ones
(``blight``, ``spot``).

``OTHER`` collects abiotic / nutritional / pest-damage cases that are not IKS
"diseases"; callers typically drop it from the training set.
"""

from __future__ import annotations

import re

#: Canonical disease-type labels (the model's new output space).
CANONICAL_TYPES: tuple[str, ...] = (
    "healthy",
    "rust",
    "scab",
    "powdery_mildew",
    "downy_mildew",
    "leaf_spot",
    "bacterial",
    "early_blight",
    "late_blight",
    "blight",
    "mosaic_virus",
    "leaf_mold",
    "anthracnose",
    "rot",
    "pest_damage",
    "other",
)

#: Types usually excluded from the disease classifier's training set: "other"
#: is abiotic/misc; "pest_damage" is insect/mite damage, not a pathogen — keep or
#: drop per experiment. Exposed so the prep script can filter consistently.
NON_DISEASE_TYPES: frozenset[str] = frozenset({"other", "pest_damage"})

# Ordered (regex, type). FIRST match wins — order matters. English + Portuguese.
_RULES: tuple[tuple[str, str], ...] = (
    # healthy / normal
    (r"health|normal|saudavel|sadia", "healthy"),
    # blights — specific before generic
    (r"late.?blight", "late_blight"),
    (r"early.?blight", "early_blight"),
    # mildews (before generic 'mold'/'mildew')
    (r"powdery.?mildew|oidio", "powdery_mildew"),
    (r"downy.?mildew|mildio", "downy_mildew"),
    # leaf mold / sooty mould
    (r"leaf.?mold|leaf.?mould|\bmold\b|\bmould\b|mofo|fumagina|sooty", "leaf_mold"),
    # bacterial (before 'spot'/'blight' generics; catches 'bacterial spot/blight/streak')
    (r"bacteri|crestamento|cancro|canker|streak|panicle|halo.?blight", "bacterial"),
    # rust
    (r"\brust\b|ferrugem", "rust"),
    # scab
    (r"scab|\blixa\b|verrugose", "scab"),
    # anthracnose
    (r"anthracnose|antracnose", "anthracnose"),
    # mosaic / viral
    (r"mosaic|mosaico|yellow.?leaf.?curl|tungro|woodiness|endurecimento|mottle|\bvirus\b|leprose",
     "mosaic_virus"),
    # rots
    (r"black.?rot|\besca\b|measles|podridao|red.?stripe|charcoal|\brot\b", "rot"),
    # leaf spots (septoria, cercospora, target, gray, brown, angular, ring, scald, algal, generic spot/mancha)
    (r"septoria|cercospor|gray.?leaf.?spot|grey.?leaf.?spot|target|brown.?spot|"
     r"angular|ring.?spot|areolate|leaf.?scorch|\bscorch\b|scald|escaldadura|"
     r"\balga|algae|leaf.?spot|mancha|\bspot\b|clorose|chlorosis",
     "leaf_spot"),
    # generic / other blights, incl. rice blast and web blight (mela/soreshin)
    (r"northern.?leaf.?blight|leaf.?blight|\bblight\b|\bblast\b|brusone|queima|helmint|"
     r"bipolaris|turcicum|diplodia|physoderma|mela|soreshin", "blight"),
    # pest / insect / mite damage (not a pathogen)
    (r"spider.?mite|\bmite\b|acaro|hispa|dead.?heart|lagarta|larva|worm|caterpillar|"
     r"miner|mineiro|whitefly|mosca|\bgall|cochonilha|scale|ataque|dione|hedylepta|bicho",
     "pest_damage"),
    # LAST: a plain "<crop> leaf" with no disease word is a healthy leaf
    # (PlantDoc names its healthy classes this way, e.g. "Apple leaf"). Disease
    # classes reach a disease rule above first, so only healthy leaves fall here.
    (r"^[a-z0-9 ]+ leaf$", "healthy"),
)


def _normalise(class_name: str) -> str:
    s = class_name.lower().replace("_", " ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)          # drop accents/punct after lowering
    return re.sub(r"\s+", " ", s).strip()


def to_disease_type(class_name: str) -> str:
    """Map one source class name to its canonical disease type.

    Returns ``"other"`` for abiotic / nutritional / unrecognised labels so the
    caller can decide whether to keep or drop them — nothing is silently lost.
    """
    s = _normalise(class_name)
    for pattern, dtype in _RULES:
        if re.search(pattern, s):
            return dtype
    return "other"


def is_disease(dtype: str) -> bool:
    """True if this type is a pathogen class we'd train the disease model on."""
    return dtype not in NON_DISEASE_TYPES


__all__ = [
    "CANONICAL_TYPES",
    "NON_DISEASE_TYPES",
    "is_disease",
    "to_disease_type",
]
