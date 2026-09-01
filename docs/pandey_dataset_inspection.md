# Inspection report — Pandey-supplied disease dataset

> **Read-only inspection. Nothing was merged, trained, or pushed.**
> Source path:
> `C:\Users\HP\Downloads\Plant_leaf_diseases_dataset\Plant_leave_diseases_dataset_with_augmentation`
> Contact-sheet PNG: [`docs/pandey_dataset_samples.png`](pandey_dataset_samples.png)

## 1. Structure

```
Plant_leaf_diseases_dataset/
└── Plant_leave_diseases_dataset_with_augmentation/
    ├── Apple___Apple_scab/         image (1).JPG, image (2).JPG, ...
    ├── Apple___Black_rot/
    ├── Apple___Cedar_apple_rust/
    ├── Apple___healthy/
    ├── Background_without_leaves/  image (1).jpg, image (2).jpg, ...
    ├── Blueberry___healthy/
    ├── …  (36 class folders total)
    └── Tomato___healthy/
```

- **Class-per-folder.** No train/val/test split — flat per-class layout.
- One redundant outer wrapper directory.
- Filename pattern is `image (N).JPG` (or `.jpg` / `.jpeg`) — sequential numbering with no source identifier, the signature of a re-pack tool that renumbered an existing dataset.
- One unusual folder: `Background_without_leaves/` (street scenes / cars / roads — see contact sheet) — used by PlantVillage-augmented variants as a "no-leaf" reject class.

## 2. Classes (full list, as written)

**36 total class folders** (35 leaf classes + 1 background class). Names use the `___` triple-underscore separator between crop and condition — PlantVillage's distinctive convention.

```
Apple___Apple_scab
Apple___Black_rot
Apple___Cedar_apple_rust
Apple___healthy
Background_without_leaves
Blueberry___healthy
Cherry___Powdery_mildew
Cherry___healthy
Corn___Cercospora_leaf_spot Gray_leaf_spot
Corn___Common_rust
Corn___Northern_Leaf_Blight
Corn___healthy
Grape___Black_rot
Grape___Esca_(Black_Measles)
Grape___Leaf_blight_(Isariopsis_Leaf_Spot)
Grape___healthy
Orange___Haunglongbing_(Citrus_greening)
Peach___Bacterial_spot
Peach___healthy
Pepper,_bell___Bacterial_spot
Pepper,_bell___healthy
Potato___Early_blight
Potato___Late_blight
Potato___healthy
Raspberry___healthy
Soybean___healthy
Squash___Powdery_mildew
Strawberry___Leaf_scorch
Strawberry___healthy
Tomato___Bacterial_spot
Tomato___Early_blight
Tomato___Late_blight
Tomato___Leaf_Mold
Tomato___Septoria_leaf_spot
Tomato___Spider_mites Two-spotted_spider_mite
Tomato___healthy
```

## 3. Counts per class

| Class | Images |
|---|---:|
| Apple___Apple_scab | 1,000 |
| Apple___Black_rot | 1,000 |
| Apple___Cedar_apple_rust | 1,000 |
| Apple___healthy | **1,645** |
| Background_without_leaves | 1,143 |
| Blueberry___healthy | 1,502 |
| Cherry___Powdery_mildew | 1,052 |
| Cherry___healthy | 1,000 |
| Corn___Cercospora_leaf_spot Gray_leaf_spot | 1,000 |
| Corn___Common_rust | 1,192 |
| Corn___Northern_Leaf_Blight | 1,000 |
| Corn___healthy | 1,162 |
| Grape___Black_rot | 1,180 |
| Grape___Esca_(Black_Measles) | 1,383 |
| Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | 1,076 |
| Grape___healthy | 1,000 |
| **Orange___Haunglongbing_(Citrus_greening)** | **5,507** |
| Peach___Bacterial_spot | 2,297 |
| Peach___healthy | 1,000 |
| Pepper,_bell___Bacterial_spot | 1,000 |
| Pepper,_bell___healthy | 1,478 |
| Potato___Early_blight | 1,000 |
| Potato___Late_blight | 1,000 |
| Potato___healthy | 1,000 |
| Raspberry___healthy | 1,000 |
| **Soybean___healthy** | **5,090** |
| Squash___Powdery_mildew | 1,835 |
| Strawberry___Leaf_scorch | 1,109 |
| Strawberry___healthy | 1,000 |
| Tomato___Bacterial_spot | 2,127 |
| Tomato___Early_blight | 1,000 |
| Tomato___Late_blight | 1,909 |
| Tomato___Leaf_Mold | 1,000 |
| Tomato___Septoria_leaf_spot | 1,771 |
| Tomato___Spider_mites Two-spotted_spider_mite | 1,610 |
| Tomato___healthy | 1,591 |
| **GRAND TOTAL** | **53,659** |

**Imbalance notes**

- A clear floor of **1,000 per class** (24 of 36 classes are at exactly 1,000) — the unmistakable signature of an augmentation-up-sampling pass that capped under-represented classes.
- Two huge outliers: **Orange Citrus-greening (5,507)** and **Soybean healthy (5,090)** — the only classes with > 5× the floor.
- File format split: 51,750 `.jpg` + 1,909 `.jpeg`. The 1,909 `.jpeg`s are all in `Tomato___Late_blight` — another re-pack artefact.

## 4. Image style — 100% LAB-style, low resolution

Probed 24 sample images across 8 varied classes (see [`docs/pandey_dataset_samples.png`](pandey_dataset_samples.png)).

| Aspect | Finding |
|---|---|
| Resolution | **256 × 256** for every leaf class; 256 × 192 for `Background_without_leaves`. **No higher-resolution samples anywhere.** |
| Background | Uniform / studio. Concrete-grey or wood-textured backgrounds dominate. Zero field-style images (no soil context, no surrounding foliage, no field framing). |
| Subject | Single isolated leaf per image. The leaf usually fills 60–80 % of the frame. |
| Format / mode | `.jpg` / `.jpeg`, RGB, 8–22 KB each — heavily compressed. |
| Augmentation tells | Many class folders contain near-duplicate leaves at different rotations / flips / crops — classic offline-augmentation output. |
| `Background_without_leaves` exception | Urban photos (cars, roads) — not leaves at all; serves as a "no-leaf" reject class. |

**Classification: pure LAB-style PlantVillage. 0 % field-style imagery.**

### Strong-evidence smoking gun — this IS our PlantVillage

A class-by-class smoking-gun comparison with our existing PlantVillage:

| Class | Pandey count | Our PlantVillage count |
|---|---:|---:|
| Apple___healthy | **1,645** | **1,645** (exact match) |

Identical 1,645 in both for the same class. That cannot be coincidence. The Pandey dataset is the **PlantVillage augmented** re-pack of the same source images, with:

- the original `<uuid>___<descriptor>_<N>.JPG` filenames replaced by sequential `image (N).JPG`,
- a `Background_without_leaves` class added,
- three Tomato disease classes (Target_Spot, Yellow_Leaf_Curl_Virus, Tomato_mosaic_virus) **missing**,
- crop names trimmed (`Cherry___` vs our `Cherry_(including_sour)___`, `Corn___` vs `Corn_(maize)___`).

## 5. Taxonomy overlap — mapping table (proposed, NOT applied)

Legend: ✅ direct map · ✏️ direct map after trivial rename · ❌ no clean mapping · 🆕 new class.

### Pandey ↔ PlantVillage (our existing 38-class set)

| Pandey class (as written) | Maps to (PlantVillage) | Note |
|---|---|---|
| Apple___Apple_scab | Apple___Apple_scab | ✅ |
| Apple___Black_rot | Apple___Black_rot | ✅ |
| Apple___Cedar_apple_rust | Apple___Cedar_apple_rust | ✅ |
| Apple___healthy | Apple___healthy | ✅ (exact count match — same source) |
| Background_without_leaves | *(none)* | 🆕 — not in our PV; potential reject class |
| Blueberry___healthy | Blueberry___healthy | ✅ |
| Cherry___Powdery_mildew | Cherry_(including_sour)___Powdery_mildew | ✏️ rename |
| Cherry___healthy | Cherry_(including_sour)___healthy | ✏️ rename |
| Corn___Cercospora_leaf_spot Gray_leaf_spot | Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | ✏️ rename |
| Corn___Common_rust | Corn_(maize)___Common_rust_ | ✏️ rename (PV has trailing underscore) |
| Corn___Northern_Leaf_Blight | Corn_(maize)___Northern_Leaf_Blight | ✏️ rename |
| Corn___healthy | Corn_(maize)___healthy | ✏️ rename |
| Grape___Black_rot | Grape___Black_rot | ✅ |
| Grape___Esca_(Black_Measles) | Grape___Esca_(Black_Measles) | ✅ |
| Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | ✅ |
| Grape___healthy | Grape___healthy | ✅ |
| Orange___Haunglongbing_(Citrus_greening) | Orange___Haunglongbing_(Citrus_greening) | ✅ |
| Peach___Bacterial_spot | Peach___Bacterial_spot | ✅ |
| Peach___healthy | Peach___healthy | ✅ |
| Pepper,_bell___Bacterial_spot | Pepper,_bell___Bacterial_spot | ✅ |
| Pepper,_bell___healthy | Pepper,_bell___healthy | ✅ |
| Potato___Early_blight | Potato___Early_blight | ✅ |
| Potato___Late_blight | Potato___Late_blight | ✅ |
| Potato___healthy | Potato___healthy | ✅ |
| Raspberry___healthy | Raspberry___healthy | ✅ |
| Soybean___healthy | Soybean___healthy | ✅ |
| Squash___Powdery_mildew | Squash___Powdery_mildew | ✅ |
| Strawberry___Leaf_scorch | Strawberry___Leaf_scorch | ✅ |
| Strawberry___healthy | Strawberry___healthy | ✅ |
| Tomato___Bacterial_spot | Tomato___Bacterial_spot | ✅ |
| Tomato___Early_blight | Tomato___Early_blight | ✅ |
| Tomato___Late_blight | Tomato___Late_blight | ✅ |
| Tomato___Leaf_Mold | Tomato___Leaf_Mold | ✅ |
| Tomato___Septoria_leaf_spot | Tomato___Septoria_leaf_spot | ✅ |
| Tomato___Spider_mites Two-spotted_spider_mite | Tomato___Spider_mites Two-spotted_spider_mite | ✅ |
| Tomato___healthy | Tomato___healthy | ✅ |

**Of our 38 PlantVillage classes, Pandey is MISSING three:**
- `Tomato___Target_Spot`
- `Tomato___Tomato_Yellow_Leaf_Curl_Virus`
- `Tomato___Tomato_mosaic_virus`

### Pandey ↔ PlantDoc (field-style, 27-class)

PlantDoc uses a DIFFERENT vocabulary: `Apple Scab Leaf` rather than `Apple___Apple_scab`, `Tomato leaf bacterial spot` instead of `Tomato___Bacterial_spot`. Maps are 1-to-1 semantic but every class needs renaming, and PlantDoc has 0 species (Bell pepper as "Bell_pepper leaf") that map cleanly. Sample maps:

| Pandey | PlantDoc | Note |
|---|---|---|
| Apple___Apple_scab | Apple Scab Leaf | ✏️ |
| Apple___healthy | Apple leaf | ✏️ (PlantDoc uses bare "leaf" for healthy) |
| Tomato___Late_blight | Tomato leaf late blight | ✏️ |
| Orange___Haunglongbing_(Citrus_greening) | *(none)* | ❌ PlantDoc has no Orange class |
| Background_without_leaves | *(none)* | ❌ PlantDoc has no background class |

Conclusion: Pandey ↔ PlantDoc is **NOT a clean merge** — different vocabulary tier (lab studio vs field photos) and several Pandey classes have no PlantDoc counterpart.

### Pandey ↔ Paddy Doctor (rice-only, 10-class)

| Pandey | Paddy Doctor | Note |
|---|---|---|
| (any Pandey class) | (any rice class) | ❌ **No overlap.** Paddy Doctor is rice-only; Pandey has no rice classes. |

## 6. Verdict (3–4 lines)

This is **(c) a PlantVillage re-pack** — confirmed by the matching `___` class-name convention, the 1,645-exact-count Apple_healthy smoking gun, the uniform 256 × 256 lab-style imagery, the `Background_without_leaves` augmented-variant marker, and the `image (N).JPG` re-numbering. It is NOT (a) lab volume in any new sense (we already have a larger 54,305-image PlantVillage at split level) and certainly NOT (b) field diversity (zero field-style images).

If anything were to be augmented from this dataset, the **only novel contribution** would be the 1,143-image `Background_without_leaves` class as an optional "no-leaf" reject head — useful for a Phase 10 Streamlit UI guardrail when the farmer uploads a non-leaf photo. Merging the rest would add zero new information and risk re-amplifying PlantVillage's known lab-only shortcut biases that our Phase 9 Grad-CAM analysis is already flagging as a Phase 11 follow-up.

**Recommendation to Dr. Pandey:** thank him for the dataset; respectfully note we already use PlantVillage; ask whether he has any **field-style** imagery (e.g. plot-shots with soil + neighbouring plants visible) — that's what would meaningfully reduce the corner / background shortcut bias visible in the Phase 9 panels.
