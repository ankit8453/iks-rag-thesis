# Llama's queries vs the hand-written evaluation set (§6o)

Compared on 17 disease-labelled queries, scored by the pipeline's own cross-encoder (top-1 relevance).

| | mean score | strong | marginal | none |
|---|---|---|---|---|
| hand-written | 0.1691 | 1 | 5 | 11 |
| Llama (Strategy B) | 0.0445 | 1 | 0 | 16 |

Per-query: Llama better on 0, hand-written better on 14, tied on 3 (tie = within 0.05).

**Verdict.** THE HAND-WRITTEN QUERIES ARE BETTER. The deployed bridge is writing weaker queries than it could, so real-world performance is worse than the evaluation suggests. Fix the Strategy-B prompt, cheapest first: few-shot examples taken from the corpus, then a corpus word list, then two-round retrieval. Not fine-tuning.

---

## Query by query

**q17** — Squash Powdery mildew leaf  ·  hand 0.2927 → llama 0.0069 (-0.2858)
- hand : _white powdery coating spreading over the surface of the leaves_
- llama: _In loam soil with moderate moisture, where the leaves of the squash exhibit a white powdery coating, particularly noticeable on the underside, what symptoms and remedies are described in the classical Sanskrit treatises for this condition?._

**q13** — Tomato leaf mosaic virus  ·  hand 0.2706 → llama 0.0306 (-0.2400)
- hand : _mottled light and dark green patches with puckering and curling of the leaves_
- llama: _In loam soil with moderate moisture, where the leaves of the tomato plant display mottled yellow-green discoloration and curling, what are the symptoms and remedies described in the classical Sanskrit treatises?._

**q16** — Bell_pepper leaf spot  ·  hand 0.2258 → llama 0.0151 (-0.2107)
- hand : _dark sunken spots scattered across the leaves that later dry out_
- llama: _In loam soil with moderate moisture, where the leaves of the bell pepper exhibit small, dark spots with pale or ringed centres, and a medium texture, what are the described symptoms and their underlying causes?._

**q02** — Apple rust leaf  ·  hand 0.2239 → llama 0.0356 (-0.1883)
- hand : _orange to brown powdery pustules raised on the surface of tree leaves_
- llama: _In loam soil with moderate moisture, where apple plants exhibit small, orange-to-brown powdery pustules on their leaves, what are the symptoms and remedies described in the classical Sanskrit treatises?._

**q04** — Corn leaf blight  ·  hand 0.1669 → llama 0.0061 (-0.1608)
- hand : _long brown withering patches spreading and drying the leaves_
- llama: _In loam soil with moderate moisture, where corn is grown, what plant affliction presents with spreading brown necrotic patches on its leaves._

**q14** — Tomato leaf yellow virus  ·  hand 0.5755 → llama 0.4180 (-0.1575)
- hand : _leaves turning yellow, cupping upward and the plant remaining stunted_
- llama: _In loam soil with moderate moisture, where the medium texture is not visibly affected, and on a plant that is not specified, what are the symptoms described as yellowing of leaves with mottled yellow-green discoloration and curling, resembling a disease that affects the foliage of a tomato._

**q11** — Tomato leaf bacterial spot  ·  hand 0.1361 → llama 0.0133 (-0.1228)
- hand : _small greasy dark specks on leaves that later dry and tear_
- llama: _In loam soil with moderate moisture, where the leaves of the tomato plant exhibit small, dark spots with pale centres and a fine, dark border, and the soil feels medium in texture._

**q07** — Potato leaf late blight  ·  hand 0.1493 → llama 0.0342 (-0.1151)
- hand : _rapidly spreading water-soaked blackening patches rotting the leaves in damp weather_
- llama: _In loam soil with moderate moisture, where the potato plants display spreading brown necrotic patches on their leaves, what are the observable symptoms and underlying causes described in the classical Sanskrit treatises?._

**q05** — Corn Gray leaf spot  ·  hand 0.1175 → llama 0.0042 (-0.1133)
- hand : _greyish rectangular dead spots forming between the veins of the leaf_
- llama: _In loam soil with moderate moisture, where corn is grown, what plant affliction presents with small, dark, circular lesions with pale centers, often surrounded by a ring of lighter tissue, and is characterized by a general decline in plant vigor?._

**q18** — grape leaf black rot  ·  hand 0.1218 → llama 0.0213 (-0.1005)
- hand : _dark sunken decaying areas on leaves and shrivelling of the fruit on a climbing vine_
- llama: _In loam soil with moderate moisture, where the grape plants display dark sunken decaying areas on their leaves, what are the observable symptoms and underlying causes described in the classical Sanskrit treatises?._

**q10** — Tomato Septoria leaf spot  ·  hand 0.1043 → llama 0.0196 (-0.0847)
- hand : _numerous small dark spots with pale centres covering the leaves_
- llama: _In loam soil with moderate moisture, where the leaves of the tomato exhibit distinct dark spots with pale or ringed centres, what are the symptoms and remedies described in the classical Sanskrit treatises?._

**q12** — Tomato mold leaf  ·  hand 0.0983 → llama 0.0161 (-0.0822)
- hand : _fuzzy discoloured mould growing on the underside of the leaves_
- llama: _In loam soil with moderate moisture, where the leaves of the tomato plant are covered in a fuzzy, discoloured, and somewhat soft growth, what are the symptoms and remedies described in the classical Sanskrit treatises?._

**q01** — Apple Scab Leaf  ·  hand 0.1039 → llama 0.0264 (-0.0775)
- hand : _dark rough corky lesions and scabby patches spreading over the leaves of a tree_
- llama: _In loam soil with moderate moisture, where apple plants exhibit dark, rough, corky lesions on their leaves, what are the observable symptoms and underlying causes described in the classical Sanskrit treatises?._

**q03** — Corn rust leaf  ·  hand 0.0828 → llama 0.0090 (-0.0738)
- hand : _rust-coloured powdery eruptions scattered along the leaf blade_
- llama: _In loam soil with moderate moisture, where corn is grown, what plant affliction presents with orange-to-brown powdery pustules on its leaves?._

**q08** — Tomato Early blight leaf  ·  hand 0.0669 → llama 0.0225 (-0.0444)
- hand : _ringed brown lesions enlarging on the leaves and causing them to fall_
- llama: _In loam soil with moderate moisture, where the leaves of the tomato plant display spreading brown necrotic patches with yellowish tinted edges, what symptoms and remedies are described in the classical Sanskrit treatises?._

**q06** — Potato leaf early blight  ·  hand 0.0765 → llama 0.0342 (-0.0423)
- hand : _dark brown spots with concentric rings appearing first on the older lower leaves_
- llama: _In loam soil with moderate moisture, where the potato plants display spreading brown necrotic patches on their leaves, what are the observable symptoms and underlying causes described in the classical Sanskrit treatises?._

**q09** — Tomato leaf late blight  ·  hand 0.0617 → llama 0.0430 (-0.0187)
- hand : _dark decaying blotches with pale mould at the leaf margins during wet weather_
- llama: _In loam soil with moderate moisture, where the leaves of the tomato plant exhibit spreading brown necrotic patches with a soft, wet, and decaying appearance, what are the described symptoms and their underlying causes?._
