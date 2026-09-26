# Llama's queries vs the hand-written evaluation set (§6o)

Compared on 17 disease-labelled queries, scored by the pipeline's own cross-encoder (top-1 relevance).

| | mean score | strong | marginal | none |
|---|---|---|---|---|
| hand-written | 0.1691 | 1 | 5 | 11 |
| Llama (Strategy B) | 0.0977 | 2 | 1 | 14 |

Per-query: Llama better on 1, hand-written better on 11, tied on 5 (tie = within 0.05).

**Verdict.** THE HAND-WRITTEN QUERIES ARE BETTER. The deployed bridge is writing weaker queries than it could, so real-world performance is worse than the evaluation suggests. Fix the Strategy-B prompt, cheapest first: few-shot examples taken from the corpus, then a corpus word list, then two-round retrieval. Not fine-tuning.

---

## Query by query

**q17** — Squash Powdery mildew leaf  ·  hand 0.2927 → llama 0.0134 (-0.2793)
- hand : _white powdery coating spreading over the surface of the leaves_
- llama: _Leaves covered in a white powdery coating on a squash plant._

**q16** — Bell_pepper leaf spot  ·  hand 0.2258 → llama 0.0085 (-0.2173)
- hand : _dark sunken spots scattered across the leaves that later dry out_
- llama: _Distinct dark spots with pale centres appearing on the leaves of a bell pepper._

**q02** — Apple rust leaf  ·  hand 0.2239 → llama 0.0323 (-0.1916)
- hand : _orange to brown powdery pustules raised on the surface of tree leaves_
- llama: _Leaves with orange-to-brown powdery pustules on apple._

**q07** — Potato leaf late blight  ·  hand 0.1493 → llama 0.0111 (-0.1382)
- hand : _rapidly spreading water-soaked blackening patches rotting the leaves in damp weather_
- llama: _Leaves with spreading brown necrotic patches._

**q18** — grape leaf black rot  ·  hand 0.1218 → llama 0.0145 (-0.1073)
- hand : _dark sunken decaying areas on leaves and shrivelling of the fruit on a climbing vine_
- llama: _dark sunken decaying areas spreading over the leaves of a grape._

**q11** — Tomato leaf bacterial spot  ·  hand 0.1361 → llama 0.0353 (-0.1008)
- hand : _small greasy dark specks on leaves that later dry and tear_
- llama: _Distinct dark spots with pale centres appearing on the leaves of a tomato._

**q12** — Tomato mold leaf  ·  hand 0.0983 → llama 0.0123 (-0.0860)
- hand : _fuzzy discoloured mould growing on the underside of the leaves_
- llama: _fine stippling and bronzing on the leaves of a tomato._

**q10** — Tomato Septoria leaf spot  ·  hand 0.1043 → llama 0.0279 (-0.0764)
- hand : _numerous small dark spots with pale centres covering the leaves_
- llama: _Distinct dark spots with pale or ringed centres on the leaves of a tomato._

**q05** — Corn Gray leaf spot  ·  hand 0.1175 → llama 0.0413 (-0.0762)
- hand : _greyish rectangular dead spots forming between the veins of the leaf_
- llama: _Leaves with small, circular, dark spots with pale centres on a corn plant._

**q03** — Corn rust leaf  ·  hand 0.0828 → llama 0.0118 (-0.0710)
- hand : _rust-coloured powdery eruptions scattered along the leaf blade_
- llama: _Leaves with orange-to-brown powdery pustules on the corn._

**q06** — Potato leaf early blight  ·  hand 0.0765 → llama 0.0261 (-0.0504)
- hand : _dark brown spots with concentric rings appearing first on the older lower leaves_
- llama: _Leaves with spreading brown necrotic patches on a potato._

**q08** — Tomato Early blight leaf  ·  hand 0.0669 → llama 0.0222 (-0.0447)
- hand : _ringed brown lesions enlarging on the leaves and causing them to fall_
- llama: _Leaves with spreading brown necrotic patches on a tomato plant._

**q09** — Tomato leaf late blight  ·  hand 0.0617 → llama 0.0222 (-0.0395)
- hand : _dark decaying blotches with pale mould at the leaf margins during wet weather_
- llama: _Leaves with spreading brown necrotic patches on a tomato plant._

**q01** — Apple Scab Leaf  ·  hand 0.1039 → llama 0.0750 (-0.0289)
- hand : _dark rough corky lesions and scabby patches spreading over the leaves of a tree_
- llama: _dark rough corky patches spreading over the leaves of an apple tree._

**q14** — Tomato leaf yellow virus  ·  hand 0.5755 → llama 0.5824 (+0.0069)
- hand : _leaves turning yellow, cupping upward and the plant remaining stunted_
- llama: _leaves turning yellow with excessive paleness and gradual shedding on a tomato plant._

**q04** — Corn leaf blight  ·  hand 0.1669 → llama 0.2001 (+0.0332)
- hand : _long brown withering patches spreading and drying the leaves_
- llama: _Leaves withering and drying, turning brown with spreading necrotic patches on a corn plant._

**q13** — Tomato leaf mosaic virus  ·  hand 0.2706 → llama 0.5243 (+0.2537)
- hand : _mottled light and dark green patches with puckering and curling of the leaves_
- llama: _leaves turning mottled yellow-green with gradual curling and distortion._
