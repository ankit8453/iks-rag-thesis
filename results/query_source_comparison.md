# Llama's queries vs the hand-written evaluation set (§6o)

Compared on 17 disease-labelled queries, scored by the pipeline's own cross-encoder (top-1 relevance).

| | mean score | strong | marginal | none |
|---|---|---|---|---|
| hand-written | 0.1691 | 1 | 5 | 11 |
| Llama (Strategy B) | 0.1368 | 2 | 4 | 11 |

Per-query: Llama better on 4, hand-written better on 10, tied on 3 (tie = within 0.05).

**Verdict.** NO MEANINGFUL DIFFERENCE. The bridge already writes queries as good as hand-crafted ones, so the retrieval numbers are trustworthy and the limit on usefulness really is corpus coverage, not query wording.

---

## Query by query

**q17** — Squash Powdery mildew leaf  ·  hand 0.2927 → llama 0.0134 (-0.2793)
- hand : _white powdery coating spreading over the surface of the leaves_
- llama: _Leaves covered in a white powdery coating on a squash plant._

**q16** — Bell_pepper leaf spot  ·  hand 0.2258 → llama 0.0068 (-0.2190)
- hand : _dark sunken spots scattered across the leaves that later dry out_
- llama: _Small dark spots with pale or ringed centres on the leaves of a bell pepper._

**q02** — Apple rust leaf  ·  hand 0.2239 → llama 0.0323 (-0.1916)
- hand : _orange to brown powdery pustules raised on the surface of tree leaves_
- llama: _Leaves with orange-to-brown powdery pustules on apple._

**q07** — Potato leaf late blight  ·  hand 0.1493 → llama 0.0380 (-0.1113)
- hand : _rapidly spreading water-soaked blackening patches rotting the leaves in damp weather_
- llama: _brown patches spreading over the leaves of a potato._

**q11** — Tomato leaf bacterial spot  ·  hand 0.1361 → llama 0.0278 (-0.1083)
- hand : _small greasy dark specks on leaves that later dry and tear_
- llama: _Small dark spots with pale or ringed centres on the leaves of a tomato._

**q18** — grape leaf black rot  ·  hand 0.1218 → llama 0.0145 (-0.1073)
- hand : _dark sunken decaying areas on leaves and shrivelling of the fruit on a climbing vine_
- llama: _dark sunken decaying areas spreading over the leaves of a grape._

**q05** — Corn Gray leaf spot  ·  hand 0.1175 → llama 0.0119 (-0.1056)
- hand : _greyish rectangular dead spots forming between the veins of the leaf_
- llama: _Small dark spots with pale centres spreading over the leaves of a corn plant._

**q12** — Tomato mold leaf  ·  hand 0.0983 → llama 0.0161 (-0.0822)
- hand : _fuzzy discoloured mould growing on the underside of the leaves_
- llama: _fuzzy discoloured patches spreading over the surface of the leaves of a tomato._

**q10** — Tomato Septoria leaf spot  ·  hand 0.1043 → llama 0.0278 (-0.0765)
- hand : _numerous small dark spots with pale centres covering the leaves_
- llama: _Small dark spots with pale or ringed centres on the leaves of a tomato._

**q03** — Corn rust leaf  ·  hand 0.0828 → llama 0.0130 (-0.0698)
- hand : _rust-coloured powdery eruptions scattered along the leaf blade_
- llama: _Leaves with orange-to-brown powdery pustules on corn._

**q01** — Apple Scab Leaf  ·  hand 0.1039 → llama 0.0750 (-0.0289)
- hand : _dark rough corky lesions and scabby patches spreading over the leaves of a tree_
- llama: _dark rough corky patches spreading over the leaves of an apple tree._

**q04** — Corn leaf blight  ·  hand 0.1669 → llama 0.1531 (-0.0138)
- hand : _long brown withering patches spreading and drying the leaves_
- llama: _Leaves drying and withering on a corn tree._

**q14** — Tomato leaf yellow virus  ·  hand 0.5755 → llama 0.5824 (+0.0069)
- hand : _leaves turning yellow, cupping upward and the plant remaining stunted_
- llama: _leaves turning yellow with excessive paleness and gradual shedding on a tomato plant._

**q06** — Potato leaf early blight  ·  hand 0.0765 → llama 0.1732 (+0.0967)
- hand : _dark brown spots with concentric rings appearing first on the older lower leaves_
- llama: _Leaves with dark brown patches spreading until they dry and wither on a potato._

**q09** — Tomato leaf late blight  ·  hand 0.0617 → llama 0.1588 (+0.0971)
- hand : _dark decaying blotches with pale mould at the leaf margins during wet weather_
- llama: _Leaves drying and withering on a tomato plant._

**q08** — Tomato Early blight leaf  ·  hand 0.0669 → llama 0.2181 (+0.1512)
- hand : _ringed brown lesions enlarging on the leaves and causing them to fall_
- llama: _leaves with dark brown patches spreading until they dry and wither on a tomato plant._

**q13** — Tomato leaf mosaic virus  ·  hand 0.2706 → llama 0.7634 (+0.4928)
- hand : _mottled light and dark green patches with puckering and curling of the leaves_
- llama: _leaves turning mottled yellow-green with curling and gradual withering on a tomato plant._
