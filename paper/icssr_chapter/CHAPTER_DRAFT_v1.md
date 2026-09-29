# An IKS-Grounded Multimodal Advisory System: Connecting Image-Based Plant Diagnosis with Classical Indian Agricultural Texts

**Ankit Pawar**
M.Tech (Computer Science and Engineering), PDPM Indian Institute of Information Technology, Design and Manufacturing, Jabalpur

**Dr. Akshay Pandey**
Department of Computer Science and Engineering, PDPM Indian Institute of Information Technology, Design and Manufacturing, Jabalpur

*Revised and extended version of the paper presented at the International Seminar on Revitalising Indigenous Knowledge Systems: Tribal Heritage, Cultural Continuity and Social Transformation in India, Iswar Saran P.G. College, University of Allahabad, Prayagraj, 11–12 September 2026.*

---

## Abstract

India's classical agricultural treatises describe a sustainable, organic practice of plant care that survives today only in printed translations, organised by the observed condition of the plant rather than by the disease names used in modern diagnosis, and absent from the digital tools on which farmers and extension workers increasingly rely. This chapter reports the construction and evaluation of an advisory system that connects photograph-based recognition of plant disease and soil condition with the treatment knowledge of six such treatises, and that is constrained to cite the passage it draws on or to state that the texts are silent. Its three components are a verse-level digital corpus of 270 passages; a semantic bridge that re-expresses a modern diagnostic label in the descriptive vocabulary of the texts; and a generator restricted to retrieved passages. Evaluated on forty questions, the final system returned a cited, verifiable answer for 51.9 per cent of answerable questions, up from 13.6 per cent at the first honest measurement, and in no evaluation run did it fabricate a citation. The most consequential result, however, is a boundary: eleven of seventeen questions phrased as modern disease names have no answer in any of the seven sources, because the treatises organise plant disorder by cause—imbalance of wind, bile and phlegm, over-watering, insects, unhealthy soil—while modern diagnosis organises it by the appearance of the lesion. Adding the one treatise devoted to plant disease did not move that boundary. The chapter argues that a faithful digital representation of indigenous knowledge must preserve its organising categories and be honest about where they stop, and records the digitisation and evaluation errors that had to be found by reading the texts rather than trusting the numbers.

**Keywords:** Indian Knowledge Systems; Vrikshayurveda; digital preservation; retrieval-augmented generation; artificial intelligence and faithfulness

---

## 1. Introduction

A farmer photographs a diseased leaf. Software can now name the disease with useful accuracy. What it cannot do is tell the farmer what the Indian agricultural tradition would have advised, because that advice lives in Sanskrit treatises that exist, for practical purposes, only as scanned English translations with no searchable structure. The knowledge has been documented; it has not been made usable.

Six treatises are at issue here. The *Vrikshayurveda* of Surapala (c. tenth century) is the fullest classical statement of plant care, from soil selection and planting through nourishment, disease and treatment (Sadhale, 1996). Varahamihira's *Brihat Samhita* devotes chapters to the prediction of rainfall, the growth of crops, the exploration of underground water and the treatment of trees (Bhat, 1981). *Krishi-Parashara* is a manual of the agricultural year: seasons, sowing, rain signs, cattle (Majumdar & Banerji, 1960). The *Upavanavinoda* of Sarangadhara covers arboriculture and garden layout (Majumdar, 1935). *Kashyapiyakrishisukti* is a treatise on cultivation—land selection, water, seed, sowing and harvest (Jugnu, 2013). And the *Vishvavallabha* of Chakrapani Mishra, composed around 1577 at the court of Maharana Pratap of Mewar, is a nine-chapter work on plant life in an arid region, with the largest dedicated chapter on plant disease and its treatment in the tradition (Sadhale, 2004). Together they record an organic and sustainable practice: liquid manures such as *kunapajala*, seed treatments with milk and mustard, decoctions of bark and leaf, careful reading of the soil and the sky (Nene, 2006).

Three barriers stand between these texts and use. First, they have no structured digital form. Second, they describe a plant's trouble in terms of what is observed and what has caused it, not in terms of the disease taxonomy—scab, rust, blight, Septoria leaf spot—through which modern diagnosis proceeds; a modern label therefore retrieves nothing from them. Third, the AI tools now used for agricultural advice neither draw on these sources nor guarantee that what they say about them is true. A general-purpose language model asked about *Vrikshayurveda* will produce a fluent, confident, and frequently invented remedy, attributed to a verse that does not exist. For heritage knowledge this is not an ordinary error. It is a distortion of the tradition itself, presented to a reader who has no practical means of checking it.

This chapter reports a system built to address all three barriers, and what its evaluation revealed. The system takes a photograph of a leaf, a photograph of the soil and the farmer's crop name; recognises the disease and the soil condition; re-expresses the diagnosis in the descriptive language the treatises use; searches a verse-level corpus of the six texts; and returns advice that cites the passage it rests on—or, when no passage addresses the observed condition, says so and stops. The chapter is a revised and extended version of the paper presented at the seminar. It adds the sixth treatise, a complete evaluation on the wording the system itself generates rather than on hand-written test questions, and the finding that emerged from that evaluation: a measured boundary between where the classical and the modern accounts of plant disease meet, and where they do not.

The chapter proceeds as follows. Section 2 describes the texts and how they think about plant health. Section 3 sets out why artificial intelligence applied to heritage knowledge needs a discipline of its own. Section 4 describes the system in plain terms. Section 5 records what digitising the texts taught us. Section 6 explains how the system was evaluated and Section 7 what was found. Section 8 discusses what the boundary between the two traditions means for digital preservation, and Sections 9 and 10 give limitations and conclusions.

## 2. The texts, and how they think about plant health

Table 1 lists the six treatises and the modern source that was added alongside them.

**Table 1.** Sources in the digital corpus.

| Source | Date | Subject | Passages |
|---|---|---|---|
| *Vrikshayurveda* (Surapala) | c. 10th c. | plant care, disorders, causes and remedies, planting, watering, seed treatment | 52 |
| *Brihat Samhita* (Varahamihira), ch. 21–29, 40, 54, 55 | 6th c. | rainfall prognostication, growth of crops, exploration of water springs, treatment of trees | 46 |
| *Kashyapiyakrishisukti* (Kashyapa) | early medieval | cultivation: land, water, seed, sowing, harvest, storage | 39 |
| *Vishvavallabha* (Chakrapani Mishra) | c. 1577 | groundwater, reservoirs, soil, planting, care, nourishment, diseases and treatment, botanical wonders | 37 |
| *Upavanavinoda* (Sarangadhara) | 13th c. | arboriculture and garden layout | 15 |
| *Krishi-Parashara* (Parashara) | c. 1st millennium | seasons, sowing calendar, rain signs, ploughing | 13 |
| NITI Aayog natural-farming manual (modern) | 2026 | insect-pest management, bio-inputs, seed treatment, soil health | 68 |
| **Total** | | | **270** |

The feature of these texts that turned out to govern everything that follows is how they organise plant disorder. They do so by *cause*, in the same humoral framework that Ayurveda applies to the human body. *Vishvavallabha* opens its disease chapter in exactly these terms: "Like human beings, trees also suffer from diseases due to imbalance in wind, bile, and phlegm. As such, I shall describe hereunder their symptoms and remedies" (Sadhale, 2004, VIII.2). The chapter then proceeds cause by cause—wind disorder, bile disorder, phlegm disorder, indigestion from over-watering, indigestion from over-manuring, over-medication, insects, unhealthy soil, the wrong season—and for each gives the symptoms and the remedy. *Vrikshayurveda* has the same structure. Its account of causes runs from scorching heat and insect-eaten roots, through storm damage, fire, lightning and wounds, to the imbalance of the three elements, and its treatments follow: "Diseases caused by imbalance of *vata* can be cured by flesh, marrow, and ghee. The sprinkling of *kunapa* water also removes all the disorders caused by the *vata* element" (Sadhale, 1996, v. 185); for the *kapha* type, "the paste of white mustard should be deposited at the root and the trees should be watered with a mixture of sesame and ashes" (v. 188).

The symptoms these texts describe are correspondingly general: paleness, yellowness, drying, withering, the shedding of leaves and fruit, dieback at the tips of branches, bark falling off, oozing, failure to flower. A tree affected by wind disorder "is dry, small, slender, tall, sleepy, and faded. It bears no, or very few, flowers or fruits" (Sadhale, 2004, VIII.9). A tree suffering from over-watering shows "sleeplessness, paleness, falling of shoots, drying of branches at the end, infestation by ants, foul smell like that of fish" (VIII.25). What the texts do not describe, anywhere, is the appearance of a lesion: they have no vocabulary of spots with pale centres, of pustules, of powdery coatings or corky patches.

A second feature matters almost as much. The remedies are, with a few exceptions, general to the symptom rather than specific to the crop. A remedy given for yellowing applies to any plant that yellows. This was confirmed for us by Dr. Sunita T. Pandey, an agronomist who has worked on *kunapajala*, and it is borne out by the one previous computational treatment of *Vrikshayurveda* we are aware of, an expert system that diagnoses purely from questions about symptoms and never asks which crop is affected (Rananavare & Chitnis, 2024). The qualification is real—Surapala does give some crop-named recipes for mango, pomegranate and coconut—so the claim is that the texts are predominantly symptom-general, not that they are wholly so.

Modern plant pathology cuts the same reality along a different line. It names diseases by causal organism and by the appearance of the damage: apple scab, corn rust, Septoria leaf spot, early and late blight, powdery mildew, black rot. Its diagnostic vocabulary is a vocabulary of lesions. The two traditions are not in disagreement about the plant; they are asking it different questions. That difference is the heart of this chapter, and Section 7 returns to it with numbers.

## 3. Why artificial intelligence and heritage knowledge need a discipline of their own

Large language models produce fluent text, and they produce it whether or not they know the answer. The literature calls the second case hallucination (Ji et al., 2023): a confident statement with no basis in any source. In most settings this is a nuisance. Applied to indigenous knowledge it is something worse, for three reasons.

The reader cannot check. A farmer or a student who is told that *Vrikshayurveda* prescribes a particular decoction for a particular disease, with a chapter and verse attached, has no way of verifying that the verse exists, let alone that it says what is claimed. The asymmetry between the fluency of the answer and the difficulty of checking it is where trust gets abused.

The error attaches itself to a real tradition. An invented remedy attributed to a real text does not merely mislead about the plant; it misrepresents what the tradition said. Repeated at scale, it becomes a false record of the knowledge system, indistinguishable from the true one to anyone who did not read the original.

And the tradition's own boundaries are erased. Every knowledge system has things it does not address. A model that answers every question, drawing on nothing in particular, represents the tradition as covering ground it never covered.

The system described here therefore rests on a single design rule, stated to the generator as an instruction it is not permitted to break: answer only from the passages you have been given; cite the source text, chapter and verse for every claim; and if the passages do not address the observed condition, say that the texts do not contain enough information, and stop. Refusal, under this rule, is not a failure of the system. It is the system being faithful to a silence in the sources. We would rather the farmer received nothing than received a fabrication presented as heritage. That principle, which sounds obvious, turned out to be the one the evaluation kept testing.

## 4. What was built

Figure 1 shows the system. Its parts are described here in plain terms; full technical detail is given in the thesis on which this chapter draws.

![Figure 1. The advisory system. The generator may use only the retrieved passages; if none addresses the observed condition it declines rather than answers.](figures/fig1_architecture.png)

**The digital corpus.** Each treatise was obtained as a scanned translation, converted to text by optical character recognition, cleaned, and divided into passages of roughly verse-group length—two to three thousand characters each. Every passage carries its book, chapter, verse range and translator, so that a citation can be traced back to a page. *Vrikshayurveda* and *Vishvavallabha* were taken from the Asian Agri-History Foundation's translations (Sadhale, 1996, 2004), *Brihat Samhita* from Bhat's (1981), and *Krishi-Parashara* and *Upavanavinoda* from Majumdar's editions (Majumdar & Banerji, 1960; Majumdar, 1935). *Kashyapiyakrishisukti* was available to us only as a Sanskrit text and was translated for this project with machine assistance; a verse-by-verse check of that translation against the Chowkhamba edition (Jugnu, 2013) has been prepared and is in progress, and its passages should be read with that caveat. *Vishvavallabha* was located last, as a complete scan of the Foundation's edition on the Internet Archive, and its English translation was recovered from that archive's own OCR and verified verse by verse against the page images. A modern government manual on natural farming (NITI Aayog, 2026) was added alongside the classical texts, kept in a separate tier so that classical and modern sources are never confused in a citation. The corpus holds 270 passages from seven sources.

**Two vision models.** A disease model, an EfficientNet-B4 network (Tan & Le, 2019), was trained in three stages on public photograph collections of increasing difficulty: laboratory images (Hughes & Salathé, 2015), field images of rice (Petchiammal et al., 2022), and photographs taken in the wild (Singh et al., 2020). It reached 72.3 per cent accuracy on the hardest of these, which is at the published frontier for that collection. But an interpretability check using Grad-CAM heat-maps (Selvaraju et al., 2017)—the same kind of check used in the plant-disease work of Pandey and Jain (2022)—showed that the model was right for the wrong reason. It was attending to the background of the photograph, the soil and the sky, rather than to the leaf. Three attempted fixes failed, and are reported as such in the thesis. The one that worked was to retrain the model on images cropped to the leaf alone. Accuracy fell to 66.6 per cent, and the heat-maps moved onto the lesion. We regard a lower number earned from the leaf as worth more than a higher number earned from the background, and it is the leaf-focused model that the system uses. A second, smaller model reads the soil photograph for soil type (89.9 per cent accuracy), moisture (95.8 per cent) and texture (67.9 per cent).

**The symptom bridge.** This is the component that makes the connection possible, and it exists because of the vocabulary gap described in Section 2. The disease model produces a label such as "Apple scab". The treatises contain no such phrase, and a search for it finds nothing: in a direct comparison, searching with the modern label scored between 0.01 and 0.04 on the retrieval measure we use, where 1.0 is a perfect match. The bridge asks a language model, Llama 3.1 (Grattafiori et al., 2024), to rewrite the label as a description of what would be seen on the plant, in the observational register the texts use: "dark rough corky patches spreading over the leaves". Searched with that description, the same treatises scored between 0.59 and 0.96. The bridge is not a refinement; it is what makes the modern label reach the classical text at all. Section 7 reports what happened when we tested the bridge's actual output rather than assuming it.

**Retrieval and grounded generation.** The rewritten description is used to search the corpus by meaning, using sentence embeddings (Xiao et al., 2023) combined, by reciprocal rank fusion (Cormack et al., 2009), with a conventional keyword index (Robertson & Zaragoza, 2009), and the candidates are re-ranked by a second model that reads query and passage together. The top five passages, each headed by its exact citation, are handed to the generator under the rule stated in Section 3. The generator writes the advice, or declines. This is the retrieval-augmented pattern of Lewis et al. (2020), with the abstention rule added and enforced.

**Crops the model was not trained on.** The disease model can only name diseases it has seen. Rather than let it guess silently, the farmer names the crop; the model names the disease; and where the crop lies outside the model's training the system says so, shows how confident it is, advises through the symptom bridge with an explicit caution, and routes uncertain cases to expert review rather than to the farmer.

## 5. What digitising the texts taught us

A digital archive can look complete and be wrong. Four things went wrong in ours, each found by reading rather than by any automatic check, and each is recorded here because an IKS digitisation project elsewhere will meet the same hazards.

**Optical character recognition can silently lose text.** A setting in the recognition tool truncated long pages: a page carrying some twenty verses came out as two. Nothing flagged it. It was caught by comparing the recognised text with the printed page, and both affected books had to be re-processed.

**A wrong boundary pulls in the wrong text.** The routine that located chapters in *Brihat Samhita* ended each wanted chapter at the next wanted chapter rather than at the next chapter of any kind. The four-page chapter on the growth of crops thereby absorbed the thirteen chapters that follow it—on the fluctuation of prices, the signs of swords, architecture, the milky ocean—and for a period a quarter of the corpus was material no one had chosen. The system had been retrieving astrology in answer to questions about leaves.

**Every passage must be uniquely citable.** Numbered lists in a book's front matter were read as verse numbers, so that five different passages all carried the label "verse 1.2". A citation to it could not say which passage was meant, and a measure of citation accuracy computed against such labels was meaningless. Every passage now has a label of its own.

**The evaluation can be wrong in the same ways as the archive.** Three times our own scoring misled us. A relevance score that measured word overlap rated sixteen of sixteen new test questions "answerable"; reading the passages gave eight. A citation format that the generator copied wrongly—writing "Source 2" where the book's name belonged—made a well-grounded system look as though it were failing. And test labels written before a new treatise was added scored every passage from that treatise as irrelevant. In each case the remedy was the same: read the actual passage, or the actual answer, rather than trust the number.

The lesson for digital preservation is not that automation is untrustworthy but that it must be checked against the physical book at every stage, including the stage at which the digital object is judged.

## 6. How the system was evaluated

The system was tested on forty questions. Seventeen are phrased as modern disease names—the seventeen diseases the vision model can recognise, from apple scab to grape black rot—because those are the only questions the deployed system can generate. Twenty-three are ordinary farming questions of the kind a person might put to the texts directly: insects on the shoots, treating seed before sowing, whether a piece of land will support trees, enriching poor soil, making liquid manure, the season for sowing, the signs of coming rain, watering a young tree, finding water before digging a well, planting a sapling, a broken branch, a tree that will not flower, protecting stored grain.

Whether each question is answerable was decided by reading, in full, every passage the system retrieved for it. Twenty-seven questions are answerable. Eleven are not, although their subject lies within the treatises' scope, because no passage in any source addresses them. Two more are deliberate traps—questions about synthetic fungicide dosages and exact fertiliser percentages—that no classical text could answer, on which the correct behaviour is refusal.

Two decisions about the test bear stating. First, the disease questions were evaluated using the description the system's own bridge generates from the disease label, not a description written by hand. Earlier runs had used hand-written descriptions, and they overstated the system, because nothing in the deployed pipeline ever produces them. Second, the labels are ours, not yet an expert's, and every number below is preliminary until an agronomist and a Sanskritist have ratified them.

Four things were measured. Whether the search brought back a passage from the right treatise. Whether the answer cited a passage that had genuinely been retrieved. Whether the system declined when it should—on the thirteen unanswerable questions—and whether it declined when it should not, on the twenty-seven answerable ones. And, as a control, whether the same language model, given no passages at all, would invent citations to the texts.

## 7. What was found

### 7.1 The system did not invent

Across every evaluation run, on every version of the corpus, the model given no texts at all produced no citation to the texts: zero invented references. This is the result on which everything else rests. Whatever else the system gets wrong, it does not manufacture heritage.

On the thirteen unanswerable questions it declined, or answered only in part with a caveat, twelve times (92.3 per cent). The exception matched a description of corky scab patches to a passage about caterpillar holes: a real passage, honestly cited, wrongly applied. Citation, we learned, guarantees traceability, not correctness; a reader can at least see the passage and judge.

### 7.2 The texts answer ordinary farming questions well

Of the twenty-three ordinary questions, most were answered directly and from the expected place. *"How much and how often should a young tree be watered?"* found *Vrikshayurveda*'s schedule by soil and season: in marshy land once in five days, in ordinary soil morning and evening for ten days, then every alternate day in winter (Sadhale, 1996, vv. 110–111). *"Natural signs that rain is coming"* found *Brihat Samhita*'s list—ants shifting their eggs, the mating of snakes, chameleons on treetops fixing their gaze on the sky (Bhat, 1981, ch. 28). *"A branch broken by strong wind"* found the instruction to dress the spot with honey and ghee and sprinkle it with milk and water. *"Preparing a liquid manure"* found, in *Vishvavallabha*, the recipe for *kunapa* itself: fat, marrow, skin and blood cooked in water, milk and cold water added, sesame oilcake, honey and ghee stirred in, the pot kept warm for a fortnight (Sadhale, 2004, VII.2–3). *"Protecting harvested grain"* found *Kashyapiyakrishisukti*'s instructions on earthen pots, wooden bins and pits in firm ground, guarded against damp, mice and parrots.

### 7.3 Where the two traditions do not meet

Of the seventeen disease-name questions, eleven have no answer in any of the seven sources. The searches did not fail; the concept is absent. There is no passage for Septoria leaf spot because the texts have no category into which a small dark spot with a pale centre would fall. There is none for powdery mildew, for rust pustules, for corky scab. The treatises describe a tree that is pale, or drying, or shedding, or failing to fruit, and prescribe accordingly; they do not describe what a lesion looks like. Figure 2 sets the two organising schemes side by side.

![Figure 2. The boundary the evaluation measured. Adding Vishvavallabha, the one treatise with a dedicated disease chapter, did not move it: that chapter is organised by cause too.](figures/fig2_boundary.png)

The strongest evidence for this reading came from the last text added. Before *Vishvavallabha* was in the corpus, it was reasonable to suppose that the gap lay in our choice of books: that a treatise with a proper disease chapter would fill it. *Vishvavallabha* has such a chapter—seventy-nine verses, the largest in the work. Adding it produced not a single new answer for a question about spots, pustules, powder or mould. Its chapter is organised by cause, exactly as Surapala's is, and describes symptoms in the same terms of paleness and drying. What it did add were answers about drying and sickly trees, insects living on the leaves, trees planted in unhealthy soil, and the *kunapa* recipe. Two independent treatises, five centuries apart, organise plant disease the same way. The boundary is in the tradition, not in the library.

A simple rule describes where the line falls. A question about how a lesion *looks* has no answer in these texts. A question about drying, paleness, soil, insects, manure or water has one, wholly or in part. The six disease-name questions that are answerable are the ones whose description crosses that line: the yellowing and mottling of a virus-infected leaf, the drying and withering of a blight, which the texts do describe.

### 7.4 The system's own wording mattered more than expected

The test that produced the sharpest lesson was the one we had not planned. Every retrieval number in the project had been measured on hand-written test questions, on the assumption that they resembled what the bridge produces. When the bridge's actual output was captured and scored, it was 3.8 times worse than the hand-written wording. Every one of its seventeen descriptions opened with the soil ("In loam soil with moderate moisture…") although the soil was not implicated; every one asked what the treatises say rather than describing the plant; and several kept modern pathological terms—"necrotic", "lesion"—that appear nowhere in the corpus. The language model was disobeying its instructions in a consistent way.

The fix was to rewrite the instructions: show the model the corpus's own register with worked examples, tell it explicitly that soil belongs in the description only when soil is implicated and never first, forbid the question form, and list the words the translations use. No retraining was needed. After two rounds the bridge's output scored level with the hand-written wording, and on one question—a mosaic virus—better. The wider point for anyone building on heritage texts is that the component that translates between vocabularies must be measured, not assumed; ours had been the weakest link for months without anyone knowing.

A side effect is worth recording. Forcing the corpus's vocabulary made the descriptions less specific. Septoria leaf spot, bacterial spot and a pepper leaf spot now produce the same description apart from the crop's name. We think this is correct rather than lossy: the texts do not distinguish those conditions either, and the bridge has stopped asserting distinctions the target tradition does not make.

### 7.5 The numbers

Table 2 gives the final results and the first honest measurement for comparison; Figure 3 shows the progression across the three evaluation stages.

**Table 2.** Answer quality on the forty-question test set.

| | First honest run (233 passages, hand-written wording) | Final (270 passages, system's own wording) |
|---|---|---|
| Answers backed by a genuinely retrieved citation | 13.6% | **51.9%** |
| Answerable questions refused outright | 81.8% | **40.7%** |
| Unanswerable questions correctly declined | 100% | 92.3% |
| Invented citations, model given no texts | 0% | **0%** |
| Search finds a correct treatise among the top five | 100% | 96.3% |

![Figure 3. Answers backed by a genuinely retrieved citation, and answerable questions refused outright, across the three evaluation stages. At every stage the model given no texts at all produced zero invented citations.](figures/fig3_results.png)

Grounded answers rose almost fourfold and outright refusal halved. What changed between the two columns was not the model but the honesty of the plumbing around it: unique passage labels; the bridge's instructions; a citation format the generator had been copying wrongly; fairer scoring of answers that answer part of a question and decline the rest; three test labels corrected on reading; and the sixth treatise. One figure moved the other way and is reported as such: the proportion of citations pointing at the *exact* passage used, as opposed to a passage from the right book, fell from 52.6 to 39.6 per cent in the final run, and the session data needed to establish why was lost before it could be examined. We give the number with that caveat rather than omit it.

On search alone, the full system found a passage from a correct treatise for twenty-six of the twenty-seven answerable questions and outperformed plain keyword search on every measure. The margin was modest for the ordinary farming questions, which already use the texts' own words, and large for the disease questions, which do not: the bridge earns its place precisely where the vocabularies diverge.

## 8. Discussion: what the boundary means

It would be easy to read the system's refusal of most modern disease questions as failure. We read it as a measurement.

The classical texts and modern pathology partition the same plant along different axes. The texts ask *why* a tree suffers—an excess of wind, of water, of manure; an insect; a wound; the wrong soil or season—and their remedies follow the cause. Modern diagnosis asks *what* the damage looks like and *which organism* produced it. Neither is mistaken; they are answers to different questions, built for different purposes, in different centuries. A system that mapped one onto the other by force—that returned a "classical remedy for Septoria leaf spot"—would be manufacturing a correspondence the tradition never drew. That is the fabrication the design rule of Section 3 exists to prevent, and it is the fabrication a general-purpose language model produces on request.

Three implications follow for the digital preservation of indigenous knowledge. First, an archive of a knowledge system should be searchable in that system's own terms. The bridge in our system does not translate the texts into modern categories; it translates the modern label into the texts' descriptive register, and it succeeds exactly to the extent that a description can be found on both sides. Second, faithfulness includes knowing where the text stops. A representation that answered every question would be a less faithful representation, not a more useful one. The 92.3 per cent rate at which the system declined the unanswerable is as much a result as the 51.9 per cent at which it answered. Third, the boundary itself is information. That eleven of seventeen common modern leaf diseases have no counterpart in six centuries of Sanskrit agronomy tells a historian of science something about what those authors could observe and chose to record; it tells an extension worker where the tradition can help and where it cannot; and it tells a system builder what not to promise.

For the wider argument of this volume—that indigenous knowledge is to be revitalised, not merely archived—the chapter's contribution is a working example of what revitalisation with integrity looks like when the tool is artificial intelligence. The knowledge is made reachable from a photograph; every use of it is traceable to a verse; and the tradition is represented with its silences intact.

## 9. Limitations and future work

The test questions and their labels were written by the authors and await ratification by an agronomist and a Sanskrit scholar; all numbers are preliminary until then. Forty questions is a small set, and several per-category figures rest on very few cases. The corpus holds English translations, not the Sanskrit originals, and one scholarly introduction (to *Upavanavinoda*) is not yet digitised. "Correct" throughout means faithful to the text; whether the remedies work is a separate question that no one has yet put to the field. One exact-citation figure is unexplained, as noted. And the vision model recognises only the diseases it was trained on, so the disease half of the system is bounded by its training data in a way the corpus is not.

Three directions follow. Expert-built questions with passage-level labels would make the evaluation defensible and allow recall to be measured. Letting the farmer describe what they see, in their own words and language, alongside the photograph, would bypass the disease-label vocabulary altogether and reach the texts on the terms they answer best. And a soil advisory path—the soil model's readings map directly onto the corpus's strongest ground, soil preparation and water—has been built but not yet evaluated, and may prove the more useful half of the system.

## 10. Conclusion

We built a system that reaches six classical Indian agricultural treatises from a photograph, and that either cites the verse it relies on or says the texts are silent. Across every evaluation it invented nothing. Its answers, where it gave them, were traceable to a page; its refusals, where it gave them, were mostly correct. The measurement that mattered most was not an accuracy but a boundary: the classical texts organise plant disease by cause and describe it by the plant's condition, modern diagnosis organises it by appearance and organism, and for most modern leaf diseases the two do not meet. Adding the one classical work devoted to plant disease confirmed rather than closed the gap.

Faithful digitisation, we conclude, means preserving a knowledge system's own categories and being honest about their limits. A tool that does both can make indigenous knowledge usable without rewriting it. A tool that does neither can produce, at scale and with great fluency, a tradition that never existed.

## Acknowledgements

We thank Dr. Sunita T. Pandey for guidance on the symptom-based organisation of the treatises and on *kunapajala*, and the Asian Agri-History Foundation, whose translations made the corpus possible.

## References

Bhat, M. R. (Trans.). (1981). *Varahamihira's Brhat Samhita, with English translation, exhaustive notes and literary comments* (Part 1). Motilal Banarsidass. *[verify year of the Part 1 printing used]*

Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. In *Proceedings of the 32nd International ACM SIGIR Conference* (pp. 758–759). ACM. https://doi.org/10.1145/1571941.1572114

Grattafiori, A., Dubey, A., Jauhri, A., et al. (2024). *The Llama 3 herd of models* (arXiv:2407.21783). arXiv. https://doi.org/10.48550/arXiv.2407.21783

Hughes, D. P., & Salathé, M. (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics* (arXiv:1511.08060). arXiv. https://doi.org/10.48550/arXiv.1511.08060

Jugnu, S. (Ed. & Trans.). (2013). *Kashyapiya Krishi-Paddhati* [Kashyapiyakrishisukti, with Hindi commentary]. Chowkhamba. *[verify imprint (Chowkhamba Sanskrit Series / Surbharati) and place]*

Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E., Bang, Y. J., Madotto, A., & Fung, P. (2023). Survey of hallucination in natural language generation. *ACM Computing Surveys, 55*(12), Article 248. https://doi.org/10.1145/3571730

Majumdar, G. P. (Ed. & Trans.). (1935). *Upavana-Vinoda: A Sanskrit treatise on arbori-horticulture* (Indian Positive Sciences Series No. 1). Indian Research Institute.

Majumdar, G. P., & Banerji, S. C. (Eds. & Trans.). (1960). *Krishi-Parashara* (Bibliotheca Indica, Work No. 285). The Asiatic Society. *[verify publisher line]*

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. In *Advances in Neural Information Processing Systems 33* (pp. 9459–9474).

Nene, Y. L. (2006). Kunapajala – A liquid organic manure of antiquity. *Asian Agri-History, 10*(4), 315–321. *[verify volume and pages]*

NITI Aayog. (2026). *Empowering farmers: Natural farming training toolkit and best practices guide*. Government of India. ISBN 978-81-991080-0-4.

Pandey, A., & Jain, K. (2022). A robust deep attention dense convolutional neural network for plant leaf disease identification and classification from smart phone captured real world images. *Ecological Informatics, 70*, Article 101725. https://doi.org/10.1016/j.ecoinf.2022.101725

Petchiammal, A., Kiruba, B., Murugan, D., & Arjunan, P. (2022). *Paddy Doctor: A visual image dataset for automated paddy disease classification and benchmarking* (arXiv:2205.11108). arXiv. *[verify author list]*

Rananavare, S., & Chitnis, P. (2024). Technology from traditional knowledge: Vrikshayurveda-based expert system for diagnosis and management of plant diseases. *Journal of Ayurveda and Integrative Medicine, 15*(1). https://pmc.ncbi.nlm.nih.gov/articles/PMC10825595/ *[verify authors, issue, DOI]*

Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval, 3*(4), 333–389. https://doi.org/10.1561/1500000019

Sadhale, N. (Trans.). (1996). *Surapala's Vrikshayurveda (The science of plant life by Surapala)* (Agri-History Bulletin No. 1). Asian Agri-History Foundation.

Sadhale, N. (Trans.). (2004). *Vishvavallabha (Dear to the world: The science of plant life)* (Agri-History Bulletin No. 5). Asian Agri-History Foundation.

Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. In *Proceedings of the IEEE International Conference on Computer Vision* (pp. 618–626). https://doi.org/10.1109/ICCV.2017.74

Singh, D., Jain, N., Jain, P., Kayal, P., Kumawat, S., & Batra, N. (2020). PlantDoc: A dataset for visual plant disease detection. In *Proceedings of the 7th ACM IKDD CoDS and 25th COMAD* (pp. 249–253). https://doi.org/10.1145/3371158.3371196

Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. In *Proceedings of the 36th International Conference on Machine Learning* (PMLR 97, pp. 6105–6114).

Xiao, S., Liu, Z., Zhang, P., & Muennighoff, N. (2023). *C-Pack: Packaged resources to advance general Chinese embedding* (arXiv:2309.07597). arXiv. https://doi.org/10.48550/arXiv.2309.07597
