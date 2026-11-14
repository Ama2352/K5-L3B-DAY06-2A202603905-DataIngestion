# Fixed benchmark for the T5 PDF corpus

20 English questions grounded in the JMLR 2020 T5 paper, physical PDF pages 4-40. Split fixed before retrieval: 15 dev and 5 test (q04, q08, q12, q16, q20). Scope filters are explicitly in questions q06/q11 (Setup, dev) and q16/q20 (Experiments, test). section values come from actual main headings; mixed page 10 belongs to both sections. No filter is inferred from a gold answer/page after retrieval.

Questions include direct facts, paraphrases and conditions/exceptions. Evidence is an exact normalized source excerpt, verified before the first PDF experiment. Exact repeated evidence locations are listed as alternative_gold where present. Dev/test share one paper and some topics, so this is not a document-disjoint test; only five test questions give limited evidence.

Selection rule fixed before retrieval: dev Evidence Hit@3, then Hit@1, fewer chunks, strategy name. The pipeline persists selection before evaluating test. Gold and configs are held fixed across A/B/C; no tuning or gold changes after viewing PDF test results.

Coherence locations are predetermined physical pages 4,8,12,16,20,24,28,32,36,40, offset at each normalized page midpoint. Pick the earliest-start chunk covering the location when overlap yields multiple candidates. The sample spans the selected range but can land in figures or tables; report this limitation.

Hit requires matching source/page and sufficient gold evidence inside one chunk. Grounding is a separate qualitative judgement of the full top-3; Hit@3 is not an LLM-answer accuracy score. Annotation assistance and subsequent student review status are recorded in results/review_labels.json.

Run `python -m starter.src.run validate` from the repository root to verify the PDF hash, extracted pages, metadata and evidence. The prior Python Tutorial corpus/annotations are available only in earlier Git history; none are reused as PDF results.
