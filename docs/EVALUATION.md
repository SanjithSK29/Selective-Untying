# Evaluation protocol and claim boundaries

## Completed measurements

The main experiment contains 12 paired SFT seeds for each of tied, conflict,
frequency and frequency-matched random selection. Each run evaluates the first
500 qualifying TinyStoriesInstruct validation examples. These are repeated uses
of the same prompts, not 6,000 independent prompts. The random selector uses one
fixed frequency-matched token set across all SFT seeds.

- **All-keyword satisfaction:** fraction of generated responses containing every
  requested word, using case-insensitive whole-word matching.
- **Macro keyword recall:** mean over prompts of the fraction of requested words
  present. This differs from pooling all keyword occurrences when prompt sizes differ.
- **Response NLL:** token-weighted teacher-forced negative log-likelihood of
  reference response tokens, including EOS and excluding prompt/padding targets.

Generation uses greedy decoding, EOS stopping, up to 256 new tokens and a sliding
256-token context. Longer generations may lose the instruction from their context.
Training uses the same pretrained weights, matching minibatch orders for matching
seeds, and zero-initialized offsets. Full untying copies the original output weights.

## Statistical analysis

Pair methods by seed. Report the mean of seed-level differences and a descriptive
Student-t 95% interval with n-1 degrees of freedom. The code in `report_results.py`
uses sample SD, not population SD. The main n is 12; exploratory n is 3. Source
JSON metrics are rounded, so derived intervals are approximate. The analysis is
conditional on one checkpoint, prompt sample, calibration and random selection.

The original project criterion requires a >=2 percentage point mean satisfaction
gain over both frequency and matched random, with no mean NLL degradation versus
tied. The 12-seed results fail all three conditions. The observed paired intervals
also exclude a +2pp satisfaction benefit in this setting. Do not translate that
into universal equivalence or claims about all model scales/tasks. No formal
equivalence test or proof of a zero effect is claimed.

Keyword recall was added during the extension and is a secondary metric. The
12-seed extension followed inspection of the initial results. Describe this
history transparently, and avoid calling the entire analysis preregistered.

## Human evaluation status

**No blinded human coherence or keyword-integration ratings have been completed
or supplied.** No human scores are included in this repository. Missing-word lists
cannot reconstruct full stories and cannot substitute for coherence ratings.
The project therefore makes no demonstrated fluency, naturalness, coherence, or
semantic-instruction-following claim. Lower reference NLL is not a direct measure
of generated-story fluency. The scaling result is described as a response-NLL /
keyword-satisfaction pattern, rather than a proven fluency trade-off.

If ratings are later added, export actual generated texts from saved SFT weights,
select a fixed matched prompt set before inspecting preferences, hide method/seed
labels, randomize presentation order, and use at least two independent raters.
Assess coherence separately from natural keyword integration with anchored scales.
Keep the private randomization key out of the rating sheets, preserve all ratings,
report agreement and uncertainty, and label the analysis exploratory. Do not rate
reference stories as if they were model generations.

## Overlap audit

`audit_overlap.py` reconstructs the exact filtered validation prefix, SFT training
prefix and story-only calibration prefix. It scans the pinned official TinyStories
training split for normalized exact matches. Normalization collapses whitespace,
casefolds text and preserves punctuation. The machine-readable output records
source revisions, scanned counts, matched prompt indices, and lexical metrics on
the remaining subset when available. It does not check near duplicates or semantic
similarity. It cannot recover NLL on a subset from aggregate NLL records.

Overlap is disclosed; historical results are never silently replaced by a cleaned
evaluation. A new clean evaluation must carry its own sample description and results.

## Completed overlap findings

The pinned official pretraining split was fully scanned: 2,119,719 records.
There were 129/500 normalized exact pretraining matches, 0/500 SFT-prefix matches,
and 0/500 calibration-prefix matches. The 500 evaluation stories were unique.
All 48 missing-word/recall records aligned with the reconstructed prompt list.
This supports prompt-list consistency but cannot recover the original source revision.
The post-hoc 371-prompt subset gives conflict satisfaction differences of -0.09pp
versus both frequency and matched random. Their paired 95% intervals are respectively
[-1.34,+1.16]pp and [-1.21,+1.03]pp. Subset NLL is not available. See reports/NONOVERLAP.md.
