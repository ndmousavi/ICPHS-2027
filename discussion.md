
me:  

1) was actually what comes directly to mind, and 2) is an idea that makes sense i think. 3) i don't find interesting, 4) is somewhat interesting but I think too complicated, 5) is always a good idea but probably too costly?

me: what do you reckon would be needed to convince reviewers for a phonetic conference: icphs?

For ICPhS specifically, reviewers will be phoneticians first — so the bar is different from a speech tech venue. A few things that would likely matter:

The subspace estimation step is the weak point. Fitting a speaker classifier on 30 speakers × 7 tasks = 210 recordings to define a 768-dimensional subspace is statistically shaky. Reviewers will ask whether the estimated subspace is reliable enough to draw conclusions from. You'd need to justify the method carefully or use cross-validation.

Phonetic interpretability of the subspace decomposition. Phonetics reviewers won't accept "speaker subspace" as a black box — they'll want to know what acoustic properties those directions correspond to. Ideally you'd show that the speaker subspace correlates with something interpretable (e.g., F0 range, voice quality), otherwise it stays an opaque ML construct.

The single-speaker case still needs justification. Even with the subspace analysis, reviewers may push back on why P21 is the right case to examine this question rather than it being a post-hoc rationalization of an outlier. Having population-level results (idea 1) alongside this would insulate the paper considerably.

What would likely be sufficient: the subspace projection analysis paired with a correlation between the speaker subspace dimensions and interpretable acoustic features (e.g., eGeMAPS dimensions), so reviewers can see a phonetically grounded explanation rather than just a geometric one. That would make idea 4 feel like a phonetic finding rather than an ML analysis that happens to use speech data.


me: explain 4

Liu et al. (2023) showed that in wav2vec 2.0's representational space, speaker identity and phonetic content tend to be encoded in roughly orthogonal directions — meaning the dimensions that best predict who is speaking are largely separate from the dimensions that best predict what is being said.

The idea is to exploit that structure to diagnose why P21 appears less peripheral in the wav2vec PCA (Figure 1c) compared to the rhythm features (Figure 1a).

Concretely:

1. Estimate the subspaces — use a labeled subset of the corpus to fit a linear classifier predicting speaker identity and a separate one predicting phoneme labels. The weight vectors (or the directions of maximum variance conditioned on speaker/phoneme) define two subspaces within the 768-dimensional wav2vec space.
2. Project P21's recordings onto each subspace — compute P21's distance from the population in the speaker subspace separately from the phonetic subspace.
3. Interpret the result — if P21 remains peripheral in the speaker subspace but not in the phonetic subspace, that confirms the speaker-specific signal is present in the SSL representation but gets visually "hidden" by the PCA because the overall 768 dimensions mix in a lot of phonetic variance that dilutes it. If P21 is not peripheral in either subspace, it suggests the SSL model genuinely encodes this speaker's individuality differently than hand-crafted features do.

The payoff for the paper is that it would move from the descriptive observation ("P21 looks less extreme in wav2vec PCA") to a mechanistic explanation of where speaker individuality goes in learned representations.


me:  last one: ❯ it's technically challenging to do 4, no? could there be a simpler alternative?

Yes, quite technically demanding — you need phoneme labels for subspace estimation, which means forced alignment on Persian, and the small dataset makes the subspace estimates unreliable anyway.

A simpler alternative that gets at the same underlying question: compare P21's extremeness across individual wav2vec layers (rather than decomposing subspaces). You just extract representations from each of the 12 transformer layers separately, run the same PCA + distance analysis you already have, and plot how P21's separation from the population changes layer by layer.

This is compelling because:
- Earlier layers are known to encode more acoustic/speaker information, later layers more linguistic content
- If P21's peripherality decreases in later layers, that's interpretable evidence that the speaker-specific signal gets progressively overwritten by phonetic encoding — essentially the same story as idea 4 but arrived at without needing phoneme labels or classifier-based subspace estimation
- The method is identical to what you already do, just applied 12 more times

The phonetic argument would be: speaker individuality is present in the SSL representation but is deprioritized as the model builds increasingly abstract linguistic structure — which is actually a more interesting phonetic claim than the subspace version, and reviewers can follow the logic without needing ML background.

The main cost is it's correlational rather than mechanistic — you'd infer the speaker/phonetic tradeoff from layer depth rather than measuring it directly. But for ICPhS that's probably a feature, not a bug.


me: this is what claude thinks of my idea ;)

 Yes, the logic is sound and it's a clean design. A few thoughts on where it could go wrong:

 The "hope for a linear trend" is the risk. wav2vec layers are not monotonically acoustic-to-linguistic — empirically, speaker information tends to peak in intermediate layers (around layers 1-6) before declining, rather than decreasing smoothly from layer 1. You might get a non-monotonic curve, which is still interpretable but harder to argue cleanly.

 You need to fix which speakers count as outliers. If you redefine "top 10 most peripheral" per layer, the set shifts across layers and the comparison becomes muddled. The cleaner approach is to identify the 10 most peripheral speakers using the rhythm or eGeMAPS representation (which is the paper's starting point), then track those same speakers across wav2vec layers — that way the outlier set is fixed and the layer analysis is purely about what happens to them inside the SSL model.

 30 speakers, top 10 is a third of the corpus — reviewers might challenge the threshold. Framing it as "the upper tercile" or justifying the cutoff with a gap in the distance distribution would help.

 What would make this convincing at ICPhS: if the trend holds across most of the 10 speakers individually (not just in the mean), because a mean trend driven by 2-3 speakers would be fragile. Showing individual trajectories alongside the group trend would preempt that critique.

 Overall though — yes, this is feasible, phonetically motivated, and extensible enough for 8 pages.


me:  one more idea:
❯ we also have the speaker task, perhaps this can be contrasted with the speaker idiosyncrasies

That's a really strong addition and fits the corpus design perfectly.

The contrast you'd get is: within-speaker dispersion (how spread out a speaker's 7 task recordings are — task-induced variability) versus between-speaker distance (how far a speaker sits from the population centroid — idiosyncrasy). You can track both across wav2vec layers.

The prediction would be that in later layers within-speaker task dispersion increases (the model becomes more sensitive to register and speaking style) while between-speaker distances decrease (speaker idiosyncrasy is deprioritized). That's a clean, falsifiable double dissociation.

This also reframes P21 elegantly. Looking at Figure 1a, P21's 7 task recordings cluster tightly together despite being far from the population — meaning low within-speaker task variability but high idiosyncrasy. If later wav2vec layers compress between-speaker distances while expanding within-speaker task spread, it would explain precisely why P21 becomes less peripheral in the SSL representation: the model trades speaker identity for task sensitivity.

For ICPhS this is compelling because it connects directly to register and style variation — classic phonetics territory — rather than framing the whole paper as an ML analysis. The corpus's 7-task design, which was incidental to the original rhythm study, becomes a genuine analytical asset here.

Combined with the layer-wise outlier tracking, you'd have a coherent 8-page argument: speaker idiosyncrasy is real, persists in hand-crafted features, but is progressively displaced by task-related variation as SSL representations become more abstract.

what you have in the paper' figures is one datapoint per sample, right? I would continue for now to use the data utterance-global and not focus on single phonemes (if i understood you correctly). And yes, the peripherality measure is key point, i would have thought some kind of distance in the 2 d space with respect to the center point

me: sure.  i guess i could extract the layer embeddings with nkululeko. as the facebook-.300 model has 24 layers, i would suggest to do this in 3-layer steps, i.e. 8 embedding sets per utterance

That works, but two things to flag before you run it.

First, "facebook-.300" is nkululeko's default alias for `wav2vec2` — it resolves to `facebook/wav2vec2-large-robust-ft-swbd-300h`, a 24-layer *ASR-fine-tuned* model (fine-tuned on Switchboard, English conversational telephone speech). That's a different model from the one already behind the Fig. 1c wav2vec2 embeddings, which are `facebook/wav2vec2-base` (12 layers, purely self-supervised, no fine-tuning, 768-dim). Switching models for the layer-wise analysis means:
- you lose direct comparability with the existing Fig. 1c point — you'd be reporting "layer trajectory in model B" next to "single point from model A," which reviewers may flag
- the fine-tuning objective is English ASR (CTC), which pushes later layers toward English phonetic/orthographic targets — for a Persian corpus that's an extra confound on top of the SSL-vs-linguistic story you're trying to tell. A plain self-supervised model (`wav2vec2-large-xlsr-53`, also 24 layers, 1024-dim, multilingual pretraining) would isolate the effect you actually want to claim. `wav2vec2-base` itself only has 12 layers, so if you want to stay consistent with what's already extracted, that's the simpler path and skips the model-choice question entirely — 12 layers in 1-layer or 2-layer steps instead of 3-layer steps on a 24-layer model.

Second, nkululeko's layer-selection config is easy to get backwards. In `feats_wav2vec2.py`, `FEATS.wav2vec2.layer` is counted **from the top down**: the model is truncated to `num_hidden_layers - wav2vec2.layer` layers, and you get the output of that truncated stack. So `wav2vec2.layer = 0` (default) means the full model / last layer, and to get a specific layer L counting from the bottom on a 24-layer model, set `wav2vec2.layer = 24 - L`. For 3-layer steps (L = 3, 6, 9, 12, 15, 18, 21, 24) that's `wav2vec2.layer` values `21, 18, 15, 12, 9, 6, 3, 0` — worth double-checking against the HF config's `num_hidden_layers` for whichever model you land on, since the arithmetic is model-specific.

One more practical blocker: the audio paths in the existing dataframes point to `/Users/neda/Documents/Corpus/Pertsch/persian/...`, which isn't reachable from this machine. Extraction needs either that corpus mounted/copied here or to run on the machine where it lives — worth sorting out before writing the 8 configs.

me: first there's a new nkululeko version: 1.11.0 you got that? second: yes, using the xlsr-53 model is a good idea

Checked — 1.11.0 is real, released 2026-09-17, PyPI's current `info.version`. Its one changelog entry ("introducing LODO — leave-one-database-out for multidb") doesn't touch feature extraction. I pulled the actual 1.11.0 wheel and diffed `feats_wav2vec2.py` against what I'd read before: byte-identical. So the layer-indexing gotcha stands as described — `FEATS.wav2vec2.layer` truncates `num_hidden_layers - layer` from the top, so for a bottom-up layer L you set `wav2vec2.layer = num_hidden_layers - L`.

And confirmed `facebook/wav2vec2-large-xlsr-53`'s HF config: `num_hidden_layers = 24`, `hidden_size = 1024` — matches your "24 layers" and it's self-supervised only (CommonVoice + BABEL + multilingual LibriSpeech pretraining, no ASR fine-tuning), so it sidesteps the Switchboard-ASR confound from the earlier "facebook-.300" default. For 3-layer steps (L = 3, 6, 9, 12, 15, 18, 21, 24) on this model that's `wav2vec2.layer` = 21, 18, 15, 12, 9, 6, 3, 0.

Still outstanding before this can actually run: audio access (the `/Users/neda/...` path issue above), and whether to keep `wav2vec2-base`'s existing Fig. 1c point as a separate reference or drop it now that the layer-wise analysis uses a different model.

me: [data now available locally under data/persian] don't change the ipynb. We want nkululeko configurations for all layers in scripts, as well as a bash script to call them sequentially on a free GPU. don't run them yet! i'll do this in a tmux shell.

me: wait. the scripts in nkulukeo_01 are most probably outdated

Fair — cross-checked everything against the actual released 1.11.0 wheel from PyPI instead of the local checkout (both turned out identical for the modules that matter here).

me: what's actually the best nkululeko module to extract the layer embeddings? i guess the predict module, right? [...] we want as result a csv file per layer with all pre-existing columns + the 768 cols for the embeddings

Yes — `nkululeko.predict --list ... --type feats --model wav2vec2-large-xlsr-53 --config <layer ini>` is the right tool, not `explore`. `--list` mode preserves the input CSV's existing columns and appends `feat_0..feat_N` for the embedding, no `[DATA]`/`[MODEL]`/`[EXPL]` gymnastics needed — each layer config is just a one-line `[FEATS] wav2vec2.layer = N`. Two corrections along the way: (1) xlsr-53 is 1024-dim, not 768 (768 was the old wav2vec2-base run) — `feat_0..feat_1023` per layer. (2) Neither `predict` nor `explore`/`train`'s wav2vec2 extractor resamples audio — `explore` hard-asserts 16kHz and would crash outright, `predict` reads at native rate with no check and would silently produce embeddings computed on the wrong time axis. The corpus is natively 44.1kHz, so added a standalone `resample_to_16k.py` (plain torchaudio, no nkululeko dependency) as an explicit pre-step producing `data/persian_16k/` + `persian_nkululeko_data_16k.csv`, which `run_wav2vec2_layers.sh` now checks for before running any layer.

me: the paper that is the basis for this experiment is now in `paper/`. Neda asks: could you please have a look at the method of measuring variability in the paper, if it is a good way to approach it or if you have another suggestion, or aspects to consider?

Read `paper/icphs2027-template.tex` (Method §2.3, "Analysis"). Two measures per speaker, both computed from PCA scores (features z-standardized, PCs retained at 80% variance, separately per representation):
- **Within-speaker task dispersion**: plain Euclidean distance of each of a speaker's 7 task recordings from that speaker's own centroid, averaged over the 7.
- **Between-speaker peripherality**: PC scores first centered per task on the 30-speaker task mean (removing shared task effects), then Mahalanobis distance from the task-specific population center using one pooled covariance matrix over all 210 task-centered points, averaged over the speaker's 7 tasks.
- Cross-representation correspondence via Spearman rho, permutation test (10k, speaker-label permutation) for significance, bootstrap (10k, speaker-level) for CIs. Plus an RSA layer: task-centered PC centroids per speaker → 30×30 Euclidean RDM → Spearman between representations' RDMs, Mantel-style permutation.

Overall verdict: sound and appropriately rigorous — rank-based comparison (not raw distance magnitudes) is the right call given wildly different PCA dimensionality across representations (4 dims for rhythm, 11 for eGeMAPS, presumably much more for any 1024-dim wav2vec2 layer), and permutation + bootstrap over Spearman is standard, defensible practice for n=30. The task-centering step before Mahalanobis is the strongest part of the design — it correctly separates "this speaker is unusual *given the task*" from "this task pulls everyone in some direction," which is exactly what the P21 motivating case needs.

Things worth considering:
1. **Euclidean (within) vs. Mahalanobis (between) asymmetry isn't explained in the text.** It's actually well-motivated — 7 points/speaker can't support a per-speaker covariance estimate, so Euclidean is the fallback, while 210 pooled points can support one shared covariance matrix for the between-speaker measure — but a phonetics reviewer will likely ask "why two different distance metrics for two conceptually parallel measures?" One sentence stating the sample-size rationale would preempt that.
2. **Covariance conditioning once wav2vec2 layers are added.** The rhythm/eGeMAPS PC spaces are tiny (4–11 dims) relative to n=210, so the pooled covariance for Mahalanobis is well-estimated. An 1024-dim wav2vec2 layer's PCA could plausibly need many more components to hit 80% variance (correlated dims can keep this low, but it should be checked, not assumed) — if retained dimensionality gets anywhere near n/10, the covariance matrix starts to get unstable, and Mahalanobis distances become sensitive to noise in the smallest eigenvalues. Worth reporting the retained-PC count per layer, and considering a shrinkage covariance estimator (Ledoit-Wolf) for the higher-dimensional spaces as a robustness check.
3. **Averaging 7 per-task Mahalanobis distances into one peripherality score can hide within-speaker inconsistency** — a speaker peripheral on 1 of 7 tasks and unremarkable on the rest averages the same as one mildly peripheral on all 7. Same concern as the earlier layer-tracking discussion about individual trajectories vs. the group mean: worth at least a supplementary plot of each speaker's 7 per-task distances (or their spread) alongside the mean, so a reader can see whether peripherality is a stable trait or driven by one recording.
4. **Multiple comparisons once wav2vec2 layers enter the picture.** Right now it's one pair (rhythm–eGeMAPS) for each of 3 measures — fine as is. But with 8 layers + 2 hand-crafted representations, a fully-crossed comparison is 45 pairs × 3 measures; even the layer-vs-both-hand-crafted-sets-only subset (16 pairs) needs a stated comparison scheme and, given many correlations will be tested, a multiple-comparisons correction (FDR is the natural choice, doesn't require independence) — this should be decided before the Learned Representations results are written, since that section is currently still empty.
5. **A complementary global (not per-speaker) metric worth considering**: a variance-ratio / F-ratio style between-vs-within-speaker separability index per representation (à la the forensic-phonetics speaker-discriminability literature, e.g. Rose/Kinoshita), computed once per representation rather than per speaker. This would directly answer "how separable are speakers *in this space overall*" as a single number per representation, complementing the per-speaker peripherality ranks and connecting the paper to an established phonetics literature on exactly this question beyond Nolan's speaker-space framing already cited.
6. **Robustness to the 80% variance cutoff.** Since retained dimensionality differs so much by representation, it's worth a quick sensitivity check (e.g., also reporting results at 90%, or at a fixed number of components) to confirm the rank correlations aren't an artifact of where that cutoff happens to fall for each representation.

None of these are fatal — the core design (task-centered Mahalanobis + rank correlation + permutation/bootstrap) is a good, defensible approach. Items 1 and 4 are the ones I'd address before submission; 2, 3, 5, 6 are strengthen-if-time.

