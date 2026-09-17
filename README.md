# ICPHS-2027

Scripts and results for a paper to be submitted to ICPHS 2027.

The paper investigates speaker idiosyncrasy in a Persian corpus of 30 speakers performing 7 tasks each, comparing how consistently a speaker's peripherality (distance from the population) shows up across different acoustic representations: hand-crafted rhythm features, eGeMAPS, and wav2vec2 SSL embeddings. An initial finding is that rhythm-based and eGeMAPS-based peripherality rankings are not significantly correlated across speakers, motivated by the single-speaker observation that speaker P21 is a strong outlier in rhythm space but far less so in eGeMAPS/wav2vec2 space. The planned next step is to track this effect across wav2vec2's transformer layers to test whether speaker individuality is progressively displaced by phonetic/linguistic content as representations become more abstract.
