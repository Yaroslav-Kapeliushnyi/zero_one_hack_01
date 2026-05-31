# Relevant Papers — Industrial Process Sequence Modeling

> Compiled for Zero One Hack 2026 Industrial Track
> 36-hour hackathon: next-step prediction, sequence completion, anomaly detection on semiconductor manufacturing sequences

---

## Topic 1: Process Mining — Next Activity Prediction (Core Task)

**1. ProcessTransformer: Predictive Business Process Monitoring with Transformer Network**
- Bukhsh, Saeed, Dijkman — 2021 — ICPM / arXiv:2104.00721
- Proposes a Transformer with self-attention for next-activity prediction on event logs, treating process traces exactly like token sequences. Achieves >80% accuracy on 9 real-world logs, beating LSTM baselines by capturing long-range dependencies.
- **Key insight:** Process steps as vocabulary tokens, prefix as input, next step as classification target. Direct framing for our semiconductor sequences.

**2. POP-ON: Prediction of Process Using One-Way Language Model Based on NLP Approach**
- Moon, Park, Jeong — Applied Sciences 2021 — doi:10.3390/app11020864
- Adapts GPT-2 as a causal language model for next-event prediction in manufacturing MES/ERP logs. Treats each process step name as a "word" and fine-tunes GPT-2 with process-specific preprocessing.
- **Key insight:** Our problem IS a language modeling problem. A decoder-only (GPT-style) transformer with step names as tokens. Validated specifically on manufacturing data.

**3. An Innovative Next Activity Prediction Approach Using Process Entropy and DAW-Transformer (2025)**
- arXiv:2502.10573 — 2025
- DAW-Transformer (Dynamic Attribute-Aware Transformer) with dynamic windowing to handle all historical events rather than fixed-length windows. Uses process entropy to guide model complexity selection.
- **Key insight:** For long sequences (~150 steps), a dynamic window is superior to a fixed prefix window.

---

## Topic 2: Full Suffix / Sequence Completion

**4. ASTON: Encoder-Decoder Model for Suffix Prediction in Predictive Monitoring**
- Rama-Maneiro et al. — arXiv:2211.16106 — 2022
- Encoder-decoder (LSTM encoder + attention + LSTM decoder) specifically designed for predicting the full remaining sequence (suffix). Uses beam search with length normalization. Tested against 6 baselines on 12 event logs.
- **Key insight:** Beam search prevents greedy error cascade on long completions — critical for 115–150 step sequences. Decouples prefix encoding from suffix generation.

**5. An In-Context Foundation Model for Predictive Process Monitoring on Event Logs**
- Berti et al. — RWTH Aachen — 2024
- Trains a single foundation model across heterogeneous event logs; adapts to new logs via few-shot in-context examples without retraining. Mixture-of-experts prefix encoder + prototype classifier.
- **Key insight:** If data is scarce, few-shot in-context adaptation works. Prototype-based head is interpretable.

---

## Topic 3: Anomaly Detection in Process Sequences

**6. LogGPT: Log Anomaly Detection via GPT**
- arXiv:2309.14482 — 2023
- Trains GPT for next-log-entry prediction, then uses top-K prediction lists as anomaly signal: if the actual observed step is NOT in the top-K predictions, it's anomalous. Adds RL fine-tuning to sharpen anomaly sensitivity.
- **Key insight:** Anomaly detection for free from the next-step predictor. If actual next step has low predicted probability → anomaly. One model covers Tasks 1 and 3.

**7. Evaluating the Ability of LLMs to Solve Semantics-Aware Process Mining Tasks**
- Van der Aa et al. — arXiv:2407.02310 — 2024
- Benchmarks LLMs on sequence validity, activity ordering validity, and next-activity prediction. Out-of-the-box LLMs fail; fine-tuned LLMs consistently outperform smaller encoder models on all three tasks.
- **Key insight:** Fine-tuning a small LLM on our sequences enables next-step prediction AND anomaly detection (sequence validity) in a single model.

**8. TRACE-GPT: Generative Pre-Training of Time-Series Data for Unsupervised Fault Detection in Semiconductor Manufacturing**
- Lee, Choi, Kim — arXiv:2309.11427 — 2023 — Samsung
- GPT-style causal pretraining on semiconductor CVD equipment sequences. Anomaly score = next-value prediction cross-entropy exceeds threshold. Achieves F1=1.000 on CVD dataset, near-SOTA with zero labels.
- **Key insight:** Directly from semiconductor manufacturing domain. Same GPT objective for both generation and anomaly scoring. Cross-entropy as anomaly score — no separate model needed.

**9. Anomaly Detection for Service-Oriented Business Processes Using Conformance Analysis**
- MDPI Algorithms 2022 — doi:10.3390/a15080257
- Discovers a process model from successful traces, then uses alignment-based conformance checking to score new traces. "Conformance lifeline" tracks fitness as a partial trace unfolds — enables early anomaly detection mid-sequence.
- **Key insight:** Symbolic complement. PM4Py implements this out of the box — fast interpretable baseline for anomaly detection.

---

## Topic 4: Transformer Anomaly Detection in Manufacturing

**10. Knowledge-Augmented Anomaly Detection in Small Lot Production for Semantic Temporal Process Data**
- TUM Munich — ICRA/Manuscript 2023
- Transformer-based autoencoder (reconstruction network) on robotic manufacturing process data. Trains only on nominal executions; anomaly score = reconstruction error. 94.1% overall accuracy, outperforms LSTM-AE and CNN-AE.
- **Key insight:** Reconstruction-based approach: train on valid sequences only, flag anything the model can't reconstruct. Useful fallback if we don't know what anomalies look like.

---

## Topic 5: Manufacturing Error Sequence Analysis

**11. Identifying Cause-and-Effect Relationships of Manufacturing Errors Using Seq2Seq Learning**
- Scientific Reports (Nature) — 2022 — doi:10.1038/s41598-022-26534-y — Volkswagen
- Benchmarks LSTM, GRU, Transformer on car-body production multi-station sequences. Transformer outperforms recurrent models especially for longer prediction horizons. 71.68% of real sequences contain source or knock-on errors.
- **Key insight:** Confirms Transformer > LSTM for long manufacturing sequences. Error propagation across steps is the dominant failure mode — relevant to our anomaly detection design.

---

## Topic 6: Declarative Process Models & Simple Baselines

**12. Declarative Process Mining (Survey)**
- Di Ciccio, Montali — Springer Handbook Chapter — 2022
- Comprehensive review of Declare-based process mining: discovering LTL constraints from event logs, monitoring running traces for violations. All techniques reduce to automata over vocabulary of activity names.
- **Key insight:** PM4Py's Declare miner can auto-discover constraints ("step A must precede step B") from training sequences and flag violations in new sequences — fast interpretable baseline.

**13. A Framework for Streaming Event-Log Prediction in Business Processes**
- arXiv:2412.16032 — 2024
- Compares n-gram, prefix-tree, LSTM, and ensemble methods for next-activity prediction. Simple n-gram ensembles approach LSTM accuracy, beat LSTMs early when data is sparse.
- **Key insight:** N-gram / prefix-tree = zero-training-time baseline that performs surprisingly well. Start there, layer transformer on top. Soft-voting ensemble can beat either alone.

---

## Recommended Hackathon Stack (from papers)

| Task | Approach |
|------|----------|
| Next-step prediction | GPT-2-scale causal transformer (papers 1, 2, 3) |
| Full sequence completion | Encoder-decoder + beam search (paper 4) OR autoregressive decode |
| Anomaly detection (neural) | Top-K prediction confidence from same model (papers 6, 8) |
| Anomaly detection (symbolic) | PM4Py Declare miner + conformance check (paper 12) |
| Fast baseline (day 1) | N-gram soft-voting ensemble (paper 13) |

**One-model strategy:** Train a GPT-2-scale causal transformer on sequences (step names as tokens).
- Task 1: argmax of next-token distribution = predicted next step
- Task 2: autoregressive decode until END token = completed sequence
- Task 3: cross-entropy / top-K miss = anomaly score
