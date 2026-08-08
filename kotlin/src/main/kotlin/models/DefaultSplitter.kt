package models

// A lazily-initialized shared default splitter. Constructing the OpenNLP splitter
// is relatively expensive (loads a binary model), so defer construction until
// first use and reuse the instance afterwards.
val defaultSplitter: SentenceSplitter by lazy { OpenNlpSentenceSplitter(OPENNLP_MODEL_PATH) }
