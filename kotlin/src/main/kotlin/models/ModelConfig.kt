package models

import kotlin.io.path.Path
import java.nio.file.Path as JPath

/**
 * Configuration for model paths. The OpenNLP model path can be overridden via
 * the system property `opennlp.model.path` or the environment variable
 * `OPENNLP_MODEL_PATH`. If neither is set, the bundled default path is used.
 */
val OPENNLP_MODEL_PATH: JPath
    get() {
        System.getProperty("opennlp.model.path")?.let { if (it.isNotBlank()) return Path(it) }
        System.getenv("OPENNLP_MODEL_PATH")?.let { if (it.isNotBlank()) return Path(it) }
        return Path("models/OpenNLP/opennlp-en-ud-ewt-sentence-1.3-2.5.4.bin")
    }
