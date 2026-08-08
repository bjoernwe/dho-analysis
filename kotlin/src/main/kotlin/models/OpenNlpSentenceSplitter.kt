package models

import opennlp.tools.sentdetect.SentenceDetectorME
import opennlp.tools.sentdetect.SentenceModel
import java.nio.file.Path
import kotlin.io.path.Path
import kotlin.io.path.inputStream

class OpenNlpSentenceSplitter(modelPath: Path = OPENNLP_MODEL_PATH) : SentenceSplitter {

    private val detector = SentenceDetectorME(modelPath.inputStream().use { SentenceModel(it) })

    override fun split(text: String): List<String> =
        if (text.isBlank()) emptyList() else detector.sentDetect(text).toList()
}
