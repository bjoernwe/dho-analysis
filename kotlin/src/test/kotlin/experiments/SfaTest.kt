package experiments

import algorithms.calcPca
import algorithms.calcSfa
import algorithms.scoreLabelVector
import data.Sentence
import models.ZeroShotClassifier
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import kotlin.random.Random

class SfaTest {

    @Test
    fun `composed label loadings reproduce the sentence projections`() {
        val labels = (1..8).map { "label$it" }
        val sentences = (0 until 120).map { i ->
            Sentence(msgId = (i / 3).toLong(), date = "2020-01-%02d 12:00:00".format(i / 3 % 28 + 1), sentence = "s$i")
        }
        val model = SyntheticClassifier(labels)

        val pca = sentences.calcPca(labels = labels, nComponents = 4, model = model)
        val sfa = pca.calcSfa(nSlowFeatures = 3)

        // Independently rebuild the raw label-score matrix and push it through the composed
        // loadings; it must match what calcSfa derived via the PCA scores.
        for (i in sentences.indices) {
            val x = DoubleArray(labels.size) { l -> model.scoreSingle(sentences[i].sentence, labels[l]).toDouble() }
            val viaLoadings = sfa.scoreLabelVector(x)
            for (m in 0 until sfa.nSlowFeatures) {
                assertEquals(sfa.projected[i][m], viaLoadings[m], 1e-9)
            }
        }

        assertEquals(labels.size, sfa.loadings[0].size)
        assertEquals(3, sfa.slowness.size)
    }

    @Test
    fun `messages are aggregated and ordered chronologically`() {
        val labels = (1..4).map { "label$it" }
        // Twelve messages of three sentences each, stored newest first: the source jsonl is grouped
        // by thread rather than sorted by date, so calcSfa has to reorder them itself.
        val messageCount = 12
        val sentences = (messageCount downTo 1).flatMap { m ->
            (1..3).map { s -> Sentence(m.toLong(), "2020-01-%02d 12:00:00".format(m), "m${m}s$s") }
        }

        val sfa = sentences.calcPca(labels = labels, nComponents = 2, model = SyntheticClassifier(labels))
            .calcSfa(nSlowFeatures = 2)

        // One series step per message, oldest first, regardless of the input order.
        val dates = sfa.messages.map { it.date }
        assertEquals(messageCount, dates.size)
        assertEquals(dates.sorted(), dates)
        assertEquals("2020-01-01 12:00:00", dates.first())

        // Sentence rows stay aligned with the input order, which is still newest first.
        assertEquals(sentences.size, sfa.projected.size)
        assertEquals(sentences.map { it.msgId }, sfa.sentences.map { it.msgId })
    }

    // Deterministic stand-in for the ONNX model: every (text, label) pair gets a stable
    // pseudo-random score in [0, 1], so PCA has real structure to work with without any model files.
    private class SyntheticClassifier(private val labels: List<String>) : ZeroShotClassifier {
        override val batchSizeTokenBudget = 1000

        override fun scoreBatch(texts: List<String>, label: String): List<Float> {
            val labelIndex = labels.indexOf(label)
            return texts.map { text ->
                val random = Random(text.hashCode() * 31L + labelIndex)
                // Give the labels different variances so PCA components are well separated.
                (random.nextDouble() * (1.0 + labelIndex) / labels.size).toFloat()
            }
        }

        override fun close() {}
    }
}
