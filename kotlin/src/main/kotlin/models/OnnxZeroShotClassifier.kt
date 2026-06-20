package models

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import ai.djl.util.PairList
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import java.nio.LongBuffer
import java.nio.file.Path
import kotlin.io.path.readText
import kotlin.math.exp

class OnnxZeroShotClassifier(modelDir: Path, modelFile: String = "model.onnx") : ZeroShotClassifier {

    // Padding is required so batchEncode can stack pairs of unequal length into one rectangular tensor.
    private val tokenizer = HuggingFaceTokenizer.newInstance(modelDir, mapOf("padding" to "true"))
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val entailmentIndex: Int
    private val contradictionIndex: Int

    init {
        val modelPath = modelDir.resolve(modelFile).toString()
        session = try {
            val opts = OrtSession.SessionOptions().apply { addCUDA(0) }
            env.createSession(modelPath, opts)
        } catch (e: OrtException) {
            System.err.println("CUDA unavailable (${e.message}), falling back to CPU")
            env.createSession(modelPath, OrtSession.SessionOptions())
        }
        val (e, c) = resolveNliIndices(modelDir)
        entailmentIndex = e
        contradictionIndex = c
    }

    override fun scoreBatch(texts: List<String>, label: String): List<Float> {
        if (texts.isEmpty()) return emptyList()

        val hypothesis = "This example is $label."
        val encodings = tokenizer.batchEncode(PairList(texts, texts.map { hypothesis }))

        val batchSize = encodings.size
        val seqLen = encodings[0].ids.size
        val shape = longArrayOf(batchSize.toLong(), seqLen.toLong())

        val idsBuffer = LongBuffer.allocate(batchSize * seqLen)
        val maskBuffer = LongBuffer.allocate(batchSize * seqLen)
        val useTypeIds = "token_type_ids" in session.inputNames
        val typeBuffer = if (useTypeIds) LongBuffer.allocate(batchSize * seqLen) else null
        for (encoding in encodings) {
            idsBuffer.put(encoding.ids)
            maskBuffer.put(encoding.attentionMask)
            typeBuffer?.put(encoding.typeIds)
        }
        idsBuffer.flip()
        maskBuffer.flip()
        typeBuffer?.flip()

        val inputs = buildMap<String, OnnxTensor> {
            put("input_ids", OnnxTensor.createTensor(env, idsBuffer, shape))
            put("attention_mask", OnnxTensor.createTensor(env, maskBuffer, shape))
            if (typeBuffer != null) put("token_type_ids", OnnxTensor.createTensor(env, typeBuffer, shape))
        }

        return session.run(inputs).use { result ->
            @Suppress("UNCHECKED_CAST")
            val logits = result.get("logits").get().value as Array<FloatArray>
            logits.map { twoWaySoftmax(it[entailmentIndex], it[contradictionIndex]) }
        }
    }

    override fun close() {
        session.close()
        env.close()
    }
}

// HF multi_label=True uses 2-way softmax over entailment vs contradiction (neutral excluded)
private fun twoWaySoftmax(entailment: Float, contradiction: Float): Float {
    val max = maxOf(entailment, contradiction)
    val expE = exp((entailment - max).toDouble())
    val expC = exp((contradiction - max).toDouble())
    return (expE / (expE + expC)).toFloat()
}

private fun resolveNliIndices(modelDir: Path): Pair<Int, Int> {
    val config = modelDir.resolve("config.json").readText()
    val id2label = Regex(""""(\d+)"\s*:\s*"(\w+)"""")
        .findAll(config)
        .associate { it.groupValues[1].toInt() to it.groupValues[2].lowercase() }
    val nonEntailmentLabels = setOf("contradiction", "not_entailment")
    return id2label.entries.first { it.value == "entailment" }.key to
            id2label.entries.first { it.value in nonEntailmentLabels }.key
}
