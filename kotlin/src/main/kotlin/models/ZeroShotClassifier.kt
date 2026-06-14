package models

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import java.nio.LongBuffer
import java.nio.file.Path
import kotlin.io.path.readText
import kotlin.math.exp

class ZeroShotClassifier(modelDir: Path, modelFile: String = "model.onnx") : AutoCloseable {

    private val tokenizer = HuggingFaceTokenizer.newInstance(modelDir)
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val entailmentIndex: Int
    private val contradictionIndex: Int

    init {
        val opts = OrtSession.SessionOptions().apply {
            try {
                addCUDA(0)
            } catch (e: OrtException) {
                System.err.println("CUDA unavailable (${e.message}), falling back to CPU")
            }
        }
        session = env.createSession(modelDir.resolve(modelFile).toString(), opts)
        val (e, c) = resolveNliIndices(modelDir)
        entailmentIndex = e
        contradictionIndex = c
    }

    // Mirrors Python: classifier([msg], [label], multi_label=True)
    fun score(text: String, label: String): Float {
        val encoding = tokenizer.encode(text, "This example is $label.")
        val shape = longArrayOf(1L, encoding.ids.size.toLong())

        val inputs = buildMap<String, OnnxTensor> {
            put("input_ids", OnnxTensor.createTensor(env, LongBuffer.wrap(encoding.ids), shape))
            put("attention_mask", OnnxTensor.createTensor(env, LongBuffer.wrap(encoding.attentionMask), shape))
            if ("token_type_ids" in session.inputNames) {
                put("token_type_ids", OnnxTensor.createTensor(env, LongBuffer.wrap(encoding.typeIds), shape))
            }
        }

        return session.run(inputs).use { result ->
            @Suppress("UNCHECKED_CAST")
            val logits = (result.get("logits").get().value as Array<FloatArray>)[0]
            twoWaySoftmax(logits[entailmentIndex], logits[contradictionIndex])
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
