package models

import java.nio.file.Path
import kotlin.io.path.Path

const val cacheFile = "cache/scores.db"

enum class Model(val modelName: String, val modelFile: String, val batchSizeTokenBudget: Int) {
    DEBERTA_V3_LARGE("deberta-v3-large-zeroshot-v2.0", "model.onnx", 47_000),
    MODERNBERT_LARGE("ModernBERT-large-zeroshot-v2.0", "model_fp16.onnx", 47_000);

    val modelDir: Path get() = Path("models/MoritzLaurer").resolve(modelName)
}

val activeModel = Model.DEBERTA_V3_LARGE

private val delegateModel: ZeroShotClassifier by lazy {
    OnnxZeroShotClassifier(
        modelDir = activeModel.modelDir,
        modelFile = activeModel.modelFile,
        batchSizeTokenBudget = activeModel.batchSizeTokenBudget,
    )
}

val defaultModel: ZeroShotClassifier by lazy {
    CachingZeroShotClassifier(delegateModel, Path(cacheFile), activeModel.modelName)
}
