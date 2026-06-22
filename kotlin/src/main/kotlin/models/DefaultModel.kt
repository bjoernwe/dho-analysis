package models

import java.nio.file.Path
import kotlin.io.path.Path

const val modelName = "ModernBERT-large-zeroshot-v2.0"
val modelDir: Path = Path("models/MoritzLaurer").resolve(modelName)
const val modelFile = "model_fp16.onnx"
const val cacheFile = "cache/scores.db"

val delegateModel: ZeroShotClassifier by lazy { OnnxZeroShotClassifier(modelDir = modelDir, modelFile = modelFile) }

val defaultModel: ZeroShotClassifier by lazy {
    CachingZeroShotClassifier(delegateModel, Path(cacheFile), modelName)
}
