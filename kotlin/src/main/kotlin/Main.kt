import kotlin.io.path.Path
import models.ZeroShotClassifier

fun main() {
    ZeroShotClassifier(Path("models/MoritzLaurer/ModernBERT-large-zeroshot-v2.0"), modelFile = "model_fp16.onnx").use { model ->
        val score = model.score("This is cool!", "positive")
        println(score)
    }
}
