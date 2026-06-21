package examples

import kotlin.io.path.Path
import kotlin.time.measureTime
import models.OnnxZeroShotClassifier

fun main() {
    val texts = listOf(
        "This is cool!",
        "I really did not enjoy this at all.",
        "The weather today is mild and pleasant.",
        "Stock prices plummeted after the announcement.",
        "She scored the winning goal in the final minute.",
        "The new restaurant downtown has amazing food.",
        "This meeting could have been an email.",
        "Scientists discovered a new exoplanet this week.",
    )
    val labels = listOf("positive", "negative", "neutral", "sports", "business", "technology")

    val modelName = "ModernBERT-large-zeroshot-v2.0"

    OnnxZeroShotClassifier(
        modelDir = Path("models/MoritzLaurer").resolve(modelName),
        modelFile = "model_fp16.onnx"
    ).use { model ->
        for (label in labels) {
            val scores = model.scoreBatch(texts, label)
            for ((text, score) in texts.zip(scores)) {
                println("%-50s %-12s %.3f".format(text, label, score))
            }
        }
    }
}
