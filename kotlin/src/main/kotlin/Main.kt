import kotlin.io.path.Path
import kotlin.time.measureTime
import models.ZeroShotClassifier

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
    val rounds = 20
    val count = texts.size * labels.size * rounds

    ZeroShotClassifier(Path("models/MoritzLaurer/ModernBERT-large-zeroshot-v2.0"), modelFile = "model_fp16.onnx").use { model ->
        // Warm up so session/provider init isn't counted in the timings below.
        model.score(texts[0], labels[0])

        for (label in labels) {
            val scores = model.scoreBatch(texts, label)
            for ((text, score) in texts.zip(scores)) {
                println("%-50s %-12s %.3f".format(text, label, score))
            }
        }
        println()

        val sequentialElapsed = measureTime {
            repeat(rounds) {
                for (text in texts) {
                    for (label in labels) {
                        model.score(text, label)
                    }
                }
            }
        }
        println("Sequential: $count classifications in $sequentialElapsed (${sequentialElapsed / count} per call)")

        val batchedElapsed = measureTime {
            repeat(rounds) {
                for (label in labels) {
                    model.scoreBatch(texts, label)
                }
            }
        }
        println("Batched:    $count classifications in $batchedElapsed (${batchedElapsed / count} per call)")
    }
}
