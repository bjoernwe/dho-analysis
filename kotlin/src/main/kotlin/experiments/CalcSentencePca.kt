package experiments

import data.Message
import data.filterByAuthor
import data.filterByCategory
import data.filterByThreadAuthor
import data.getSentences
import data.readMessages
import org.jetbrains.kotlinx.dataframe.DataFrame
import kotlin.io.path.Path

val featuresPath = Path("data/sentence_features.csv")
val loadingsPath = Path("data/sentence_pca_loadings.csv")

fun DataFrame<Message>.filterByLindasPracticeLogs(): DataFrame<Message> {
    return this.filterByCategory().filterByAuthor().filterByThreadAuthor()
}

fun main() {
    val result = readMessages()
        .filterByLindasPracticeLogs()
        .getSentences()
        .calcPca()

    result.writeLoadings(loadingsPath)
    result.writeFeatures(featuresPath)
}
