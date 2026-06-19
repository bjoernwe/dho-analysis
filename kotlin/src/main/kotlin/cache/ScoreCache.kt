package cache

import models.ZeroShotClassifier
import java.nio.file.Path
import java.sql.Connection
import java.sql.DriverManager
import kotlin.io.path.createParentDirectories

// Persists (model, label, text) -> score so repeated experiments don't re-run inference
// for sentence/label pairs that have already been scored.
class ScoreCache(dbPath: Path, private val modelKey: String) : AutoCloseable {

    private val connection: Connection

    init {
        dbPath.createParentDirectories()
        connection = DriverManager.getConnection("jdbc:sqlite:$dbPath")
        connection.createStatement().use {
            it.execute(
                """
                CREATE TABLE IF NOT EXISTS scores (
                    model TEXT NOT NULL,
                    label TEXT NOT NULL,
                    text TEXT NOT NULL,
                    score REAL NOT NULL,
                    PRIMARY KEY (model, label, text)
                )
                """.trimIndent()
            )
        }
    }

    fun getAll(texts: List<String>, label: String): Map<String, Float> {
        if (texts.isEmpty()) return emptyMap()

        val distinctTexts = texts.distinct()
        val placeholders = distinctTexts.joinToString(",") { "?" }
        val sql = "SELECT text, score FROM scores WHERE model = ? AND label = ? AND text IN ($placeholders)"
        connection.prepareStatement(sql).use { stmt ->
            stmt.setString(1, modelKey)
            stmt.setString(2, label)
            distinctTexts.forEachIndexed { i, text -> stmt.setString(i + 3, text) }
            stmt.executeQuery().use { rs ->
                val result = mutableMapOf<String, Float>()
                while (rs.next()) {
                    result[rs.getString("text")] = rs.getFloat("score")
                }
                return result
            }
        }
    }

    fun putAll(scores: Map<String, Float>, label: String) {
        if (scores.isEmpty()) return

        val sql = "INSERT OR REPLACE INTO scores (model, label, text, score) VALUES (?, ?, ?, ?)"
        connection.autoCommit = false
        try {
            connection.prepareStatement(sql).use { stmt ->
                for ((text, score) in scores) {
                    stmt.setString(1, modelKey)
                    stmt.setString(2, label)
                    stmt.setString(3, text)
                    stmt.setFloat(4, score)
                    stmt.addBatch()
                }
                stmt.executeBatch()
            }
            connection.commit()
        } finally {
            connection.autoCommit = true
        }
    }

    override fun close() = connection.close()
}

// Cache-aware scoring, kept outside ZeroShotClassifier so the model class stays free of
// persistence concerns.
fun ZeroShotClassifier.scoreBatchCached(cache: ScoreCache, texts: List<String>, label: String): List<Float> {
    if (texts.isEmpty()) return emptyList()

    val cached = cache.getAll(texts, label)
    val missing = texts.filterNot { it in cached }.distinct()
    val fresh = if (missing.isEmpty()) emptyMap() else
        missing.zip(scoreBatch(missing, label)).toMap().also { cache.putAll(it, label) }

    return texts.map { cached[it] ?: fresh[it]!! }
}
