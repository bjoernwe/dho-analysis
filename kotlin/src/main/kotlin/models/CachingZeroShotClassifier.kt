package models

import java.nio.file.Path
import java.sql.Connection
import java.sql.DriverManager
import kotlin.collections.iterator
import kotlin.io.path.createParentDirectories

// Delegates to another ZeroShotClassifier, persisting (model, label, text) -> score in SQLite
// so repeated experiments don't re-run inference for sentence/label pairs already scored.
class CachingZeroShotClassifier(
    private val delegate: ZeroShotClassifier,
    dbPath: Path,
    private val modelKey: String,
) : ZeroShotClassifier {

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

    override fun scoreBatch(texts: List<String>, label: String): List<Float> {
        if (texts.isEmpty()) return emptyList()

        val cached = getAll(texts, label)
        val missing = texts.filterNot { it in cached }.distinct()
        val fresh = if (missing.isEmpty()) emptyMap() else
            missing.zip(delegate.scoreBatch(missing, label)).toMap().also { putAll(it, label) }

        return texts.map { cached[it] ?: fresh[it]!! }
    }

    private fun getAll(texts: List<String>, label: String): Map<String, Float> {
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

    private fun putAll(scores: Map<String, Float>, label: String) {
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

    override fun close() {
        delegate.close()
        connection.close()
    }
}
