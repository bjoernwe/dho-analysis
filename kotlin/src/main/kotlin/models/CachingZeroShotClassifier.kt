package models

import java.nio.file.Path
import java.sql.Connection
import java.sql.DriverManager
import java.security.MessageDigest
import java.text.Normalizer
import java.nio.charset.StandardCharsets
import kotlin.collections.iterator
import kotlin.io.path.createParentDirectories

// Delegates to another ZeroShotClassifier, persisting (model, label, text) -> score in SQLite
// so repeated experiments don't re-run inference for sentence/label pairs already scored.
class CachingZeroShotClassifier(
    private val delegate: ZeroShotClassifier,
    dbPath: Path,
    private val modelKey: String,
    private val chunkSize: Int = 997,
) : ZeroShotClassifier {

    private val connection: Connection

    init {
        dbPath.createParentDirectories()
        connection = DriverManager.getConnection("jdbc:sqlite:$dbPath")
        // Create new table (start clean)
        connection.createStatement().use { stmt ->
            stmt.execute(
                """
                CREATE TABLE IF NOT EXISTS scores (
                    model TEXT NOT NULL,
                    label TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    text TEXT NOT NULL,
                    score REAL NOT NULL,
                    PRIMARY KEY (model, label, text_hash)
                )
                """.trimIndent()
            )
        }
    }

    // Normalize text to NFC to avoid misses from different Unicode normalization forms
    private fun normalize(s: String): String = Normalizer.normalize(s, Normalizer.Form.NFC)

    // SHA-256 hex
    private fun sha256Hex(s: String): String {
        val md = MessageDigest.getInstance("SHA-256")
        val bytes = md.digest(s.toByteArray(StandardCharsets.UTF_8))
        return bytes.joinToString("") { "%02x".format(it) }
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

        // Map original text -> normalized -> hash
        val normalizedByText = distinctTexts.associateWith { normalize(it) }
        val hashByText = normalizedByText.mapValues { sha256Hex(it.value) }

        // Chunk hashes to avoid hitting SQLite's parameter limit. Use configured chunkSize.
        val maxPerChunk = chunkSize

        val allHashes = hashByText.values.distinct()
        val hashToScore = mutableMapOf<String, Float>()

        var i = 0
        while (i < allHashes.size) {
            val chunk = allHashes.subList(i, minOf(i + maxPerChunk, allHashes.size))
            val placeholders = chunk.joinToString(separator = ",") { "?" }
            val sql = "SELECT text_hash, score FROM scores WHERE model = ? AND label = ? AND text_hash IN ($placeholders)"
            connection.prepareStatement(sql).use { stmt ->
                stmt.setString(1, modelKey)
                stmt.setString(2, label)
                chunk.forEachIndexed { j, h -> stmt.setString(j + 3, h) }
                stmt.executeQuery().use { rs ->
                    while (rs.next()) {
                        hashToScore[rs.getString("text_hash")] = rs.getFloat("score")
                    }
                }
            }
            i += chunk.size
        }

        val result = mutableMapOf<String, Float>()
        for (t in distinctTexts) {
            val h = hashByText[t]!!
            if (hashToScore.containsKey(h)) {
                result[t] = hashToScore[h]!!
            }
        }
        return result
    }

    private fun putAll(scores: Map<String, Float>, label: String) {
        if (scores.isEmpty()) return

        val sql = "INSERT OR REPLACE INTO scores (model, label, text_hash, text, score) VALUES (?, ?, ?, ?, ?)"
        connection.autoCommit = false
        try {
            connection.prepareStatement(sql).use { stmt ->
                for ((text, score) in scores) {
                    val normalized = normalize(text)
                    val hash = sha256Hex(normalized)
                    stmt.setString(1, modelKey)
                    stmt.setString(2, label)
                    stmt.setString(3, hash)
                    stmt.setString(4, text)
                    stmt.setFloat(5, score)
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
