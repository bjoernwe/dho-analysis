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

    // Compute normalized form and SHA-256 hash for a list of texts
    private fun computeHashes(distinctTexts: List<String>): Map<String, String> {
        return distinctTexts.associateWith { text ->
            val normalized = normalize(text)
            sha256Hex(normalized)
        }
    }

    // Query the DB in chunks for a list of hashes and return map hash -> score
    @Synchronized
    private fun queryHashesInChunks(hashes: List<String>, label: String): Map<String, Float> {
        if (hashes.isEmpty()) return emptyMap()
        val hashToScore = mutableMapOf<String, Float>()
        var i = 0
        while (i < hashes.size) {
            val chunk = hashes.subList(i, minOf(i + chunkSize, hashes.size))
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
        return hashToScore
    }

    // Insert or replace scores (thread-safe)
    @Synchronized
    private fun insertScores(scores: Map<String, Float>, label: String) {
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

    override fun scoreBatch(texts: List<String>, label: String): List<Float> {
        if (texts.isEmpty()) return emptyList()

        val cached = getAll(texts, label)
        val missing = texts.filterNot { it in cached }.distinct()
        val fresh = if (missing.isEmpty()) emptyMap() else
            missing.zip(delegate.scoreBatch(missing, label)).toMap().also { insertScores(it, label) }

        return texts.map { cached[it] ?: fresh[it]!! }
    }

    private fun getAll(texts: List<String>, label: String): Map<String, Float> {
        if (texts.isEmpty()) return emptyMap()

        val distinctTexts = texts.distinct()
        val hashByText = computeHashes(distinctTexts)
        val allHashes = hashByText.values.distinct()
        val hashToScore = queryHashesInChunks(allHashes, label)

        val result = mutableMapOf<String, Float>()
        for (t in distinctTexts) {
            val h = hashByText[t]!!
            if (hashToScore.containsKey(h)) {
                result[t] = hashToScore[h]!!
            }
        }
        return result
    }

    // putAll removed; use insertScores directly

    override fun close() {
        delegate.close()
        connection.close()
    }
}
