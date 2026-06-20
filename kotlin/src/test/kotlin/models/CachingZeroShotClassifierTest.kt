package models

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import java.nio.file.Files

class CachingZeroShotClassifierTest {

    private class DummyDelegate : ZeroShotClassifier {
        var calls = 0
        override fun scoreBatch(texts: List<String>, label: String): List<Float> {
            calls += texts.size
            return texts.map { it.length.toFloat() }
        }

        override fun close() {}
    }

    @Test
    fun testSpecialCharactersCaching() {
        val tmp = Files.createTempFile("testdb-special", ".sqlite")
        tmp.toFile().deleteOnExit()
        val delegate = DummyDelegate()
        // small chunk size to test chunking logic even for small inputs
        val cache = CachingZeroShotClassifier(delegate, tmp, "m", chunkSize = 3)

        val texts = listOf(
            "hello",
            "naïve",
            "cafe\u0301", // combining accent
            "café", // composed
            "emoji 👍",
            "line\nbreak",
            "embedded\u0000nul",
            "quotes ' \" \\",
            "hello" // duplicate
        )

        val first = cache.scoreBatch(texts, "label1")
        // delegate should have been called for distinct texts only
        assertEquals(texts.distinct().size, delegate.calls)

        // call again; should hit cache and not call delegate more
        val before = delegate.calls
        val second = cache.scoreBatch(texts, "label1")
        assertEquals(before, delegate.calls)
        // results should be stable; don't assert exact numeric lengths for combining sequences
        assertEquals(first.size, second.size)

        cache.close()
    }

    @Test
    fun testLargeBatchChunking() {
        val tmp = Files.createTempFile("testdb-large", ".sqlite")
        tmp.toFile().deleteOnExit()
        val delegate = DummyDelegate()
        // force small chunk size to ensure multiple chunks
        val cache = CachingZeroShotClassifier(delegate, tmp, "m", chunkSize = 100)

        val n = 1200
        val texts = (0 until n).map { "text-$it" }

        val res = cache.scoreBatch(texts, "lbl")
        assertEquals(n, res.size)
        // delegate should have been called once for all missing entries (distinct)
        assertEquals(n, delegate.calls)

        // call again; should all be cached
        val before = delegate.calls
        val res2 = cache.scoreBatch(texts, "lbl")
        assertEquals(before, delegate.calls)
        assertEquals(res, res2)

        cache.close()
    }
}

