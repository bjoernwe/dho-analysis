package algorithms

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import smile.math.MathEx
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin

class SfaTest {

    private val n = 500

    // One slow latent source plus three fast ones, mixed into observed coordinates by a fixed
    // invertible matrix with deliberately unequal scales. The scaling is the point: without
    // whitening, SFA collapses onto the smallest-variance direction (whose derivative is small for
    // trivial reasons) instead of the genuinely slow one.
    private fun mixedSources(): Pair<Array<DoubleArray>, DoubleArray> {
        val slow = DoubleArray(n) { sin(2 * PI * it / n) }
        val sources = Array(n) { t ->
            doubleArrayOf(
                slow[t],
                cos(2 * PI * 11 * t / n),
                sin(2 * PI * 23 * t / n),
                cos(2 * PI * 37 * t / n),
            )
        }
        val mixing = arrayOf(
            doubleArrayOf(0.001, 0.002, -0.0015, 0.0005),
            doubleArrayOf(3.0, -1.0, 2.0, 0.5),
            doubleArrayOf(-0.5, 4.0, 1.0, -2.0),
            doubleArrayOf(1.0, 0.5, -3.0, 2.5),
        )
        val observed = Array(n) { t ->
            DoubleArray(4) { i -> (0 until 4).sumOf { j -> mixing[i][j] * sources[t][j] } }
        }
        return observed to slow
    }

    @Test
    fun `recovers a known slow source despite unequal input scales`() {
        val (observed, slow) = mixedSources()

        val sfa = fitSfa(observed, nSlowFeatures = 4)
        val y = sfa.transform(observed)

        val correlation = abs(MathEx.cor(DoubleArray(n) { y[it][0] }, slow))
        assertTrue(correlation > 0.99) { "slowest feature should track the slow source, got r=$correlation" }

        for (m in 1 until sfa.slowness.size) {
            assertTrue(sfa.slowness[m] >= sfa.slowness[m - 1]) { "slowness must be ascending: ${sfa.slowness.toList()}" }
        }
        // The slow source completes one cycle over the series while the fastest completes 37, so the
        // gap between the slowest and the rest should be large.
        assertTrue(sfa.slowness[0] < 0.1 * sfa.slowness[1])
    }

    @Test
    fun `whitening spheres the series and the slow features meet the SFA constraints`() {
        val (observed, _) = mixedSources()

        val mean = MathEx.colMeans(observed)
        val centered = Array(n) { t -> DoubleArray(4) { i -> observed[t][i] - mean[i] } }
        val whitening = whiteningMatrix(covariance(centered))
        val whitened = Array(n) { t ->
            DoubleArray(whitening.size) { k -> (0 until 4).sumOf { i -> whitening[k][i] * centered[t][i] } }
        }

        val whitenedMean = MathEx.colMeans(whitened)
        val whitenedCovariance = covariance(whitened)
        for (i in whitening.indices) {
            assertEquals(0.0, whitenedMean[i], 1e-8)
            for (j in whitening.indices) {
                assertEquals(if (i == j) 1.0 else 0.0, whitenedCovariance[i][j], 1e-8)
            }
        }

        // The same constraints must survive the rotation into slow features.
        val y = fitSfa(observed, nSlowFeatures = 4).transform(observed)
        val yMean = MathEx.colMeans(y)
        val yCovariance = covariance(Array(n) { t -> DoubleArray(4) { m -> y[t][m] - yMean[m] } })
        for (i in 0 until 4) {
            assertEquals(0.0, yMean[i], 1e-8)
            for (j in 0 until 4) {
                assertEquals(if (i == j) 1.0 else 0.0, yCovariance[i][j], 1e-8)
            }
        }
    }

    @Test
    fun `transitions restrict which pairs count as a time step`() {
        val (observed, _) = mixedSources()

        // Leaving out every second pair must still find the same slow direction, just measured over
        // fewer steps -- a guard that transition filtering does not silently shift the fit.
        val all = fitSfa(observed, nSlowFeatures = 1)
        val half = fitSfa(observed, nSlowFeatures = 1, transitions = (0 until n - 1).filter { it % 2 == 0 })

        val correlation = abs(MathEx.cor(
            DoubleArray(n) { all.transform(observed[it])[0] },
            DoubleArray(n) { half.transform(observed[it])[0] },
        ))
        assertTrue(correlation > 0.99) { "transition subset changed the slow direction, r=$correlation" }
    }
}
