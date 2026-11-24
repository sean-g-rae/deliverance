package io.teknek.reckoning.tensor2;

import io.teknek.reckoning.DType;
import io.teknek.reckoning.tensor.TensorShape;
import io.teknek.reckoning.tensor.VectorTensorMathUtilsOptimized;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CompositeOpsRotaryEmbeddingTest {
    @Test
    void onDemandMatchesFullTableForQueryAndKeyShapes() {
        assertMatchesFullTable(3, 4, 2, 7, 13L);
        assertMatchesFullTable(2, 8, 5, 11, 17L);
        assertMatchesFullTable(8, 16, 3, 29, 23L);
    }

    @Test
    void onDemandMatchesFullTableAcrossFuzzCases() {
        Random random = new Random(0x524F5045L);
        for (int test = 0; test < 150; test++) {
            int heads = 1 + random.nextInt(8);
            int headSize = 2 * (1 + random.nextInt(16));
            int rows = 1 + random.nextInt(12);
            int startPosition = random.nextInt(128);
            assertMatchesFullTable(heads, headSize, rows, startPosition, random.nextLong());
        }
    }

    @Test
    void onDemandRejectsInsufficientInverseFrequencies() {
        Lighter lighter = new Lighter();
        CompositeOps ops = new CompositeOps(lighter);
        try (TensorRef target = lighter.allocate(DType.F32, TensorShape.of(1, 8));
             TensorRef inverse = lighter.allocate(DType.F32, TensorShape.of(1, 3))) {
            assertThrows(IllegalArgumentException.class,
                    () -> ops.rotaryEmbeddingOnDemand(target, 1, 8, 0, inverse));
        }
    }

    @Test
    void onDemandRejectsOddHeadSize() {
        Lighter lighter = new Lighter();
        CompositeOps ops = new CompositeOps(lighter);
        try (TensorRef target = lighter.allocate(DType.F32, TensorShape.of(1, 7));
             TensorRef inverse = lighter.allocate(DType.F32, TensorShape.of(1, 4))) {
            assertThrows(IllegalArgumentException.class,
                    () -> ops.rotaryEmbeddingOnDemand(target, 1, 7, 0, inverse));
        }
    }

    private static void assertMatchesFullTable(int heads, int headSize, int rows, int startPosition, long seed) {
        Lighter lighter = new Lighter();
        CompositeOps ops = new CompositeOps(lighter);
        int columns = heads * headSize;
        int contextEnd = startPosition + rows;
        try (TensorRef expected = lighter.allocate(DType.F32, TensorShape.of(rows, columns));
             TensorRef actual = lighter.allocate(DType.F32, TensorShape.of(rows, columns));
             TensorRef full = VectorTensorMathUtilsOptimized.precomputeFreqsCis(
                     headSize, contextEnd, 10_000.0, 1.0, lighter);
             TensorRef inverse = VectorTensorMathUtilsOptimized.precomputeInverseFrequencies(
                     headSize, 10_000.0, 1.0, lighter)) {
            fillPair(expected, actual, seed);
            ops.rotaryEmbedding(expected, heads, headSize, startPosition, full);
            ops.rotaryEmbeddingOnDemand(actual, heads, headSize, startPosition, inverse);
            for (int row = 0; row < rows; row++) {
                for (int column = 0; column < columns; column++) {
                    float expectedValue = expected.get(row, column);
                    float tolerance = 2.0e-5f + Math.abs(expectedValue) * 2.0e-5f;
                    assertEquals(expectedValue, actual.get(row, column), tolerance,
                            "heads=" + heads + " headSize=" + headSize + " rows=" + rows
                                    + " start=" + startPosition + " row=" + row + " column=" + column);
                }
            }
        }
    }

    private static void fillPair(TensorRef expected, TensorRef actual, long seed) {
        Random random = new Random(seed);
        for (int row = 0; row < expected.shape().first(); row++) {
            for (int column = 0; column < expected.shape().last(); column++) {
                float value = (random.nextFloat() - 0.5f) * 20.0f;
                expected.set(value, row, column);
                actual.set(value, row, column);
            }
        }
    }
}
