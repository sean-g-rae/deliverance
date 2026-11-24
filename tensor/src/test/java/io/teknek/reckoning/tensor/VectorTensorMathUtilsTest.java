package io.teknek.reckoning.tensor;

import io.teknek.reckoning.math.VectorMathUtils;
import io.teknek.reckoning.tensor.impl.FloatBufferTensor;
import io.teknek.reckoning.tensor2.Lighter;
import io.teknek.reckoning.tensor2.TensorRef;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.function.IntUnaryOperator;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class VectorTensorMathUtilsTest {

    /*
    From google ai (trust the answers with your life :)
    import numpy as np
    from scipy.special import logsumexp

    logits = np.array([2.0, 1.0, 0.1])
    # logsumexp provides numerical stability to avoid overflow
    logprobs = logits - logsumexp(logits)

    print(logprobs)
    # Result: array([-0.417, -1.417, -2.317])
    */

    @Test
    public void vectorToVector(){
        int rows = 1;
        int cols = 3;
        AbstractTensor original = new FloatBufferTensor(rows, cols);
        original.set(2.0f, 0, 0);
        original.set(1.0f, 0, 1);
        original.set(0.1f, 0, 2);
        AbstractTensor result = new FloatBufferTensor(rows, cols);
        VectorTensorMathUtils.logSumExpTensor(result, original);
        String origin   = "[0][0]=  2.0000 [0][1]=  1.0000 [0][2]=  0.1000";
        String expected = "[0][0]= -0.4170 [0][1]= -1.4170 [0][2]= -2.3170";
        assertEquals(expected, TensorDisplayUtil.pretty2dDisplayAll(result).trim());
    }

    //https://nhigham.com/2021/01/05/what-is-the-log-sum-exp-function/
    @Test
    public void logSumExpTest(){
        int rows =  1;
        int cols = 3;
        AbstractTensor original = new FloatBufferTensor(rows, cols);
        original.set(1.0f, 0, 0);
        original.set(2.0f, 0, 1);
        original.set(3.0f, 0, 2);
        assertEquals(3.4076058864593506f, VectorTensorMathUtils.logSumExp(original), 0.000001);
    }

    @Test
    public void percentileTest(){
        int rows =  1;
        int cols = 10;
        AbstractTensor original = new FloatBufferTensor(rows, cols);
        original.set(1.0f, 0, 0);
        original.set(2.0f, 0, 1);
        original.set(3.0f, 0, 2);
        original.set(-3.0f, 0, 3);
        original.set(3.0f, 0, 4);
        var x = VectorTensorMathUtils.valueBuckets(original);
        assertEquals(-3.0f, x.firstKey());
        assertEquals( "[2, 4]", x.get(3.0f).toString());
        assertEquals("{-3.0=[3], 0.0=[5, 6, 7, 8, 9], 1.0=[0], 2.0=[1], 3.0=[2, 4]}", x.toString());
        assertEquals(1, VectorTensorMathUtils.percentile(x, .90f, original.size()));
    }

    @Test
    public void split(){

    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("ropePrecomputers")
    public void tensorRefPrecomputesPositionMajorCosineSinePairs(String name, RopePrecomputer precompute) {
        Lighter lighter = new Lighter();
        try (TensorRef actual = precompute.apply(4, 2, 4.0, 2.0, lighter)) {
            assertEquals(8, actual.size());
            assertEquals((float) Math.cos(0.0), actual.get(0, 0), 1.0e-6f);
            assertEquals((float) Math.sin(0.0), actual.get(0, 1), 1.0e-6f);
            assertEquals((float) Math.cos(0.5), actual.get(2, 0), 1.0e-6f);
            assertEquals((float) Math.sin(0.5), actual.get(2, 1), 1.0e-6f);
            assertEquals((float) Math.cos(0.25), actual.get(3, 0), 1.0e-6f);
            assertEquals((float) Math.sin(0.25), actual.get(3, 1), 1.0e-6f);
        }
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("ropePrecomputers")
    public void tensorRefPrecomputeHasNoRowsForEmptyPositionRange(String name, RopePrecomputer precompute) {
        try (TensorRef actual = precompute.apply(4, 0, 10_000.0, 1.0, new Lighter())) {
            assertEquals(0, actual.size());
        }
    }

    private static Stream<Arguments> ropePrecomputers() {
        return Stream.of(
                Arguments.of("existing", (RopePrecomputer) VectorTensorMathUtils::precomputeFreqsCis),
                Arguments.of("optimized", (RopePrecomputer) VectorTensorMathUtilsOptimized::precomputeFreqsCis));
    }

    @FunctionalInterface
    private interface RopePrecomputer {
        TensorRef apply(int dim, int end, double theta, double scalingFactor, Lighter lighter);
    }

    @Test
    public void tensorRefPrecomputeMatchesReferenceAtEveryPositionAndFrequency() {
        int dim = 6;
        int end = 4;
        float[][] expected = VectorMathUtils.precomputeFreqsCis(dim, end, 1234.0, 0.75);
        try (TensorRef actual = VectorTensorMathUtils.precomputeFreqsCis(dim, end, 1234.0, 0.75,
                new Lighter())) {
            assertEquals(expected.length * 2, actual.size());
            assertEquals(expected.length, actual.shape().first());
            assertEquals(2, actual.shape().last());
            for (int row = 0; row < expected.length; row++) {
                assertEquals(expected[row][0], actual.get(row, 0), 1.0e-6f);
                assertEquals(expected[row][1], actual.get(row, 1), 1.0e-6f);
            }
        }
    }

    @Test
    public void tensorRefOuterProductMatchesRowMajorReference() {
        Lighter lighter = new Lighter();
        try (TensorRef xs = lighter.allocate(io.teknek.reckoning.DType.F32, TensorShape.of(1, 3));
             TensorRef ys = lighter.allocate(io.teknek.reckoning.DType.F32, TensorShape.of(1, 2))) {
            xs.set(1.0f, 0, 0);
            xs.set(-2.0f, 0, 1);
            xs.set(3.0f, 0, 2);
            ys.set(4.0f, 0, 0);
            ys.set(-5.0f, 0, 1);
            try (TensorRef actual = VectorTensorMathUtils.outerProduct(xs, ys, lighter)) {
                assertEquals(6, actual.size());
                float[] expected = {4.0f, -5.0f, -8.0f, 10.0f, 12.0f, -15.0f};
                for (int i = 0; i < expected.length; i++) {
                    assertEquals(expected[i], actual.get(0, i), 1.0e-6f);
                }
            }
        }
    }

}
