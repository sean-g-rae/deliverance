package io.teknek.reckoning.tensor;

import io.teknek.reckoning.DType;
import io.teknek.reckoning.tensor2.Lighter;
import io.teknek.reckoning.tensor2.TensorRef;
import net.jafama.FastMath;

/** Direct-fill alternatives for TensorRef math that avoid large intermediate tensors. */
public final class VectorTensorMathUtilsOptimized {
    private VectorTensorMathUtilsOptimized() {
    }

    public static TensorRef precomputeFreqsCis(int dim, int end, double theta, double scalingFactor,
            Lighter lighter) {
        int halfDim = dim / 2;
        int rows = Math.toIntExact((long) end * halfDim);
        TensorRef result = lighter.allocate(DType.F32, TensorShape.of(rows, 2));
        for (int frequency = 0; frequency < halfDim; frequency++) {
            double frequencyValue = (1.0 / FastMath.pow(theta, (2.0 * frequency) / dim)) / scalingFactor;
            for (int position = 0; position < end; position++) {
                int row = position * halfDim + frequency;
                double angle = position * frequencyValue;
                result.set((float) FastMath.cos(angle), row, 0);
                result.set((float) FastMath.sin(angle), row, 1);
            }
        }
        return result;
    }

    public static TensorRef precomputeInverseFrequencies(int dim, double theta, double scalingFactor,
            Lighter lighter) {
        int halfDim = dim / 2;
        TensorRef result = lighter.allocate(DType.F32, TensorShape.of(1, halfDim));
        for (int frequency = 0; frequency < halfDim; frequency++) {
            double value = (1.0 / FastMath.pow(theta, (2.0 * frequency) / dim)) / scalingFactor;
            result.set((float) value, 0, frequency);
        }
        return result;
    }
}
