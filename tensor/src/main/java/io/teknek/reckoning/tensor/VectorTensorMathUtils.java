package io.teknek.reckoning.tensor;

import io.teknek.reckoning.DType;
import io.teknek.reckoning.tensor2.Lighter;
import io.teknek.reckoning.tensor2.TensorRef;
import net.jafama.FastMath;

import java.util.*;

import static io.teknek.reckoning.math.VectorMathUtils.outerProduct;

public class VectorTensorMathUtils {
    public static void l2normalize(AbstractTensor x) {
        float sum = 0.0f;
        for (int i = 0; i < x.shape().last(); i++) {
            float v = x.get(0, i);
            sum += v * v;
        }
        double magnitude = FastMath.sqrt(sum);
        for (int i = 0; i < x.shape().last(); i++)
            x.set((float) (x.get(0, i) / magnitude), 0, i);
    }

    public static void logSumExpTensor(AbstractTensor result, AbstractTensor input) {
        float logsumexp = (float) logSumExp(input);
        for (int i = 0; i < input.size(); i++) {
            float v = input.get(0, i);
            result.set(v - logsumexp, 0, i);
        }
    }

    //https://nhigham.com/2021/01/05/what-is-the-log-sum-exp-function/
    // TODO: This is not the numerically stable log-sum-exp form. It should subtract max(x) before exponentiating and
    // add max(x) back after the log; otherwise large positive logits can overflow and large negative logits can
    // underflow. Use a provider-backed max/exp path when this is updated for hot inference code.
    public static double logSumExp(AbstractTensor x){
        float sum = 0.0f;
        for (int i = 0; i < x.size(); i++) {
            sum += (float) FastMath.exp(x.get(0, i));
        }
        return (float) FastMath.log(sum);
    }

    public static int percentile(SortedMap<Float, List<Integer>> valueBuckets, float perc, long size) {
        int element = (int) ((size * perc) - 1);
        Iterator<Map.Entry<Float, List<Integer>>> iter = valueBuckets.entrySet().iterator();
        int ct = 0;
        while (iter.hasNext()) {
            Map.Entry<Float, List<Integer>> entry = iter.next();
            ct += entry.getValue().size();
            //This condition returns a slightly higher percentile then requested as we are not doing
            //and exact count inside the bucket
            if (ct >= element) {
                System.out.println("arrived at element "+ ct);
                //System.out.println(entry.getValue());
                return entry.getValue().get(0); //could be a random one here
            }
        }
        return -1;
    }

    public static SortedMap<Float, List<Integer>> valueBuckets(AbstractTensor x) {
        SortedMap<Float, List<Integer>> buckets = new TreeMap<>();
        for (int i = 0; i < x.size(); i++) {
            float v = x.get(0, i);
            if (buckets.containsKey(v)) {
                buckets.get(v).add(i);
            } else {
                ArrayList<Integer> al = new ArrayList<>();
                al.add(i);
                buckets.put(v, al);
            }
        }
        return buckets;
    }

    public static TensorRef precomputeFreqsCis(int dim, int end, double theta, double scaling_factor, Lighter lighter) {

        TensorRef freqs = lighter.allocate(DType.F32, TensorShape.of(dim/2));
        float step = 0.0f;
        for (int i = 0; i < freqs.size(); i++, step += 2.0f) {
            float val = (float) ((1.0 / FastMath.pow(theta, step / dim)) / scaling_factor);
            freqs.set(val, 0, i);
        }
        TensorRef t = lighter.allocate(DType.F32, TensorShape.of(end));
        for (int i = 0 ; i <t.size(); i++){
            t.set((float)i,0, i);
        }
        TensorRef freqs_cis = outerProduct(t, freqs, lighter);
        t.close();
        freqs.close();
        TensorRef r = lighter.allocate(DType.F32, TensorShape.of((int) freqs_cis.size(), 2));
        for (int i = 0; i <freqs_cis.size(); i++){
            r.set((float) FastMath.cos(freqs_cis.get(0, i)),  i, 0);
            r.set((float) FastMath.sin(freqs_cis.get(0, i)),  i, 1);
        }
        return r;
    }

    public static TensorRef outerProduct(TensorRef xs, TensorRef ys, Lighter lighter){
        int n = (int) xs.size();
        int m = (int) ys.size();
        TensorRef results = lighter.allocate(DType.F32, TensorShape.of(n * m));
        int idx = 0;
        for (int i = 0 ; i < n; i++){
            for (int j =0; j< m; j++){
                results.set( xs.get(0, i) * ys.get(0, j), 0, idx);
                idx++;
            }
        }
        return results;
    }

    /*
        public static float[] outerProduct(float[] xs, float[] ys) {
        int n = xs.length;
        int m = ys.length;
        float[] result = new float[n * m];
        int idx = 0;
        for (float x : xs) {
            for (float y : ys) {
                result[idx++] = x * y;
            }
        }
        return result;
    }

     */

    public static void normalize(AbstractTensor t){
        double sum = 0.0;
        for (int i = 0; i < t.shape().last(); i++) {
            sum += t.get(0, i);
        }
        for (int i = 0; i < t.shape().last(); i++) {
            t.set((float) (t.get(0, i) / sum), 0, i);
        }
    }
    /*
        public static double[] normalize(double[] input) {
        double sum = 0;
        for (double p : input) sum += p;

        double[] normalized = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            normalized[i] = input[i] / sum;
        }
        return normalized;
    }
     */
}
