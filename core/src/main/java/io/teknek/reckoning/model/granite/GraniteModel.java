package io.teknek.reckoning.model.granite;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.reckoning.DType;
import io.teknek.reckoning.generator.Response;
import io.teknek.reckoning.generator.EmbedInput;
import io.teknek.reckoning.generator2.CausalSelfAttention2;
import io.teknek.reckoning.generator2.MLPBlock2;
import io.teknek.reckoning.generator2.RmsNorm2;
import io.teknek.reckoning.generator2.TransformerBlock2;
import io.teknek.reckoning.grace.PreTrainedTokenizer;
import io.teknek.reckoning.math.WrappedForkJoinPool;
import io.teknek.reckoning.model.llama.LlamaModel;
import io.teknek.reckoning.model.tensorparallel.TensorParallelCollectives;
import io.teknek.reckoning.model.tensorparallel.TensorParallelContext;
import io.teknek.reckoning.safetensors.Config;
import io.teknek.reckoning.safetensors.WeightLoader;
import io.teknek.reckoning.tensor.KvBufferCacheSettings;
import io.teknek.reckoning.tensor2.TensorRef;
import io.teknek.reckoning.tensor2.Lighter;
import io.teknek.reckoning.tensor2.Scale;
import io.teknek.reckoning.toolcallparser.ToolCallParser;

import java.util.Optional;
import java.util.stream.IntStream;

/** Dense Granite decoder implementation used by Granite 4.1 LLMON checkpoints. */
public final class GraniteModel extends LlamaModel {
    public GraniteModel(InferenceType inferenceType, Config config, WeightLoader weights,
            PreTrainedTokenizer tokenizer, DType workingMemoryDType, DType workingQType,
            Optional<DType> modelQType, MetricRegistry metricRegistry,
            KvBufferCacheSettings kvBufferCacheSettings, ToolCallParser toolCallParser,
            WrappedForkJoinPool pool, TensorParallelContext tensorParallelContext,
            TensorParallelCollectives tensorParallelCollectives, Optional<DType> outputHeadQuantization,
            Lighter lighter) {
        super(inferenceType, config, weights, tokenizer, workingMemoryDType, workingQType, modelQType,
                metricRegistry, kvBufferCacheSettings, toolCallParser, pool, tensorParallelContext,
                tensorParallelCollectives, outputHeadQuantization, lighter);
    }

    @Override
    protected EmbedInput loadInputWeights() {
        TensorRef embeddingWeights = registerModelTensorRef(weights.loadRef("model.embed_tokens.weight"));
        GraniteConfig granite = (GraniteConfig) config;
        return new EmbedInput(this) {
            @Override
            public TensorRef inputTokenToEmbedding(int inputToken, int position) {
                try (TensorRef source = embeddingWeights.slice(inputToken)) {
                    TensorRef result = makeDenseTensorRef(1, config.embeddingLength);
                    try {
                        copyAndScale(source, result);
                        return result;
                    } catch (RuntimeException | Error failure) {
                        result.close();
                        throw failure;
                    }
                }
            }

            private void copyAndScale(TensorRef source, TensorRef target) {
                TensorRef dense = source.dType() == workingDType ? source : lighter.reshape(source, workingDType);
                try {
                    lighter.copy(dense, 0, target, 0, config.embeddingLength);
                    if (granite.embeddingMultiplier != null && granite.embeddingMultiplier != 1.0f) {
                        lighter.scale(new Scale(granite.embeddingMultiplier).target(target)
                                .offsetAndLength(0, config.embeddingLength));
                    }
                } finally {
                    if (dense != source) dense.close();
                }
            }
        };
    }

    @Override
    protected boolean usesTensorRefExecution() {
        return true;
    }

    @Override
    public boolean usesKvCache2Generation() {
        return true;
    }

    @Override
    protected TransformerBlock2[] loadTransformerBlockWeights2() {
        if (tensorParallelContext.enabled()) {
            throw new UnsupportedOperationException("Granite TensorRef tensor parallelism is not ported");
        }
        DType qType = modelQType.orElse(this.modelDType);
        TransformerBlock2[] blocks = new TransformerBlock2[config.numberOfLayers];
        IntStream.range(0, config.numberOfLayers).parallel().forEach(index -> {
            String base = "model.layers." + index + ".";
            String attention = base + "self_attn.";
            String qName = attention + "q_proj.weight";
            String kName = attention + "k_proj.weight";
            String vName = attention + "v_proj.weight";
            String oName = attention + "o_proj.weight";
            CausalSelfAttention2 selfAttention = new CausalSelfAttention2(this, index,
                    registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(qName, weights::loadRef, qType)),
                    registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(kName, weights::loadRef, qType)),
                    registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(vName, weights::loadRef, qType)),
                    registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(oName, weights::loadRef, qType)),
                    lighter, metricRegistry, qName, kName, vName, oName);

            String mlp = base + "mlp.";
            String gateName = mlp + "gate_proj.weight";
            String upName = mlp + "up_proj.weight";
            String downName = mlp + "down_proj.weight";
            MLPBlock2 feedForward = new MLPBlock2(this,
                    registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(gateName, weights::loadRef, qType)),
                    registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(upName, weights::loadRef, qType)),
                    registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(downName, weights::loadRef, qType)),
                    lighter, gateName, upName, downName);

            String inputNorm = base + "input_layernorm.weight";
            String postAttentionNorm = base + "post_attention_layernorm.weight";
            blocks[index] = new TransformerBlock2(this, index,
                    Optional.of(new RmsNorm2(this, registerModelTensorRef(
                            loadAndMaybeQuantizedExcluding1DTensors(inputNorm, weights::loadRef, qType)), 0.0f)),
                    selfAttention, Optional.empty(),
                    Optional.of(new RmsNorm2(this, registerModelTensorRef(
                            loadAndMaybeQuantizedExcluding1DTensors(postAttentionNorm, weights::loadRef, qType)), 0.0f)),
                    feedForward, Optional.empty(), Optional.empty());
        });
        return blocks;
    }
}
