package io.teknek.reckoning.model.granite;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import io.teknek.reckoning.math.ActivationFunction;
import io.teknek.reckoning.safetensors.Config;

import java.util.List;
import java.util.Map;

/** Configuration for the dense Granite decoder used by Granite 4.1 LLMON. */
public final class GraniteConfig extends Config {
    public final boolean attentionBias;
    public final boolean mlpBias;
    public final float attentionDropout;
    public final float logitsScaling;
    public final boolean tieWordEmbeddings;
    public final boolean useCache;

    @JsonCreator
    public GraniteConfig(
            @JsonProperty("max_position_embeddings") int contextLength,
            @JsonProperty("hidden_size") int embeddingLength,
            @JsonProperty("intermediate_size") int hiddenLength,
            @JsonProperty("num_attention_heads") int numberOfHeads,
            @JsonProperty("num_key_value_heads") Integer numberOfKeyValueHeads,
            @JsonProperty("num_hidden_layers") int numberOfLayers,
            @JsonProperty("rms_norm_eps") float layerNormEps,
            @JsonProperty("vocab_size") int vocabularySize,
            @JsonProperty("bos_token_id") Integer bosToken,
            @JsonProperty("eos_token_id") Object eosToken,
            @JsonProperty("hidden_act") ActivationFunction.Type activationFunction,
            @JsonProperty("rope_theta") Double ropeTheta,
            @JsonProperty("rope_parameters") Map<String, Object> ropeParameters,
            @JsonProperty("residual_multiplier") Float residualMultiplier,
            @JsonProperty("attention_multiplier") Float attentionMultiplier,
            @JsonProperty("embedding_multiplier") Float embeddingMultiplier,
            @JsonProperty("logits_scaling") Float logitsScaling,
            @JsonProperty("attention_bias") Boolean attentionBias,
            @JsonProperty("mlp_bias") Boolean mlpBias,
            @JsonProperty("attention_dropout") Float attentionDropout,
            @JsonProperty("tie_word_embeddings") Boolean tieWordEmbeddings,
            @JsonProperty("use_cache") Boolean useCache,
            @JsonProperty("architectures") List<String> architectures
    ) {
        super(contextLength, embeddingLength, hiddenLength, numberOfHeads,
                numberOfKeyValueHeads == null ? numberOfHeads : numberOfKeyValueHeads,
                numberOfLayers, layerNormEps, vocabularySize, bosToken == null ? 0 : bosToken,
                toIntList(eosToken), activationFunction,
                ropeTheta(ropeTheta, ropeParameters), ropeParameters, null,
                embeddingLength / numberOfHeads, null, null,
                residualMultiplier, attentionMultiplier, embeddingMultiplier, logitsScaling,
                architectures);
        this.attentionBias = Boolean.TRUE.equals(attentionBias);
        this.mlpBias = Boolean.TRUE.equals(mlpBias);
        this.attentionDropout = attentionDropout == null ? 0.0f : attentionDropout;
        this.logitsScaling = logitsScaling == null ? 1.0f : logitsScaling;
        this.tieWordEmbeddings = Boolean.TRUE.equals(tieWordEmbeddings);
        this.useCache = useCache == null || useCache;
    }

    private static Double ropeTheta(Double ropeTheta, Map<String, Object> ropeParameters) {
        if (ropeParameters != null && ropeParameters.get("rope_theta") instanceof Number value) {
            return value.doubleValue();
        }
        return ropeTheta == null ? 10_000.0 : ropeTheta;
    }

    private static List<Integer> toIntList(Object eosToken) {
        if (eosToken instanceof Number value) return List.of(value.intValue());
        if (eosToken instanceof List<?> values) {
            return values.stream().map(value -> ((Number) value).intValue()).toList();
        }
        if (eosToken == null) return List.of();
        throw new IllegalArgumentException("Unsupported eos_token_id: " + eosToken);
    }
}
