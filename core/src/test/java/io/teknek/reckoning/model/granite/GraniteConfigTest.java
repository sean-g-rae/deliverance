package io.teknek.reckoning.model.granite;

import io.teknek.reckoning.JsonUtils;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GraniteConfigTest {

    @Test
    void parsesGranite41LlmonCheckpointConfig() throws Exception {
        Path configPath = Path.of("..", "docs/models/granite-4.1-3b-llmon-preview/config.json");
        GraniteConfig config = JsonUtils.om.readValue(configPath.toFile(), GraniteConfig.class);

        assertEquals(131072, config.contextLength);
        assertEquals(2560, config.embeddingLength);
        assertEquals(8192, config.hiddenLength);
        assertEquals(40, config.numberOfHeads);
        assertEquals(8, config.numberOfKeyValueHeads);
        assertEquals(5, config.headGroupSize);
        assertEquals(64, config.headSize);
        assertEquals(40, config.numberOfLayers);
        assertEquals(100352, config.vocabularySize);
        assertEquals(100257, config.bosToken);
        assertEquals(java.util.List.of(100257), config.eosTokens);
        assertEquals(12.0f, config.embeddingMultiplier);
        assertEquals(0.015625f, config.attentionMultiplier);
        assertEquals(0.22f, config.residualMultiplier);
        assertEquals(10.0f, config.logitsScaling);
        assertFalse(config.attentionBias);
        assertFalse(config.mlpBias);
        assertFalse(config.tieWordEmbeddings);
        assertFalse(config.useCache);
        assertTrue(config.architectures.contains("GraniteForCausalLM"));
    }

    @Test
    void roundTripsIntegerGraniteMultipliers() throws Exception {
        GraniteConfig original = JsonUtils.om.readValue("""
                {
                  "architectures": ["GraniteForCausalLM"],
                  "model_type": "granite",
                  "max_position_embeddings": 32,
                  "hidden_size": 16,
                  "intermediate_size": 32,
                  "num_attention_heads": 4,
                  "num_key_value_heads": 2,
                  "num_hidden_layers": 2,
                  "rms_norm_eps": 0.00001,
                  "vocab_size": 64,
                  "bos_token_id": 1,
                  "eos_token_id": 2,
                  "hidden_act": "silu",
                  "rope_theta": 10000,
                  "embedding_multiplier": 12,
                  "attention_multiplier": 0.015625,
                  "residual_multiplier": 0.22,
                  "logits_scaling": 10
                }
                """, GraniteConfig.class);

        GraniteConfig roundTripped = JsonUtils.om.readValue(JsonUtils.om.writeValueAsString(Map.ofEntries(
                Map.entry("architectures", original.architectures),
                Map.entry("model_type", "granite"),
                Map.entry("max_position_embeddings", original.contextLength),
                Map.entry("hidden_size", original.embeddingLength),
                Map.entry("intermediate_size", original.hiddenLength),
                Map.entry("num_attention_heads", original.numberOfHeads),
                Map.entry("num_key_value_heads", original.numberOfKeyValueHeads),
                Map.entry("num_hidden_layers", original.numberOfLayers),
                Map.entry("rms_norm_eps", original.layerNormEps),
                Map.entry("vocab_size", original.vocabularySize),
                Map.entry("bos_token_id", original.bosToken),
                Map.entry("eos_token_id", original.eosTokens.getFirst()),
                Map.entry("hidden_act", "silu"),
                Map.entry("rope_theta", 10000),
                Map.entry("embedding_multiplier", 12),
                Map.entry("attention_multiplier", original.attentionMultiplier),
                Map.entry("residual_multiplier", original.residualMultiplier),
                Map.entry("logits_scaling", 10)
        )), GraniteConfig.class);

        assertEquals(12.0f, roundTripped.embeddingMultiplier);
        assertEquals(10.0f, roundTripped.logitsScaling);
        assertEquals(0.015625f, roundTripped.attentionMultiplier);
        assertEquals(0.22f, roundTripped.residualMultiplier);
    }
}
