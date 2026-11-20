package io.teknek.reckoning.integration.granite;

import io.teknek.reckoning.DType;
import io.teknek.reckoning.generator.GeneratorParameters;
import io.teknek.reckoning.generator.Response;
import io.teknek.reckoning.model.AbstractModel;
import io.teknek.reckoning.model.AutoModelForCausaLm;
import io.teknek.reckoning.model.DoNothingGenerateEvent;
import io.teknek.reckoning.model.InferenceProfiler;
import io.teknek.reckoning.safetensors.fetch.ModelFetcher;
import io.teknek.reckoning.safetensors.prompt.PromptContext;
import io.teknek.reckoning.safetensors.prompt.llmon.LlmOnPrompt;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.time.Duration;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertTrue;

/** End-to-end Granite 4.1 LLMON query using an upstream renderer golden scenario. */
@Tag("large-model")
class Granite41LlmonPreviewPortIT {

    @Test
    void downloadsCheckpointAndRunsStructuredLlmonQuery() throws Exception {
        ModelFetcher fetcher = new ModelFetcher("ibm-granite", "granite-4.1-3b-llmon-preview")
                .withDownload(true);
        File modelDirectory = fetcher.maybeDownload();
        assertTrue(new File(modelDirectory, "config.json").isFile(),
                "Downloaded model must contain config.json: " + modelDirectory);

        // Practical LLMON document-grounding example: instruction, document, and requirement
        // are represented as separate executable/context blocks.
        String renderedPrompt = LlmOnPrompt.builder()
                .user(LlmOnPrompt.UserMessage.builder()
                        .content("Extract the launch date and owner from the attached project brief.")
                        .document(new LlmOnPrompt.Document(
                                "Project Aurora launches on 2027-03-15. The launch owner is Priya Shah.",
                                "Project Aurora Brief"))
                        .requirement("Return exactly two fields: launch_date and owner.")
                        .build())
                .build()
                .render();

        boolean previousProfiling = InferenceProfiler.isEnabled();
        InferenceProfiler.setEnabled(true);
        InferenceProfiler.reset();
        try {
            try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetcher).buildLocalTransformerModel()) {
                Response response = model.generate(UUID.randomUUID(), PromptContext.of(renderedPrompt),
                        new GeneratorParameters()
                                .withTemperature(0.0f)
                                .withSeed(42)
                                .withMaxTokens(2)
                                .withTimeout(Duration.ofSeconds(90)),
                        (a,b,c,d) -> { System.out.println(b);});
                assertTrue(response.generatedTokens != null && !response.generatedTokens.isEmpty());
                assertTrue(response.responseTextWithSpecialTokens != null
                        && !response.responseTextWithSpecialTokens.isBlank());
            }
        } finally {
            InferenceProfiler.printSummary("granite41-llmon-structured-query", 40);
            InferenceProfiler.printCounters();
            InferenceProfiler.setEnabled(previousProfiling);
        }
    }


    @Disabled("Use this to quantize the model")
    //@Test
    void quantizesCheckpointOnDemandAndRunsStructuredLlmonQuery() throws Exception {
        ModelFetcher fetcher = new ModelFetcher("ibm-granite", "granite-4.1-3b-llmon-preview")
                .withDownload(true);
        String renderedPrompt = LlmOnPrompt.builder()
                .user(LlmOnPrompt.UserMessage.builder()
                        .content("Extract the launch date and owner from the attached project brief.")
                        .document(new LlmOnPrompt.Document(
                                "Project Aurora launches on 2027-03-15. The launch owner is Priya Shah.",
                                "Project Aurora Brief"))
                        .requirement("Return exactly two fields: launch_date and owner.")
                        .build())
                .build()
                .render();

        boolean previousProfiling = InferenceProfiler.isEnabled();
        InferenceProfiler.setEnabled(true);
        InferenceProfiler.reset();
        try {
            try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetcher)
                    .withQuantizeOnDemand(DType.Q4, "edwardcapriolo",
                            "granite-4.1-3b-llmon-preview-JQ4")
                    .buildLocalTransformerModel()) {
                Response response = model.generate(UUID.randomUUID(), PromptContext.of(renderedPrompt),
                        new GeneratorParameters()
                                .withTemperature(0.0f)
                                .withSeed(42)
                                .withMaxTokens(50)
                                .withTimeout(Duration.ofSeconds(90)),
                        (a,b,c,d) -> { System.out.println(b);});
                assertTrue(response.generatedTokens != null && !response.generatedTokens.isEmpty());
                assertTrue(response.responseTextWithSpecialTokens != null
                        && !response.responseTextWithSpecialTokens.isBlank());
            }
        } finally {
            InferenceProfiler.printSummary("granite41-llmon-structured-query-q4", 40);
            InferenceProfiler.printCounters();
            InferenceProfiler.setEnabled(previousProfiling);
        }
    }

}
