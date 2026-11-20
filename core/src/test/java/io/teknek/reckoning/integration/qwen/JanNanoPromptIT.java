package io.teknek.reckoning.integration.qwen;

import io.teknek.reckoning.DType;
import io.teknek.reckoning.JsonUtils;
import io.teknek.reckoning.generator.GeneratorParameters;
import io.teknek.reckoning.generator.Response;
import io.teknek.reckoning.model.AbstractModel;
import io.teknek.reckoning.model.AutoModelConfig;
import io.teknek.reckoning.model.AutoModelForCausaLm;
import io.teknek.reckoning.model.DoNothingGenerateEvent;
import io.teknek.reckoning.model.GenerateEvent;
import io.teknek.reckoning.model.InferenceProfiler;
import io.teknek.reckoning.safetensors.fetch.ModelFetcher;
import io.teknek.reckoning.safetensors.prompt.Function;
import io.teknek.reckoning.safetensors.prompt.PromptContext;
import io.teknek.reckoning.safetensors.prompt.Tool;
import io.teknek.reckoning.safetensors.prompt.ToolCall;
import io.teknek.reckoning.safetensors.prompt.ToolResult;
import io.teknek.reckoning.safetensors.prompt.llmon.LlmOnPrompt;
import io.teknek.reckoning.toolcallparser.QwenToolCallParser;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.net.URLDecoder;
import java.util.UUID;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** End-to-end Jan-nano query using a simple prompt. */
@Tag("large-model")
class JanNanoPromptIT {
    private static final HttpClient HTTP_CLIENT = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(10)).build();
    private static final Map<String, String> WEB_FETCH_CACHE = new ConcurrentHashMap<>();

    //@Disabled("Use this to quantize the model")
    @Test
    void quantizesCheckpointOnDemandAndRunsStructuredLlmonQuery() throws Exception {
        ModelFetcher fetcher = new ModelFetcher("edwardcapriolo", "Jan-nano-JQ4")
                .withDownload(true);
        boolean previousProfiling = InferenceProfiler.isEnabled();
        InferenceProfiler.setEnabled(true);
        InferenceProfiler.reset();
        try {
            try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetcher)
                    //.withQuantizeOnDemand(DType.Q4, "edwardcapriolo",
                    //        "Jan-nano")
                    .buildLocalTransformerModel()) {
                PromptContext pc =model.promptSupport().orElseThrow().builder()
                        .addUserMessage("Count from 1 to 3").build();
                Response response = model.generate(UUID.randomUUID(), pc,
                        new GeneratorParameters()
                                .withTemperature(0.0f)
                                .withSeed(42)
                                .withMaxTokens(50)
                                .withTimeout(Duration.ofSeconds(90)),
                        (a,b,c,d) -> { System.out.print(b); System.out.flush(); });
                assertTrue(response.generatedTokens != null && !response.generatedTokens.isEmpty());
                assertTrue(response.responseTextWithSpecialTokens != null
                        && !response.responseTextWithSpecialTokens.isBlank());
            }
        } finally {
            InferenceProfiler.printSummary("jano", 40);
            InferenceProfiler.printCounters();
            InferenceProfiler.setEnabled(previousProfiling);
        }
    }

    @Test
    void runsSearchAndScrapeStyleToolLoop() throws Exception {
        ModelFetcher fetcher = new ModelFetcher("edwardcapriolo", "Jan-nano")
                .withDownload(true);
        try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetcher)
                .withConfig(JsonUtils.om.readValue("""
                        {"toolCallParser":"io.teknek.reckoning.toolcallparser.QwenToolCallParser"}
                        """, AutoModelConfig.class))
                .buildLocalTransformerModel()) {
            Tool webSearch = Tool.from(Function.builder()
                    .name("web_search")
                    .description("Search the public web and return result titles, snippets, and URLs.")
                    .addParameter("query", "string", "The web search query.", true)
                    .build());
            Tool scrape = Tool.from(Function.builder()
                    .name("scrape")
                    .description("Fetch the full text of a selected result URL.")
                    .addParameter("url", "string", "The result URL to fetch.", true)
                    .addParameter("offset", Map.of("type", "integer"), false)
                    .addParameter("limit", Map.of("type", "integer"), false)
                    .build());

            var prompt = model.promptSupport().orElseThrow().builder()
                    .addSystemMessage("Use web_search first, then scrape the strongest result before answering. Do not answer from memory.")
                    .addToolItem(webSearch)
                    .addToolItem(scrape)
                    .addUserMessage("Does Reckoning support JQ4? If so, how many bits are used for the quantized values, and how are the scale factors stored?");
            PromptContext searchPrompt = prompt.build();
            printBlock("SEARCH PROMPT", searchPrompt.getPrompt());
            StreamingTrace searchTrace = new StreamingTrace("SEARCH");
            Response searchResponse = model.generate(UUID.randomUUID(), searchPrompt,
                    new GeneratorParameters().withTemperature(0.0f).withSeed(42).withMaxTokens(2048),
                    searchTrace);
            searchTrace.finish();
            List<ToolCall> searchCalls = printStep("SEARCH RESPONSE", searchResponse);
            ToolCall search = searchCalls.stream().filter(call -> "web_search".equals(call.getName())).findFirst().orElse(null);
            if (search == null) {
                return;
            }
            String searchResults = webSearch(String.valueOf(search.getParameters().get("query")));
            printToolResult("WEB SEARCH RETURN", searchResults);
            prompt.addToolCall(search).addToolResult(ToolResult.from("web_search", search.getId(), searchResults));

            PromptContext scrapePrompt = prompt.build();
            printBlock("SCRAPE PROMPT", scrapePrompt.getPrompt());
            StreamingTrace scrapeTrace = new StreamingTrace("SCRAPE");
            Response scrapeResponse = model.generate(UUID.randomUUID(), scrapePrompt,
                    new GeneratorParameters().withTemperature(0.0f).withSeed(42).withMaxTokens(2048),
                    scrapeTrace);
            scrapeTrace.finish();
            List<ToolCall> scrapeCalls = printStep("SCRAPE RESPONSE", scrapeResponse);
            ToolCall scrapeCall = scrapeCalls.stream().filter(call -> "scrape".equals(call.getName())).findFirst().orElse(null);
            if (scrapeCall == null) {
                return;
            }
            int offset = integerArgument(scrapeCall, "offset", 0);
            int limit = integerArgument(scrapeCall, "limit", 3000);
            String page = scrape(String.valueOf(scrapeCall.getParameters().get("url")), offset, limit);
            printToolResult("SCRAPE RETURN", page);
            prompt.addToolCall(scrapeCall).addToolResult(ToolResult.from("scrape", scrapeCall.getId(), page));

            PromptContext answerPrompt = prompt.build();
            printBlock("ANSWER PROMPT", answerPrompt.getPrompt());
            StreamingTrace answerTrace = new StreamingTrace("ANSWER");
            Response answer = model.generate(UUID.randomUUID(), answerPrompt,
                    new GeneratorParameters().withTemperature(0.0f).withSeed(42).withMaxTokens(2048),
                    answerTrace);
            answerTrace.finish();
            printStep("FINAL RESPONSE", answer);
        }
    }

    private static String webSearch(String query) throws Exception {
        String encoded = java.net.URLEncoder.encode(query, StandardCharsets.UTF_8);
        HttpRequest request = HttpRequest.newBuilder(URI.create(
                        "https://html.duckduckgo.com/html/?q=" + encoded))
                .header("User-Agent", "Reckoning-JanNanoIT/1.0")
                .GET().build();
        String html = HTTP_CLIENT.send(request, HttpResponse.BodyHandlers.ofString()).body();
        Matcher matcher = Pattern.compile(
                        "class=\\\"result__a\\\"[^>]*href=\\\"([^\\\"]+)\\\"[^>]*>(.*?)</a>",
                        Pattern.DOTALL).matcher(html);
        StringBuilder results = new StringBuilder();
        int count = 0;
        while (matcher.find() && count++ < 5) {
            String url = decodeDuckDuckGoUrl(matcher.group(1));
            results.append("result-").append(count).append(" url=").append(url)
                    .append(" title=").append(stripHtml(matcher.group(2))).append('\n');
        }
        return results.toString();
    }

    private static String scrape(String url, int offset, int limit) throws Exception {
        String content = WEB_FETCH_CACHE.get(url);
        if (content == null) {
            HttpRequest request = HttpRequest.newBuilder(URI.create(url))
                    .header("User-Agent", "Reckoning-JanNanoIT/1.0")
                    .GET().build();
            byte[] bytes = HTTP_CLIENT.send(request, HttpResponse.BodyHandlers.ofByteArray()).body();
            content = stripHtml(new String(bytes, 0, Math.min(bytes.length, 100_000), StandardCharsets.UTF_8));
            WEB_FETCH_CACHE.put(url, content);
        }
        int start = Math.min(Math.max(0, offset), content.length());
        int boundedLimit = Math.min(Math.max(1, limit), 20_000);
        int end = Math.min(content.length(), start + boundedLimit);
        int nextOffset = end < content.length() ? end : -1;
        return "status=200\nurl=" + url + "\ncached=true\ntotalChars=" + content.length()
                + "\noffset=" + start + "\nlimit=" + boundedLimit + "\nnextOffset=" + nextOffset
                + "\n" + content.substring(start, end);
    }

    private static int integerArgument(ToolCall call, String name, int defaultValue) {
        Object value = call.getParameters().get(name);
        return value instanceof Number number ? number.intValue() : defaultValue;
    }

    private static List<ToolCall> printStep(String label, Response response) {
        System.out.println("\n========== " + label + " ==========");
        printBlock("REASONING", response.reasoning);
        System.out.println("response.toolCalls=" + response.toolCalls);
        List<ToolCall> parsed = new QwenToolCallParser().extract(response);
        System.out.println("parsed.toolCalls=" + parsed);
        printBlock("TEXT", response.responseText);
        System.out.println("finishReason=" + response.finishReason + " generatedTokens="
                + (response.generatedTokens == null ? 0 : response.generatedTokens.size()));
        return parsed;
    }

    private static void printToolResult(String label, String result) {
        System.out.println("\n---------- " + label + " ----------");
        System.out.println(result);
    }

    private static void printBlock(String label, String value) {
        System.out.println("\n--- " + label + " ---");
        System.out.println(value == null || value.isBlank() ? "<empty>" : value);
    }

    private static String decodeDuckDuckGoUrl(String value) {
        String url = value.startsWith("//") ? "https:" + value : value;
        int uddg = url.indexOf("uddg=");
        if (uddg >= 0) {
            String encoded = url.substring(uddg + 5);
            int separator = encoded.indexOf('&');
            return URLDecoder.decode(separator >= 0 ? encoded.substring(0, separator) : encoded,
                    StandardCharsets.UTF_8);
        }
        return url;
    }

    private static String stripHtml(String value) {
        return value.replaceAll("(?s)<[^>]*>", " ")
                .replace("&amp;", "&").replace("&quot;", "\"").replace("&#x27;", "'")
                .replaceAll("\\s+", " ").trim();
    }

    private static final class StreamingTrace implements GenerateEvent {
        private final String label;
        private final StringBuilder pending = new StringBuilder();
        private String section = "TEXT";

        private StreamingTrace(String label) {
            this.label = label;
            System.out.println("\n========== " + label + " STREAM ==========");
        }

        @Override
        public void emit(int next, String nextRaw, String nextCleaned, float timing) {
            //System.err.print(nextRaw);
            //System.err.flush();
            pending.append(nextRaw == null ? "" : nextRaw);
            drain(false);
        }

        private void finish() {
            drain(true);
            System.out.println();
        }

        private void drain(boolean finish) {
            while (true) {
                int markerStart = -1;
                String marker = null;
                for (String candidate : List.of("<think>", "</think>", "<tool_call>", "</tool_call>")) {
                    int index = pending.indexOf(candidate);
                    if (index >= 0 && (markerStart < 0 || index < markerStart)) {
                        markerStart = index;
                        marker = candidate;
                    }
                }
                if (marker == null) {
                    int keep = finish ? 0 : 12;
                    if (pending.length() > keep) {
                        print(pending.substring(0, pending.length() - keep));
                        pending.delete(0, pending.length() - keep);
                    }
                    return;
                }
                if (markerStart > 0) {
                    print(pending.substring(0, markerStart));
                }
                pending.delete(0, markerStart + marker.length());
                section = switch (marker) {
                    case "<think>" -> "REASONING";
                    case "</think>" -> "TEXT";
                    case "<tool_call>" -> "TOOL CALL";
                    case "</tool_call>" -> "TEXT";
                    default -> section;
                };
                System.out.print("\n[" + label + " / " + section + "] ");
                System.out.flush();
            }
        }

        private void print(String value) {
            if (!value.isEmpty()) {
                System.out.print(value);
                System.out.flush();
            }
        }
    }
}
