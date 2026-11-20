package io.teknek.reckoning.tensor.operations.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/** Loads the bundled native runtime once for the lifetime of the JVM. */
public final class NativeLibraryManager {
    private static final Logger LOGGER = LoggerFactory.getLogger(NativeLibraryManager.class);
    private static final Object LOCK = new Object();
    private static final List<String> LIBRARIES = List.of(
            "webgpu_dawn", "dxcompiler", "dxil", "reckoninggpu", "reckoning_tensor2");
    private static volatile boolean enabled = true;
    private static volatile boolean initialized;
    private static volatile boolean available;
    private static volatile String classifier;
    private static volatile String reason = "not initialized";

    private NativeLibraryManager() {
    }

    public static void specialNativeLoaderEnabled(boolean value) {
        synchronized (LOCK) {
            if (initialized) {
                LOGGER.warn("Native loader setting ignored after initialization: enabled={}", value);
                return;
            }
            enabled = value;
        }
    }

    public static boolean initialize() {
        if (initialized) {
            return available;
        }
        synchronized (LOCK) {
            if (initialized) {
                return available;
            }
            initialized = true;
            if (!enabled) {
                reason = "automatic native loading disabled";
                LOGGER.info("Native loader disabled; using Panama fallback");
                return false;
            }
            try {
                classifier = selectClassifier();
                Path extractionRoot = Path.of(System.getProperty("reckoning.native.tempdir",
                        System.getProperty("java.io.tmpdir")));
                Files.createDirectories(extractionRoot);
                Path directory = Files.createTempDirectory(extractionRoot, "reckoning-native-");
                Set<String> loaded = new LinkedHashSet<>();
                for (String library : LIBRARIES) {
                    try {
                        Path extracted = extractIfPresent(classifier, library, directory);
                        if (extracted == null) {
                            continue;
                        }
                        System.load(extracted.toAbsolutePath().toString());
                        loaded.add(library);
                        LOGGER.debug("Loaded bundled native library {} from {}", library, extracted);
                    } catch (Throwable failure) {
                        LOGGER.info("Optional native library unavailable library={} classifier={} reason={}",
                                library, classifier, failure.toString());
                        LOGGER.debug("Native library load failure for " + library, failure);
                        if ("reckoning_tensor2".equals(library)) {
                            throw new IllegalStateException("Required native library failed to load: " + library, failure);
                        }
                    }
                }
                if (!loaded.contains("reckoning_tensor2")) {
                    reason = "Tensor2 native library not found for classifier " + classifier;
                    LOGGER.info("Native runtime unavailable: {}; using Panama fallback", reason);
                    return false;
                }
                available = true;
                reason = "loaded bundled native libraries: " + loaded;
                LOGGER.info("Native runtime loaded successfully source=bundled-classpath "
                                + "classifier={} libraries={} extractedTo={} fallback=panama-available",
                        classifier, loaded, directory);
                return true;
            } catch (Throwable failure) {
                reason = failure.toString();
                if (failure instanceof LinkageError) {
                    System.err.println("Reckoning native loader failed: " + reason);
                }
                LOGGER.info("Native runtime unavailable classifier={} reason={}; using Panama fallback",
                        classifier, reason);
                LOGGER.debug("Native runtime load failure", failure);
                return false;
            }
        }
    }

    public static boolean available() {
        return available;
    }

    public static String reason() {
        return reason;
    }

    public static String classifier() {
        return classifier;
    }

    private static Path extractIfPresent(String platform, String library, Path directory) throws IOException {
        String extension = extension();
        for (String filename : filenames(library, extension)) {
            String resourceName = "META-INF/native/" + platform + "/" + filename;
            URL resource = NativeLibraryManager.class.getClassLoader().getResource(resourceName);
            if (resource == null) {
                continue;
            }
            Path target = directory.resolve(filename);
            try (InputStream input = resource.openStream()) {
                Files.copy(input, target, StandardCopyOption.REPLACE_EXISTING);
            }
            return target;
        }
        return null;
    }

    private static List<String> filenames(String library, String extension) {
        List<String> names = new ArrayList<>(2);
        names.add("lib" + library + extension);
        names.add(library + extension);
        return names;
    }

    private static String extension() {
        String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        return os.contains("mac") ? ".dylib" : os.contains("win") ? ".dll" : ".so";
    }

    private static String selectClassifier() {
        String override = System.getProperty("reckoning.native.classifier");
        if (override != null && !override.isBlank()) {
            LOGGER.info("Native classifier override={}", override);
            return override;
        }
        String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        String arch = System.getProperty("os.arch", "").toLowerCase(Locale.ROOT);
        String normalizedArch = arch.contains("aarch64") || arch.contains("arm64") ? "aarch_64"
                : arch.contains("amd64") || arch.contains("x86_64") || arch.contains("x86-64") ? "x86_64" : arch;
        if (os.contains("mac")) {
            return "osx-" + normalizedArch;
        }
        if (os.contains("linux")) {
            String generic = "linux-" + normalizedArch;
            if (musl()) {
                String musl = "linux-musl-" + normalizedArch;
                if (hasBundledTensor2(musl)) {
                    return musl;
                }
            }
            return generic;
        }
        return os.replaceAll("\\s+", "-") + "-" + normalizedArch;
    }

    private static boolean hasBundledTensor2(String platform) {
        String resource = "META-INF/native/" + platform + "/libreckoning_tensor2" + extension();
        return NativeLibraryManager.class.getClassLoader().getResource(resource) != null;
    }

    private static boolean musl() {
        String arch = System.getProperty("os.arch", "").toLowerCase(Locale.ROOT);
        String normalized = arch.contains("aarch64") || arch.contains("arm64") ? "aarch64" : "x86_64";
        return Files.exists(Path.of("/lib/ld-musl-" + normalized + ".so.1"))
                || Files.exists(Path.of("/usr/lib/ld-musl-" + normalized + ".so.1"));
    }
}
