# Native All Plan

## Goal

Create one public Maven artifact, `native-all`, containing all native artifacts currently built by GitHub Actions. Make it a normal compile dependency of Reckoning so embedded and web-server usage receive native support automatically.

The initial supported platform classifiers are:

- `linux-x86_64`
- `linux-musl-x86_64`
- `linux-musl-aarch_64`
- `osx-aarch_64`

The artifact should include the complete native library set produced for each platform, including the Reckoning, tensor2, and GPU/Dawn libraries.

## Maven Artifacts

- Publish `native-all` as the consumer-facing Maven artifact.
- Do not publish individual platform classifier JARs as the normal Maven Central dependency.
- Keep platform-specific JARs as intermediate build inputs for assembling `native-all`.
- The transport mechanism for those intermediate JARs is an implementation detail of the build/release workflow.
- The existing `native` module structure should not be split prematurely. First determine whether `native-all` can contain only bundled native resources while using the existing native Java classes.

## Runtime Loader

Add a Reckoning-owned Java loader that:

1. Waits until the first `AutoModel*` construction.
2. Detects the runtime platform/distribution.
3. Selects the matching platform resources from `native-all`.
4. Attempts to load the native set for that platform.
5. Performs initialization exactly once for the lifetime of the JVM.
6. Records success or failure permanently.
7. Falls back to Panama/Java operations if native loading is unavailable or fails.

There must be one shared JVM-wide initialization state across all `AutoModel*` entry points. The implementation should use one common helper/state holder rather than independent static initialization in each model class.

Native loading should not happen merely because an `AutoModel*` class is loaded. Waiting until construction allows callers to configure the kill switch first.

The first initialization should detect the platform and attempt the selected native set as one initialization operation. Do not add separate subsystem-level lazy initialization beyond that point.

## Platform Detection

Write a small Reckoning-owned detector rather than adding `os-maven-plugin` to the runtime classpath.

The detector should match the classifier vocabulary used by the Maven build for the known supported platforms. It may use standard JVM properties and the minimum additional Linux distribution detection needed for the supported classifiers.

Do not depend on Maven plugin classes at runtime. `os-maven-plugin` is Java-based, but it is packaged as Maven build tooling and would add an awkward dependency graph to the application classpath. Its behavior can be used as a reference for matching known classifier mappings.

## Kill Switch

Expose a static control through the standard model initialization API, for example:

```java
AutoModelForCausalLM.specialNativeLoaderEnabled(false);
```

The exact shared API location should support all `AutoModel*` entry points.

Rules:

- The setting must be applied before the first model construction.
- It is a final pre-initialization kill switch.
- Calls after initialization have no effect.
- When disabled, Reckoning performs no automatic bundled native loading.
- Users are responsible for configuring system-native loading through their own JVM/system configuration.

When automatic loading is enabled but the bundled native library is absent or fails to load, the new loader must not fall back to `System.loadLibrary(...)`. It should log the failure and use Panama. Users who need system-provided natives can disable the bundled loader and configure the existing system path themselves.

## Resource Layout

The combined artifact must keep platform libraries in distinct resource directories so same-named libraries cannot overwrite one another:

```text
META-INF/native/linux-x86_64/...
META-INF/native/linux-musl-x86_64/...
META-INF/native/linux-musl-aarch_64/...
META-INF/native/osx-aarch_64/...
```

The loader should extract the selected platform's complete native set into one temporary directory before loading. Keeping dependent libraries together avoids the current per-library temporary-directory behavior and makes native dependency loading deterministic.

Generated jextract classes should remain generated and should not own platform detection or resource extraction. The loader must initialize before the generated bindings perform symbol lookup.

## Logging

Use the existing SLF4J logging framework during normal operation. Web-server, benchmark, and Java-main flows should clearly report:

- detected platform
- selected classifier
- whether bundled native resources were found
- native artifact/resource source
- extraction directory when applicable
- libraries attempted and loaded
- final native status
- failure reason and Panama fallback

Unit tests may suppress or reduce this logging to avoid repeated noise.

If logging is unavailable because of an early classloading or startup problem, emit a concise diagnostic to `stderr`.

Expected outcomes should be distinguishable, including:

- native artifact not on the classpath
- unsupported platform
- bundled resource missing
- extraction failure
- native load/link failure
- native successfully loaded
- Panama fallback selected
- bundled loader disabled by the user

## Testing

Add focused tests for:

- exactly-once initialization across multiple `AutoModel*` entry points
- permanent failure state with no retry
- kill switch set before construction
- kill switch having no effect after initialization
- matching platform resource selection
- unsupported platform behavior
- `native-all` absent from the classpath
- bundled native loading failure and Panama fallback
- complete platform resource layout
- native dependency extraction into one directory
- clear load status and failure diagnostics
- generated jextract symbol lookup after loader initialization

Run the native matrix currently built by GitHub Actions against the assembled artifact:

- Linux glibc x86_64
- Linux musl x86_64
- Linux musl ARM64
- macOS ARM64

## Release Assembly

The existing GitHub native matrix remains responsible for producing the platform-specific native JAR inputs. A later assembly step should combine those inputs into `native-all` before the Maven Central release deployment.

The public release should contain one normal native dependency, `native-all`. Individual platform artifacts may remain available as internal assembly inputs and may be attached to GitHub releases if useful, but they are not the consumer-facing Maven Central contract.

The current manual release flow in `release.md` must eventually be updated so `native-all` is assembled before:

```sh
mvn -Prelease-central deploy -Dmaven.test.skip.exec=true
```
