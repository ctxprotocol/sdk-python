# Changelog

## 0.22.0

- Added `bounded_explicit_empty_result_guardrail` to the query controller stop-reason literal type so strict consumers no longer fail validation when the platform returns that live stop reason.
- Platform note (no type change): `computed_artifacts` is now capped at 4 per response by the shared artifact emission policy, so SDK/MCP consumers receive exactly the artifact list the web app displays for the identical run.

## 0.21.0

- `client.query.run()` is now backed by the durable job path (`start()` + `poll()`) instead of a held-open SSE connection. One call now reliably covers the full 1800s hosted compute ceiling and survives transient connection drops — the "sometimes works, sometimes times out on hard queries" failure mode is gone. `run()` accepts optional `interval_ms` / `timeout_ms` keyword arguments.
- `run_or_poll()` is kept as an explicit alias of the same path, and now also synthesizes a fallback developer trace when `include_developer_trace` is set but the backend omits the trace (matching prior `run()` behavior).
- `client.query.stream()` is unchanged and remains the real-time SSE surface.

## 0.20.0

- Poll defaults aligned with the hosted 1800s compute ceiling: `poll()`/`run_or_poll()` check status every 5 seconds over plain HTTP and wait up to 31 minutes by default.
- Documented that HTTP polling costs no model tokens; model turns do.

## 0.19.1

- Fixed `client.query.start()` returning `400 "Invalid query job request"` when called without explicit `tools`. The request body builder now omits optional fields (notably `tools`) when unset instead of serializing them as JSON `null`; the durable jobs endpoint's Zod schema rejects `null` for optional fields. `client.query.run()` and `client.query.stream()` were not failing (the `/api/v1/query` endpoint tolerates `null`), but are now consistent with the TS SDK and MCP behavior.

## 0.19.0

- Fixed `client.query.run()`/`client.query.stream()` raising a `ValidationError` on queries that returned a rendered image artifact (e.g. server-rendered chart PNGs). `computedArtifacts` now accepts both `chart` (structured spec + data) and `image` (hosted URL) variants.
- Added `QueryImageArtifact` (`kind: "image"` with `url`, `alt`, `title`, `content_hash`, `bytes`, `width`, `height`) and made `QueryComputedArtifact` a discriminated union over `kind`.
- Exported `QueryImageArtifact` from the client entry point alongside `QueryChartArtifact`.

## 0.18.0

- Added durable async Query jobs with `client.query.start()`, `client.query.get_status()`, and `client.query.poll()`.
- Exported `QueryJobStartResult`, `QueryJobStatusResult`, and `QueryJobStatus` from the root and client entry points.
- Kept existing `client.query.run()` and `client.query.stream()` behavior unchanged.
