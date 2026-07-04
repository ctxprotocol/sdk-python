# Changelog

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
