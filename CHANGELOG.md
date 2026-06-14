# Changelog

## 0.18.0

- Added durable async Query jobs with `client.query.start()`, `client.query.get_status()`, and `client.query.poll()`.
- Exported `QueryJobStartResult`, `QueryJobStatusResult`, and `QueryJobStatus` from the root and client entry points.
- Kept existing `client.query.run()` and `client.query.stream()` behavior unchanged.
