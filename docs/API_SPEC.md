# Erudite — Local API

Beacon Computing Initiative · Developer Seal: Chance Joaquin Boone (CJB)

Loopback only: 127.0.0.1:8766. No authentication; intended for a local development account. Host and browser Origin are checked. Eight HTTP workers maximum, one active generation; overload connections are closed and competing generation requests return 429.

GET /health reports API readiness and cached checkpoint state, not model competence.
GET /telemetry reports measured request counts/timing/tokens and OS one-minute host load.
POST /cancel requests cooperative cancellation of the active request; no tensor-kernel preemption.
POST /chat accepts fixed-length application/json with exactly a messages key:
`{"messages":[{"role":"user","content":"Hello"}]}`

History must alternate user/assistant and end in user. Client system overrides are rejected. Response is newline-delimited JSON: start (checkpoint step/dropped turns), token (decoded text), then done (generated token count), cancelled, or error. In-stream errors can follow HTTP 200, so consumers must inspect the terminal event. Do not record partial streams as completed assistant messages.

400 invalid body/history, 403 Origin/Host rejected, 415 content/framing unsupported, 429 model busy, 503 missing/incompatible checkpoint or context failure before streaming begins. The API has bounded socket I/O; generation latency is hardware/model dependent. Logs contain exception details and metadata, not prompt text.
