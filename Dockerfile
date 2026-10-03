# sparks: static Go binary on distroless. No shell, no package manager,
# runs as uid 65532 (nonroot) by default. Listens on :8088 (unprivileged).
# Build: docker build -t sparks .
# Run:   docker run --rm -p 8088:8088 --read-only sparks \
#          -backend-url https://api.openai.com -model gpt-6-sol \
#          -temps 'default=1.0'
# Probe: GET /healthz (liveness), GET /readyz (readiness).
ARG GO_VERSION=1.27

FROM --platform=$BUILDPLATFORM golang:${GO_VERSION}-alpine AS build
WORKDIR /src
COPY go.mod ./
COPY sparks/ ./sparks/
COPY cmd/sparks/ ./cmd/sparks/
# Pure Go (no CGO/deps): cross-compile natively per TARGETARCH, no emulation.
ARG TARGETARCH
RUN CGO_ENABLED=0 GOARCH=$TARGETARCH go build -trimpath -ldflags='-s -w' -o /out/sparks ./cmd/sparks

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/sparks /sparks
EXPOSE 8088
USER nonroot:nonroot
ENTRYPOINT ["/sparks"]
