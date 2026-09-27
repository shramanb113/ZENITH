# syntax=docker/dockerfile:1
# ZENITH HTTP sidecar: CGo + embedded onnxruntime + all-MiniLM-L6-v2.
FROM golang:1.24-bookworm AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN go run ./scripts/download_assets.go
ARG ZENITH_VERSION=dev
RUN CGO_ENABLED=1 go build -trimpath -ldflags="-s -w -X main.version=${ZENITH_VERSION}" -o /out/zenith ./cmd/zenith

FROM debian:bookworm-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --system --create-home --uid 10001 zenith
COPY --from=build /out/zenith /usr/local/bin/zenith
USER zenith
ENV HOME=/home/zenith
EXPOSE 7700
ENTRYPOINT ["zenith", "serve", "--http", ":7700"]
