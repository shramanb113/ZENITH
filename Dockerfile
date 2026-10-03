# syntax=docker/dockerfile:1
# ZENITH CLI + HTTP sidecar: CGo + embedded onnxruntime (gte-small bundled
# default) + Tesseract OCR for image indexing.
FROM golang:1.24-bookworm AS build
WORKDIR /src
RUN apt-get update \
 && apt-get install -y --no-install-recommends libtesseract-dev libleptonica-dev pkg-config \
 && rm -rf /var/lib/apt/lists/*
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN go run ./scripts/download_assets.go
ARG ZENITH_VERSION=dev
RUN CGO_ENABLED=1 go build -trimpath -tags ocr -ldflags="-s -w -X main.version=${ZENITH_VERSION}" -o /out/zenith ./cmd/zenith
RUN CGO_ENABLED=1 go build -trimpath -ldflags="-s -w" -o /out/zenith-client ./cmd/client

FROM debian:bookworm-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates libtesseract5 tesseract-ocr-eng \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --system --create-home --uid 10001 zenith \
 && mkdir -p /home/zenith/data \
 && chown zenith:zenith /home/zenith/data
COPY --from=build /out/zenith /usr/local/bin/zenith
COPY --from=build /out/zenith-client /usr/local/bin/zenith-client
USER zenith
ENV HOME=/home/zenith
EXPOSE 7700 8080
ENTRYPOINT ["zenith"]
CMD ["serve", "--db", "/home/zenith/data/zenith.db"]
