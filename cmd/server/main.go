package main

import (
	"log"
	"net"
	"os"
	"os/signal"
	"syscall"

	"github.com/shramanb113/ZENITH/gen/go/zenithproto"
	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/index"
	"github.com/shramanb113/ZENITH/internal/server"
	"google.golang.org/grpc"
)

func main() {
	lis, err := net.Listen("tcp", ":8080")

	if err != nil {
		log.Fatalf("Error occurred : %s", err)
	}

	idx := index.NewInMemoryIndex()
	tkz := analysis.NewStandardTokenizer()

	grpcServer := grpc.NewServer()
	zenithServer := &server.ZenithServer{
		Index:     idx,
		Tokenizer: tkz,
	}

	zenithproto.RegisterSearchServiceServer(grpcServer, zenithServer)

	stop := make(chan os.Signal, 1)

	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)

	go func() {
		log.Printf("ZENITH engine is live on %v", lis.Addr())

		if err := grpcServer.Serve(lis); err != nil {
			log.Fatalf("failed to serve: %v", err)
		}
	}()

	<-stop

	grpcServer.GracefulStop()

}
