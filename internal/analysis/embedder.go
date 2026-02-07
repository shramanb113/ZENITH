package analysis

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

type EmbedRequest struct {
	Text string `json:"text"`
}

type EmbedResponse struct {
	Embedding []float32 `json:"embedding"`
}

var nerveClient = &http.Client{
	Timeout: 10 * time.Second, // Give the AI time to think
}

func GetEmbedding(text string) ([]float32, error) {
	reqBody, _ := json.Marshal(EmbedRequest{Text: text})

	// Talk to the Python Nerve on Port 5000
	resp, err := nerveClient.Post("http://localhost:5000/embed", "application/json", bytes.NewBuffer(reqBody))
	if err != nil {
		return nil, fmt.Errorf("nerve offline: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("nerve returned error: %d", resp.StatusCode)
	}

	var res EmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return nil, fmt.Errorf("failed to decode nerve response: %w", err)
	}

	return res.Embedding, nil
}
