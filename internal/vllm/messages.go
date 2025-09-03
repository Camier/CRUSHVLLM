package vllm

import "time"

// ServerStatus represents the current state of the vLLM server
type ServerStatus int

const (
	ServerStatusIdle ServerStatus = iota
	ServerStatusStopping
	ServerStatusStarting
	ServerStatusRunning
	ServerStatusError
)

// ServerRestartMsg is sent when the vLLM server needs to restart
type ServerRestartMsg struct {
	ModelID string
}

// ServerStatusMsg provides updates about the vLLM server state
type ServerStatusMsg struct {
	Status    ServerStatus
	ModelID   string
	Message   string
	Error     error
	Timestamp time.Time
}

// ServerRestartCompleteMsg is sent when the server restart is complete
type ServerRestartCompleteMsg struct {
	ModelID   string
	Duration  time.Duration
	Timestamp time.Time
}