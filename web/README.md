# YAGPT Web Interface

Node-graph pipeline control for the YAGPT LLM training pipeline. Each pipeline stage (data prep, pretrain, SFT, LoRA, alignment, eval, generate) is a draggable node connected by wires, with real-time WebSocket streaming of training metrics.

## Quick Start

### Backend

```bash
# Install web dependencies
pip install yagpt[web]

# Start the FastAPI server
uv run uvicorn web.backend.app:app --port 8000

# Or use the entry point
yagpt-web
```

### Frontend

```bash
cd web/frontend
npm install
npm run dev
```

Then open http://localhost:3000.

## Architecture

- **Backend**: FastAPI (port 8000) wrapping existing yagpt Python classes
- **Frontend**: Next.js (port 3000) with Zustand state management
- **Real-time**: WebSocket broadcasts training metrics from a custom `WebSocketCallback`
- **Training**: Runs in background threads; one job at a time; stoppable via event flag

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/ws` | WS | WebSocket for real-time metrics |
| `/api/data/{download,tokenize,validate}` | POST | Data preparation |
| `/api/pretrain/{start,stop}` | POST | Start/stop pretraining |
| `/api/pretrain/{status,metrics}` | GET | Training status and metrics |
| `/api/pretrain/config/schema` | GET | TrainConfig field definitions |
| `/api/sft/{start,stop}` | POST | Start/stop SFT |
| `/api/lora/apply` | POST | Apply LoRA/QLoRA |
| `/api/lora/info` | GET | LoRA parameter info |
| `/api/alignment/{start,stop}` | POST | Start/stop DPO/SimPO/GRPO |
| `/api/eval/run` | POST | Run lm-eval benchmarks |
| `/api/eval/results` | GET | Get evaluation results |
| `/api/generate` | POST | Generate text |
| `/api/checkpoints` | GET | List checkpoints |
| `/api/checkpoints/inspect` | GET | Inspect a checkpoint |
