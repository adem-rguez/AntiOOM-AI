export interface FitEstimationRequest {
  parameter_count_billions: number;
  quantization?: string;
  context_size?: number;
  modality?: string;
}

export interface FitEstimationResult {
  model_weight_bytes: number;
  kv_cache_bytes: number;
  total_required_vram_bytes: number;
  fits_in_vram: boolean;
  fits_in_ram: boolean;
  recommended_gpu_layers: number;
  estimated_tok_per_sec: number;
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface ChatCompletionRequest {
  model: string;
  messages: ChatMessage[];
  temperature?: number;
  max_tokens?: number;
}

export class DisposClient {
  private baseUrl: string;

  constructor(baseUrl: string = 'http://localhost:8080') {
    this.baseUrl = baseUrl.replace(/\/$/, '');
  }

  async health(): Promise<string> {
    const res = await fetch(`${this.baseUrl}/health`);
    if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
    return res.text();
  }

  async estimateFit(req: FitEstimationRequest): Promise<FitEstimationResult> {
    const res = await fetch(`${this.baseUrl}/v1/fit-estimator`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        quantization: 'Q4_K_M',
        context_size: 4096,
        modality: 'Text',
        ...req,
      }),
    });
    if (!res.ok) throw new Error(`Fit estimation failed: ${res.statusText}`);
    return res.json();
  }

  async chatCompletion(req: ChatCompletionRequest): Promise<any> {
    const res = await fetch(`${this.baseUrl}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    });
    if (!res.ok) throw new Error(`Chat completion failed: ${res.statusText}`);
    return res.json();
  }
}
