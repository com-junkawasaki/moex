/**
 * MoEx Worker — Browser-Based Expert FFN Executor
 *
 * A worker runs inside a browser tab with WebGPU access.
 * It holds a subset of expert FFN weights and executes
 * expert computations on dispatch requests from the hub.
 *
 * Lifecycle:
 *   1. Initialize WebGPU device
 *   2. Download & cache assigned expert weights
 *   3. Connect to hub via WebSocket
 *   4. Enter dispatch–compute–return loop
 */

import {
  MessageType,
  DType,
  HEADER_SIZE,
  decodeFrameHeader,
  decodeFramePayload,
  encodeFrame,
  buildHeartbeatFrame,
  type FrameHeader,
} from "./protocol.js";

// ── Configuration ──────────────────────────────────────────────────────

interface WorkerConfig {
  hubUrl: string;            // WebSocket URL of the hub
  weightsBaseUrl: string;    // HTTP base URL for expert weight files
  hiddenDim: number;         // d = 4096 for Qwen3-30B-A3B
  ffnDim: number;            // d_ff = 2048
  numLayers: number;         // MoE layers
}

const DEFAULT_CONFIG: WorkerConfig = {
  hubUrl: "ws://localhost:8765",
  weightsBaseUrl: "/weights",
  hiddenDim: 4096,
  ffnDim: 2048,
  numLayers: 46,
};

// ── Expert weight storage ──────────────────────────────────────────────

interface ExpertWeights {
  /** Gate projection [ffn_dim, hidden_dim] — INT4 packed */
  gate: GPUBuffer;
  /** Up projection [ffn_dim, hidden_dim] — INT4 packed */
  up: GPUBuffer;
  /** Down projection [hidden_dim, ffn_dim] — INT4 packed */
  down: GPUBuffer;
  /** Quantization scales */
  scales: GPUBuffer;
}

// ── WGSL Shader sources ───────────────────────────────────────────────

const GATE_UP_SHADER = /* wgsl */ `
// Fused gate + up projection with INT4 dequantization
// Input:  x          [hidden_dim]
// Output: gate_up    [2 * ffn_dim]
// Weights: packed INT4 with per-group scales

struct Params {
  hidden_dim: u32,
  ffn_dim: u32,
  group_size: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> gate_weights: array<u32>;
@group(0) @binding(3) var<storage, read> up_weights: array<u32>;
@group(0) @binding(4) var<storage, read> scales: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= params.ffn_dim * 2u) { return; }

  let is_up = row >= params.ffn_dim;
  let actual_row = select(row, row - params.ffn_dim, is_up);

  var acc: f32 = 0.0;

  for (var k = 0u; k < params.hidden_dim; k += 1u) {
    // Read INT4 weight (2 values packed per byte, 8 per u32)
    let packed_idx = (actual_row * params.hidden_dim + k) / 8u;
    let sub_idx = (k % 8u) * 4u;

    let packed = select(
      gate_weights[packed_idx],
      up_weights[packed_idx],
      is_up,
    );
    let q_val = (packed >> sub_idx) & 0xFu;

    // Dequantize
    let group_idx = k / params.group_size;
    let n_groups = params.hidden_dim / params.group_size;
    let scale = scales[actual_row * n_groups + group_idx];
    let w = (f32(q_val) - 8.0) * scale;

    acc += input[k] * w;
  }

  output[row] = acc;
}
`;

const SILU_MUL_SHADER = /* wgsl */ `
// SiLU activation on gate output, element-wise multiply with up output
// Input:  gate_up [2 * ffn_dim]   (gate in [0..ffn_dim), up in [ffn_dim..2*ffn_dim))
// Output: intermediate [ffn_dim]

struct Params {
  ffn_dim: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> gate_up: array<f32>;
@group(0) @binding(2) var<storage, read_write> intermediate: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.ffn_dim) { return; }

  let gate_val = gate_up[i];
  let up_val = gate_up[i + params.ffn_dim];

  // SiLU(x) = x * sigmoid(x)
  let sigmoid_val = 1.0 / (1.0 + exp(-gate_val));
  let silu_val = gate_val * sigmoid_val;

  intermediate[i] = silu_val * up_val;
}
`;

const DOWN_SHADER = /* wgsl */ `
// Down projection: [hidden_dim, ffn_dim] x [ffn_dim] -> [hidden_dim]
// With INT4 dequantization

struct Params {
  hidden_dim: u32,
  ffn_dim: u32,
  group_size: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> down_weights: array<u32>;
@group(0) @binding(3) var<storage, read> scales: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= params.hidden_dim) { return; }

  var acc: f32 = 0.0;

  for (var k = 0u; k < params.ffn_dim; k += 1u) {
    let packed_idx = (row * params.ffn_dim + k) / 8u;
    let sub_idx = (k % 8u) * 4u;
    let q_val = (down_weights[packed_idx] >> sub_idx) & 0xFu;

    let group_idx = k / params.group_size;
    let n_groups = params.ffn_dim / params.group_size;
    let scale = scales[row * n_groups + group_idx];
    let w = (f32(q_val) - 8.0) * scale;

    acc += input[k] * w;
  }

  output[row] = acc;
}
`;

// ── Worker class ───────────────────────────────────────────────────────

export class MoExWorker {
  private config: WorkerConfig;
  private device: GPUDevice | null = null;
  private ws: WebSocket | null = null;
  private experts = new Map<number, ExpertWeights>();
  private assignedExpertIds: number[] = [];

  // Compute pipelines (created once)
  private gateUpPipeline: GPUComputePipeline | null = null;
  private siluMulPipeline: GPUComputePipeline | null = null;
  private downPipeline: GPUComputePipeline | null = null;

  // Reusable GPU buffers
  private inputBuffer: GPUBuffer | null = null;
  private gateUpBuffer: GPUBuffer | null = null;
  private intermediateBuffer: GPUBuffer | null = null;
  private outputBuffer: GPUBuffer | null = null;
  private readbackBuffer: GPUBuffer | null = null;

  constructor(config: Partial<WorkerConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  // ── Initialization ───────────────────────────────────────────────────

  async initialize(): Promise<void> {
    // 1. Request WebGPU device
    if (!navigator.gpu) {
      throw new Error("WebGPU is not supported in this browser");
    }

    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });
    if (!adapter) {
      throw new Error("No WebGPU adapter found");
    }

    this.device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: 256 * 1024 * 1024, // 256 MB
        maxBufferSize: 256 * 1024 * 1024,
      },
    });

    console.log("[Worker] WebGPU device initialized");

    // 2. Create compute pipelines
    this.createPipelines();

    // 3. Allocate reusable buffers
    this.allocateBuffers();

    console.log("[Worker] Pipelines and buffers ready");
  }

  private createPipelines(): void {
    if (!this.device) throw new Error("Device not initialized");

    this.gateUpPipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: {
        module: this.device.createShaderModule({ code: GATE_UP_SHADER }),
        entryPoint: "main",
      },
    });

    this.siluMulPipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: {
        module: this.device.createShaderModule({ code: SILU_MUL_SHADER }),
        entryPoint: "main",
      },
    });

    this.downPipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: {
        module: this.device.createShaderModule({ code: DOWN_SHADER }),
        entryPoint: "main",
      },
    });
  }

  private allocateBuffers(): void {
    if (!this.device) throw new Error("Device not initialized");
    const { hiddenDim, ffnDim } = this.config;

    this.inputBuffer = this.device.createBuffer({
      size: hiddenDim * 4, // f32
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.gateUpBuffer = this.device.createBuffer({
      size: ffnDim * 2 * 4, // f32
      usage: GPUBufferUsage.STORAGE,
    });

    this.intermediateBuffer = this.device.createBuffer({
      size: ffnDim * 4, // f32
      usage: GPUBufferUsage.STORAGE,
    });

    this.outputBuffer = this.device.createBuffer({
      size: hiddenDim * 4, // f32
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    this.readbackBuffer = this.device.createBuffer({
      size: hiddenDim * 4, // f32
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
  }

  // ── Weight loading ───────────────────────────────────────────────────

  async loadExpertWeights(expertIds: number[]): Promise<void> {
    this.assignedExpertIds = expertIds;

    for (const eid of expertIds) {
      console.log(`[Worker] Loading expert ${eid} weights...`);
      const weights = await this.downloadExpertWeights(eid);
      this.experts.set(eid, weights);
    }

    console.log(`[Worker] Loaded ${expertIds.length} experts`);
  }

  private async downloadExpertWeights(expertId: number): Promise<ExpertWeights> {
    if (!this.device) throw new Error("Device not initialized");

    // In production, fetch from HTTP endpoint with Cache API
    // Here we create placeholder buffers for demonstration
    const { hiddenDim, ffnDim } = this.config;
    const groupSize = 128;
    const packedSize = (dim1: number, dim2: number) =>
      Math.ceil((dim1 * dim2) / 8) * 4; // INT4 packed into u32

    const nGroupsGate = Math.ceil(hiddenDim / groupSize);
    const nGroupsDown = Math.ceil(ffnDim / groupSize);

    const gate = this.device.createBuffer({
      size: packedSize(ffnDim, hiddenDim),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: `expert-${expertId}-gate`,
    });

    const up = this.device.createBuffer({
      size: packedSize(ffnDim, hiddenDim),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: `expert-${expertId}-up`,
    });

    const down = this.device.createBuffer({
      size: packedSize(hiddenDim, ffnDim),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: `expert-${expertId}-down`,
    });

    const totalScales =
      ffnDim * nGroupsGate +  // gate scales
      ffnDim * nGroupsGate +  // up scales
      hiddenDim * nGroupsDown; // down scales

    const scales = this.device.createBuffer({
      size: totalScales * 4, // f32
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: `expert-${expertId}-scales`,
    });

    return { gate, up, down, scales };
  }

  // ── WebSocket connection ─────────────────────────────────────────────

  connect(): void {
    this.ws = new WebSocket(this.config.hubUrl);
    this.ws.binaryType = "arraybuffer";

    this.ws.addEventListener("open", () => {
      console.log("[Worker] Connected to hub");
      // Send initial heartbeat with capabilities
      this.ws!.send(buildHeartbeatFrame());
    });

    this.ws.addEventListener("message", (event) => {
      this.onMessage(event.data as ArrayBuffer);
    });

    this.ws.addEventListener("close", () => {
      console.log("[Worker] Disconnected from hub, reconnecting...");
      setTimeout(() => this.connect(), 3000);
    });

    this.ws.addEventListener("error", (err) => {
      console.error("[Worker] WebSocket error:", err);
    });
  }

  // ── Message handling ─────────────────────────────────────────────────

  private async onMessage(data: ArrayBuffer): Promise<void> {
    const header = decodeFrameHeader(data);

    switch (header.messageType) {
      case MessageType.DISPATCH:
        await this.handleDispatch(header, data);
        break;
      case MessageType.CANCEL:
        // Cancel is a hint — current implementation is non-preemptive
        break;
      case MessageType.HEARTBEAT:
        this.ws?.send(buildHeartbeatFrame());
        break;
    }
  }

  private async handleDispatch(
    header: FrameHeader,
    data: ArrayBuffer,
  ): Promise<void> {
    const { layerId, expertId, numTokens, sequenceId } = header;

    const expert = this.experts.get(expertId);
    if (!expert) {
      console.warn(`[Worker] Expert ${expertId} not loaded, ignoring dispatch`);
      return;
    }

    // Extract input tensor from payload
    const payload = decodeFramePayload(data);

    try {
      // Execute expert FFN on GPU
      const result = await this.executeExpertFFN(expert, payload, numTokens);

      // Send result back to hub
      const resultFrame = encodeFrame(
        {
          messageType: MessageType.RESULT,
          sequenceId,
          layerId,
          expertId,
          numTokens,
          hiddenDim: this.config.hiddenDim,
          dtype: DType.F16,
          flags: 0,
        },
        result,
      );

      this.ws?.send(resultFrame);
    } catch (err) {
      console.error(`[Worker] Expert ${expertId} computation failed:`, err);
    }
  }

  // ── GPU computation ──────────────────────────────────────────────────

  private async executeExpertFFN(
    expert: ExpertWeights,
    inputPayload: ArrayBuffer,
    numTokens: number,
  ): Promise<ArrayBuffer> {
    if (!this.device || !this.gateUpPipeline || !this.siluMulPipeline || !this.downPipeline) {
      throw new Error("Worker not initialized");
    }

    const { hiddenDim, ffnDim } = this.config;
    const groupSize = 128;

    // Upload input to GPU
    this.device.queue.writeBuffer(
      this.inputBuffer!,
      0,
      inputPayload,
    );

    // Create uniform buffer for params
    const paramsData = new Uint32Array([hiddenDim, ffnDim, groupSize]);
    const paramsBuffer = this.device.createBuffer({
      size: 12,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(paramsBuffer, 0, paramsData);

    const encoder = this.device.createCommandEncoder();

    // Pass 1: GateUp
    {
      const bindGroup = this.device.createBindGroup({
        layout: this.gateUpPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: this.inputBuffer! } },
          { binding: 2, resource: { buffer: expert.gate } },
          { binding: 3, resource: { buffer: expert.up } },
          { binding: 4, resource: { buffer: expert.scales } },
          { binding: 5, resource: { buffer: this.gateUpBuffer! } },
        ],
      });

      const pass = encoder.beginComputePass();
      pass.setPipeline(this.gateUpPipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil((ffnDim * 2) / 256));
      pass.end();
    }

    // Pass 2: SiLU + Multiply
    {
      const siluParams = new Uint32Array([ffnDim]);
      const siluParamsBuffer = this.device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(siluParamsBuffer, 0, siluParams);

      const bindGroup = this.device.createBindGroup({
        layout: this.siluMulPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: siluParamsBuffer } },
          { binding: 1, resource: { buffer: this.gateUpBuffer! } },
          { binding: 2, resource: { buffer: this.intermediateBuffer! } },
        ],
      });

      const pass = encoder.beginComputePass();
      pass.setPipeline(this.siluMulPipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(ffnDim / 256));
      pass.end();
    }

    // Pass 3: Down projection
    {
      const bindGroup = this.device.createBindGroup({
        layout: this.downPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: this.intermediateBuffer! } },
          { binding: 2, resource: { buffer: expert.down } },
          { binding: 3, resource: { buffer: expert.scales } },
          { binding: 4, resource: { buffer: this.outputBuffer! } },
        ],
      });

      const pass = encoder.beginComputePass();
      pass.setPipeline(this.downPipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(hiddenDim / 256));
      pass.end();
    }

    // Readback
    encoder.copyBufferToBuffer(
      this.outputBuffer!,
      0,
      this.readbackBuffer!,
      0,
      hiddenDim * 4,
    );

    this.device.queue.submit([encoder.finish()]);

    // Map and read result
    await this.readbackBuffer!.mapAsync(GPUMapMode.READ);
    const result = this.readbackBuffer!.getMappedRange().slice(0);
    this.readbackBuffer!.unmap();

    return result;
  }

  // ── Cleanup ──────────────────────────────────────────────────────────

  destroy(): void {
    this.ws?.close();
    this.inputBuffer?.destroy();
    this.gateUpBuffer?.destroy();
    this.intermediateBuffer?.destroy();
    this.outputBuffer?.destroy();
    this.readbackBuffer?.destroy();

    for (const [, expert] of this.experts) {
      expert.gate.destroy();
      expert.up.destroy();
      expert.down.destroy();
      expert.scales.destroy();
    }

    this.device?.destroy();
    console.log("[Worker] Destroyed");
  }
}
