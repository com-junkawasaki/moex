/**
 * MoEx Hub — Coordination Engine
 *
 * The hub orchestrates distributed MoE inference:
 *  1. Executes attention / layer-norm / router on the hub device
 *  2. Dispatches activated expert inputs to WebGPU workers
 *  3. Collects results with hedged-dispatch tail-latency mitigation
 *  4. Combines weighted expert outputs and proceeds to next layer
 *
 * This is a reference implementation — the actual attention kernels
 * would be backed by PyTorch, ONNX Runtime, or a native GPU backend.
 */

import { WebSocketServer, WebSocket } from "ws";
import {
  MessageType,
  DType,
  decodeFrameHeader,
  decodeFramePayload,
  buildDispatchFrame,
  buildCancelFrame,
  buildHeartbeatFrame,
  type FrameHeader,
} from "./protocol.js";

// ── Configuration ──────────────────────────────────────────────────────

interface HubConfig {
  port: number;
  numExperts: number;       // E — total experts per MoE layer
  numMoeLayers: number;     // L_moe
  topK: number;             // k — experts activated per token
  hiddenDim: number;        // d
  hedgingFactor: number;    // h — replicas to contact per expert
  timeoutMs: number;        // per-expert timeout
  replicationFactor: number; // r — copies of each expert
}

const DEFAULT_CONFIG: HubConfig = {
  port: 8765,
  numExperts: 128,
  numMoeLayers: 46,
  topK: 8,
  hiddenDim: 4096,
  hedgingFactor: 2,
  timeoutMs: 500,
  replicationFactor: 2,
};

// ── Worker state ───────────────────────────────────────────────────────

interface WorkerInfo {
  ws: WebSocket;
  id: string;
  expertIds: number[];
  healthy: boolean;
  latencyP50: number;       // estimated p50 latency in ms
  consecutiveTimeouts: number;
}

// ── Expert dispatch tracking ───────────────────────────────────────────

interface PendingExpert {
  expertId: number;
  layerId: number;
  sequenceId: number;
  dispatchedTo: string[];  // worker IDs
  resolved: boolean;
  result: ArrayBuffer | null;
  resolve: (result: ArrayBuffer) => void;
  timer: ReturnType<typeof setTimeout>;
}

// ── Hub class ──────────────────────────────────────────────────────────

export class MoExHub {
  private config: HubConfig;
  private workers = new Map<string, WorkerInfo>();
  private expertToWorkers = new Map<number, string[]>(); // expertId -> workerIds
  private sequenceCounter = 0;
  private wss: WebSocketServer | null = null;

  constructor(config: Partial<HubConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  // ── Lifecycle ────────────────────────────────────────────────────────

  start(): void {
    this.wss = new WebSocketServer({ port: this.config.port });
    console.log(`[Hub] Listening on ws://localhost:${this.config.port}`);

    this.wss.on("connection", (ws) => this.onWorkerConnect(ws));

    // Periodic heartbeat
    setInterval(() => this.broadcastHeartbeat(), 10_000);
  }

  stop(): void {
    this.wss?.close();
    console.log("[Hub] Stopped");
  }

  // ── Worker management ────────────────────────────────────────────────

  private onWorkerConnect(ws: WebSocket): void {
    const workerId = `worker-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
    const assignedExperts = this.assignExperts(workerId);

    const info: WorkerInfo = {
      ws,
      id: workerId,
      expertIds: assignedExperts,
      healthy: true,
      latencyP50: 20, // initial estimate
      consecutiveTimeouts: 0,
    };

    this.workers.set(workerId, info);

    // Update expert -> worker mapping
    for (const eid of assignedExperts) {
      const existing = this.expertToWorkers.get(eid) ?? [];
      existing.push(workerId);
      this.expertToWorkers.set(eid, existing);
    }

    console.log(
      `[Hub] Worker ${workerId} connected, assigned experts: [${assignedExperts.join(",")}]`
    );

    ws.on("message", (data) =>
      this.onWorkerMessage(workerId, data as ArrayBuffer)
    );

    ws.on("close", () => this.onWorkerDisconnect(workerId));
    ws.on("error", () => this.onWorkerDisconnect(workerId));
  }

  private onWorkerDisconnect(workerId: string): void {
    const info = this.workers.get(workerId);
    if (!info) return;

    // Remove from expert -> worker mapping
    for (const eid of info.expertIds) {
      const workers = this.expertToWorkers.get(eid);
      if (workers) {
        const filtered = workers.filter((w) => w !== workerId);
        if (filtered.length > 0) {
          this.expertToWorkers.set(eid, filtered);
        } else {
          this.expertToWorkers.delete(eid);
        }
      }
    }

    this.workers.delete(workerId);
    console.log(`[Hub] Worker ${workerId} disconnected`);
  }

  /** Assign expert IDs to a new worker using round-robin */
  private assignExperts(workerId: string): number[] {
    const workerIndex = this.workers.size; // 0-based
    const N = Math.max(this.workers.size + 1, 1);
    const expertsPerWorker = Math.ceil(
      (this.config.replicationFactor * this.config.numExperts) / N
    );

    const assigned: number[] = [];
    for (let i = 0; i < this.config.numExperts; i++) {
      // Consistent hashing: assign expert i to worker if hash matches
      const hash = simpleHash(i, workerIndex, this.config.replicationFactor);
      if (hash < expertsPerWorker) {
        assigned.push(i);
      }
    }

    return assigned.slice(0, Math.min(assigned.length, expertsPerWorker));
  }

  // ── Message handling ─────────────────────────────────────────────────

  private pendingExperts = new Map<string, PendingExpert>(); // key: `${layerId}-${expertId}-${seqId}`

  private onWorkerMessage(workerId: string, data: ArrayBuffer): void {
    const header = decodeFrameHeader(data);

    switch (header.messageType) {
      case MessageType.RESULT:
        this.onExpertResult(workerId, header, data);
        break;
      case MessageType.HEARTBEAT:
        this.onHeartbeat(workerId);
        break;
    }
  }

  private onExpertResult(
    workerId: string,
    header: FrameHeader,
    data: ArrayBuffer,
  ): void {
    const key = `${header.layerId}-${header.expertId}-${header.sequenceId}`;
    const pending = this.pendingExperts.get(key);

    if (!pending || pending.resolved) {
      // Already resolved (hedged duplicate) or unknown — discard
      return;
    }

    // Accept first response
    pending.resolved = true;
    pending.result = decodeFramePayload(data);
    clearTimeout(pending.timer);
    pending.resolve(pending.result);

    // Cancel other hedged workers
    for (const wid of pending.dispatchedTo) {
      if (wid !== workerId) {
        const worker = this.workers.get(wid);
        if (worker?.ws.readyState === WebSocket.OPEN) {
          worker.ws.send(
            buildCancelFrame(header.sequenceId, header.layerId, header.expertId)
          );
        }
      }
    }

    // Update worker latency estimate
    const worker = this.workers.get(workerId);
    if (worker) {
      worker.consecutiveTimeouts = 0;
    }

    this.pendingExperts.delete(key);
  }

  private onHeartbeat(workerId: string): void {
    const worker = this.workers.get(workerId);
    if (worker) {
      worker.healthy = true;
      worker.consecutiveTimeouts = 0;
    }
  }

  // ── Dispatch ─────────────────────────────────────────────────────────

  /**
   * Dispatch a single expert activation with hedging.
   * Returns a promise that resolves with the expert's output tensor.
   */
  async dispatchExpert(
    layerId: number,
    expertId: number,
    hiddenStates: Float32Array,
    numTokens: number,
    gateWeight: number,
  ): Promise<ArrayBuffer> {
    const seqId = this.sequenceCounter++;
    const candidateWorkers = this.selectWorkersForExpert(
      expertId,
      this.config.hedgingFactor,
    );

    if (candidateWorkers.length === 0) {
      throw new Error(`No workers available for expert ${expertId}`);
    }

    // Build dispatch frame
    const frame = buildDispatchFrame(
      seqId,
      layerId,
      expertId,
      hiddenStates,
      numTokens,
      this.config.hiddenDim,
      gateWeight,
    );

    return new Promise<ArrayBuffer>((resolve, reject) => {
      const key = `${layerId}-${expertId}-${seqId}`;

      const pending: PendingExpert = {
        expertId,
        layerId,
        sequenceId: seqId,
        dispatchedTo: candidateWorkers.map((w) => w.id),
        resolved: false,
        result: null,
        resolve,
        timer: setTimeout(() => {
          if (!pending.resolved) {
            pending.resolved = true;
            this.pendingExperts.delete(key);
            // Mark workers as potentially unhealthy
            for (const w of candidateWorkers) {
              w.consecutiveTimeouts++;
              if (w.consecutiveTimeouts >= 3) {
                w.healthy = false;
                console.warn(`[Hub] Worker ${w.id} marked unhealthy`);
              }
            }
            reject(new Error(`Expert ${expertId} timed out on layer ${layerId}`));
          }
        }, this.config.timeoutMs),
      };

      this.pendingExperts.set(key, pending);

      // Send to all hedged workers concurrently
      for (const worker of candidateWorkers) {
        if (worker.ws.readyState === WebSocket.OPEN) {
          worker.ws.send(frame);
        }
      }
    });
  }

  /**
   * Dispatch all activated experts for a single layer.
   * Returns combined expert outputs.
   */
  async dispatchLayer(
    layerId: number,
    activatedExperts: { expertId: number; gateWeight: number }[],
    hiddenStates: Float32Array,
    numTokens: number,
  ): Promise<ArrayBuffer[]> {
    const promises = activatedExperts.map((expert) =>
      this.dispatchExpert(
        layerId,
        expert.expertId,
        hiddenStates,
        numTokens,
        expert.gateWeight,
      )
    );

    return Promise.all(promises);
  }

  /** Select up to h workers that hold the given expert, preferring healthy/fast ones */
  private selectWorkersForExpert(
    expertId: number,
    h: number,
  ): WorkerInfo[] {
    const workerIds = this.expertToWorkers.get(expertId) ?? [];
    const candidates = workerIds
      .map((id) => this.workers.get(id))
      .filter((w): w is WorkerInfo => w !== undefined && w.healthy)
      .sort((a, b) => a.latencyP50 - b.latencyP50);

    return candidates.slice(0, h);
  }

  // ── Heartbeat ────────────────────────────────────────────────────────

  private broadcastHeartbeat(): void {
    const frame = buildHeartbeatFrame();
    for (const [, worker] of this.workers) {
      if (worker.ws.readyState === WebSocket.OPEN) {
        worker.ws.send(frame);
      }
    }
  }

  // ── Status ───────────────────────────────────────────────────────────

  getStatus(): {
    workerCount: number;
    healthyWorkers: number;
    expertCoverage: number;
  } {
    const healthy = [...this.workers.values()].filter((w) => w.healthy).length;
    return {
      workerCount: this.workers.size,
      healthyWorkers: healthy,
      expertCoverage: this.expertToWorkers.size,
    };
  }
}

// ── Utility ────────────────────────────────────────────────────────────

function simpleHash(expertId: number, workerIndex: number, r: number): number {
  // Simple deterministic hash for expert placement
  let h = expertId * 2654435761 + workerIndex * 40503;
  h = ((h >>> 16) ^ h) * 0x45d9f3b;
  h = ((h >>> 16) ^ h) * 0x45d9f3b;
  h = (h >>> 16) ^ h;
  return Math.abs(h) % (r * 128);
}

// ── Main entry point ───────────────────────────────────────────────────

if (import.meta.url === `file://${process.argv[1]}`) {
  const hub = new MoExHub();
  hub.start();

  process.on("SIGINT", () => {
    hub.stop();
    process.exit(0);
  });
}
