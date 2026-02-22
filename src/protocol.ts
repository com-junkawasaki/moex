/**
 * MoEx Binary Transport Protocol
 *
 * Defines the binary frame format for hub–worker communication
 * over WebSocket binary frames.
 */

// ── Constants ──────────────────────────────────────────────────────────

export const MOEX_MAGIC = 0x4d6f4578; // "MoEx"
export const MOEX_VERSION = 1;
export const HEADER_SIZE = 28; // bytes

// ── Message types ──────────────────────────────────────────────────────

export const enum MessageType {
  DISPATCH = 0x01,
  RESULT = 0x02,
  CANCEL = 0x03,
  HEARTBEAT = 0x04,
  WEIGHT_SYNC = 0x05,
}

// ── Data types ─────────────────────────────────────────────────────────

export const enum DType {
  F16 = 0x01,
  BF16 = 0x02,
  INT8 = 0x03,
  INT4 = 0x04,
}

const DTYPE_BYTES: Record<DType, number> = {
  [DType.F16]: 2,
  [DType.BF16]: 2,
  [DType.INT8]: 1,
  [DType.INT4]: 0.5,
};

// ── Flags ──────────────────────────────────────────────────────────────

export const FLAG_COMPRESSED = 1 << 0;

// ── Frame header structure ─────────────────────────────────────────────

export interface FrameHeader {
  magic: number;
  version: number;
  messageType: MessageType;
  payloadLength: number;
  sequenceId: number;
  layerId: number;
  expertId: number;
  numTokens: number;
  hiddenDim: number;
  dtype: DType;
  flags: number;
}

// ── Encoder ────────────────────────────────────────────────────────────

export function encodeFrame(
  header: Omit<FrameHeader, "magic" | "version" | "payloadLength">,
  payload: ArrayBuffer,
): ArrayBuffer {
  const total = HEADER_SIZE + payload.byteLength;
  const buf = new ArrayBuffer(total);
  const view = new DataView(buf);

  // Header
  view.setUint32(0x00, MOEX_MAGIC, false); // big-endian magic
  view.setUint16(0x04, MOEX_VERSION, true);
  view.setUint16(0x06, header.messageType, true);
  view.setUint32(0x08, payload.byteLength, true);
  view.setUint32(0x0c, header.sequenceId, true);

  // Tensor metadata
  view.setUint16(0x10, header.layerId, true);
  view.setUint16(0x12, header.expertId, true);
  view.setUint32(0x14, header.numTokens, true);
  view.setUint16(0x18, header.hiddenDim, true);
  view.setUint8(0x1a, header.dtype);
  view.setUint8(0x1b, header.flags);

  // Payload
  new Uint8Array(buf, HEADER_SIZE).set(new Uint8Array(payload));

  return buf;
}

// ── Decoder ────────────────────────────────────────────────────────────

export function decodeFrameHeader(buf: ArrayBuffer): FrameHeader {
  if (buf.byteLength < HEADER_SIZE) {
    throw new Error(
      `Frame too small: ${buf.byteLength} < ${HEADER_SIZE}`,
    );
  }

  const view = new DataView(buf);
  const magic = view.getUint32(0x00, false);

  if (magic !== MOEX_MAGIC) {
    throw new Error(
      `Invalid magic: 0x${magic.toString(16)} (expected 0x${MOEX_MAGIC.toString(16)})`,
    );
  }

  return {
    magic,
    version: view.getUint16(0x04, true),
    messageType: view.getUint16(0x06, true) as MessageType,
    payloadLength: view.getUint32(0x08, true),
    sequenceId: view.getUint32(0x0c, true),
    layerId: view.getUint16(0x10, true),
    expertId: view.getUint16(0x12, true),
    numTokens: view.getUint32(0x14, true),
    hiddenDim: view.getUint16(0x18, true),
    dtype: view.getUint8(0x1a) as DType,
    flags: view.getUint8(0x1b),
  };
}

export function decodeFramePayload(buf: ArrayBuffer): ArrayBuffer {
  return buf.slice(HEADER_SIZE);
}

// ── Utility ────────────────────────────────────────────────────────────

/** Compute payload size in bytes for a tensor */
export function tensorPayloadSize(
  numTokens: number,
  hiddenDim: number,
  dtype: DType,
): number {
  return Math.ceil(numTokens * hiddenDim * DTYPE_BYTES[dtype]);
}

/** Build a DISPATCH frame for a single expert activation */
export function buildDispatchFrame(
  sequenceId: number,
  layerId: number,
  expertId: number,
  hiddenStates: Float32Array, // [numTokens, hiddenDim]
  numTokens: number,
  hiddenDim: number,
  gateWeight: number,
): ArrayBuffer {
  // Convert float32 hidden states to float16 payload
  const f16 = float32ToFloat16(hiddenStates);

  // Append gate weight (single f16 value) to payload
  const gateF16 = float32ToFloat16(new Float32Array([gateWeight]));
  const payload = new Uint8Array(f16.byteLength + gateF16.byteLength);
  payload.set(new Uint8Array(f16), 0);
  payload.set(new Uint8Array(gateF16), f16.byteLength);

  return encodeFrame(
    {
      messageType: MessageType.DISPATCH,
      sequenceId,
      layerId,
      expertId,
      numTokens,
      hiddenDim,
      dtype: DType.F16,
      flags: 0,
    },
    payload.buffer,
  );
}

/** Build a CANCEL frame */
export function buildCancelFrame(
  sequenceId: number,
  layerId: number,
  expertId: number,
): ArrayBuffer {
  return encodeFrame(
    {
      messageType: MessageType.CANCEL,
      sequenceId,
      layerId,
      expertId,
      numTokens: 0,
      hiddenDim: 0,
      dtype: DType.F16,
      flags: 0,
    },
    new ArrayBuffer(0),
  );
}

/** Build a HEARTBEAT frame */
export function buildHeartbeatFrame(): ArrayBuffer {
  return encodeFrame(
    {
      messageType: MessageType.HEARTBEAT,
      sequenceId: 0,
      layerId: 0,
      expertId: 0,
      numTokens: 0,
      hiddenDim: 0,
      dtype: DType.F16,
      flags: 0,
    },
    new ArrayBuffer(0),
  );
}

// ── Float16 conversion (simplified) ───────────────────────────────────

function float32ToFloat16(input: Float32Array): ArrayBuffer {
  const output = new Uint16Array(input.length);
  for (let i = 0; i < input.length; i++) {
    output[i] = toFloat16(input[i]);
  }
  return output.buffer;
}

function toFloat16(val: number): number {
  const f32 = new Float32Array([val]);
  const u32 = new Uint32Array(f32.buffer)[0];

  const sign = (u32 >> 16) & 0x8000;
  let exponent = ((u32 >> 23) & 0xff) - 127 + 15;
  let mantissa = (u32 >> 13) & 0x3ff;

  if (exponent <= 0) {
    return sign; // underflow to zero
  } else if (exponent >= 31) {
    return sign | 0x7c00; // overflow to infinity
  }

  return sign | (exponent << 10) | mantissa;
}
