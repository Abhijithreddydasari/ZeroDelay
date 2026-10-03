"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { Decision, MicPermission, SessionState, VoicePhase } from "./types";
import {
  ApiError,
  converse,
  converseStream,
  type ConverseResponse,
  type RetrievedDiagram,
} from "./api";
import { base64ToArrayBuffer, encodeWav, mergeChunks } from "./audio";

export type { VoicePhase };

// Real voice loop wiring the ZeroDelay backend into the app shell:
//   listen (record mic PCM) -> WAV -> POST /converse/stream (ASR + session + TTS)
//   -> play validated audio chunks -> back to listening.
// The same Web Audio amplitude drives the VoiceVisual orb for both the human's
// turn (mic input) and the AI's turn (TTS playback).

// Voice-activity thresholds (RMS 0..1). A turn = speech, then a pause.
const VOICE_THRESHOLD = 0.08;
const SPEECH_MIN_MS = 300; // ignore coughs / short blips
const SILENCE_CONFIRM_MS = 900; // pause that ends the technician's turn

type Options = {
  active: boolean;
  sessionId: string | null;
  onDecision?: (
    decision: Decision,
    query: string,
    diagram?: RetrievedDiagram | null,
    session?: SessionState
  ) => void;
};

export function useVoiceLoop({ active, sessionId, onDecision }: Options) {
  const [permission, setPermission] = useState<MicPermission>("idle");
  const [phase, setPhase] = useState<VoicePhase>("idle");
  const [amplitude, setAmplitude] = useState(0);
  const [transcript, setTranscript] = useState<string | null>(null);
  const [partial, setPartial] = useState(""); // spoken answer streaming in, live
  const [decision, setDecision] = useState<Decision | null>(null);
  const [error, setError] = useState<string | null>(null);

  const phaseRef = useRef<VoicePhase>("idle");
  const setPhaseSafe = useCallback((p: VoicePhase) => {
    phaseRef.current = p;
    setPhase(p);
  }, []);

  const streamRef = useRef<MediaStream | null>(null);
  const ctxRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sampleRateRef = useRef(48000);

  // Capture / VAD state (refs so the audio callback stays cheap).
  const capturingRef = useRef(false);
  const chunksRef = useRef<Float32Array[]>([]);
  const speechStartRef = useRef<number | null>(null);
  const silenceStartRef = useRef<number | null>(null);
  const hasSpokenRef = useRef(false);
  const busyRef = useRef(false);

  const onDecisionRef = useRef(onDecision);
  onDecisionRef.current = onDecision;

  const armCapture = useCallback(() => {
    capturingRef.current = false;
    chunksRef.current = [];
    speechStartRef.current = null;
    silenceStartRef.current = null;
    hasSpokenRef.current = false;
  }, []);

  const playWav = useCallback(
    async (b64: string) => {
      const ctx = ctxRef.current;
      if (!ctx) return;
      setPhaseSafe("speaking");
      if (ctx.state === "suspended") await ctx.resume().catch(() => {});

      const audioBuffer = await ctx.decodeAudioData(base64ToArrayBuffer(b64));
      const source = ctx.createBufferSource();
      source.buffer = audioBuffer;
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 512;
      source.connect(analyser);
      analyser.connect(ctx.destination);

      const data = new Uint8Array(analyser.frequencyBinCount);
      let raf = 0;
      const tick = () => {
        analyser.getByteTimeDomainData(data);
        let sum = 0;
        for (let i = 0; i < data.length; i++) {
          const v = (data[i] - 128) / 128;
          sum += v * v;
        }
        setAmplitude(Math.min(1, Math.sqrt(sum / data.length) * 4));
        raf = requestAnimationFrame(tick);
      };
      tick();

      await new Promise<void>((resolve) => {
        source.onended = () => resolve();
        source.start();
      });

      cancelAnimationFrame(raf);
      source.disconnect();
      analyser.disconnect();
      setAmplitude(0);
    },
    [setPhaseSafe]
  );

  const handleUtterance = useCallback(
    async (wav: Blob) => {
      if (busyRef.current || !sessionId) return;
      busyRef.current = true;
      setError(null);
      setPartial("");
      setPhaseSafe("thinking");
      setAmplitude(0.16); // gentle "alive" pulse while the model thinks

      let playback = Promise.resolve();
      try {
        // Receive validated audio, then apply the backend's final session state.
        let res: ConverseResponse;
        const turnId = crypto.randomUUID();
        let audioPlayed = false;
        try {
          res = await converseStream(wav, sessionId, turnId, {
            onQuery: (q) => setTranscript(q),
            onDelta: (t) => setPartial((prev) => prev + t),
            onAudioChunk: (chunk) => {
              audioPlayed = true;
              playback = playback.then(() => playWav(chunk));
            },
          });
        } catch (e) {
          // Retry only before any stream event. The same turn ID makes replay safe.
          if (e instanceof ApiError && (e.status === 422 || e.started)) throw e;
          setPartial("");
          res = await converse(wav, sessionId, turnId);
          setTranscript(res.query);
        }
        // The final decision carries the authoritative spoken text, so drop the
        // live preview and let the parent render the completed AI message.
        setPartial("");
        setDecision(res.decision);
        // Surface the most relevant retrieved diagram that actually has an image
        // (matches the one the backend feeds its vision model).
        const diagram =
          res.retrieval?.diagrams?.find((d) => d.image_exists) ?? null;
        onDecisionRef.current?.(res.decision, res.query, diagram, res.session);
        if (!audioPlayed && res.audio_chunks_base64?.length) {
          for (const chunk of res.audio_chunks_base64) playback = playback.then(() => playWav(chunk));
        } else if (!audioPlayed && res.tts_wav_base64) {
          playback = playback.then(() => playWav(res.tts_wav_base64!));
        }
        await playback;
      } catch (e) {
        // Keep the mic disarmed until any already received audio has finished.
        await playback.catch(() => {});
        if (e instanceof ApiError && e.status === 422) {
          setError("Didn't catch that — please try again.");
        } else if (e instanceof ApiError) {
          setError(`Backend error: ${e.message}`);
        } else {
          setError("Can't reach the ZeroDelay backend on port 8000.");
        }
      } finally {
        busyRef.current = false;
        setPartial("");
        armCapture();
        // Return to listening only if the session is still open.
        if (ctxRef.current) setPhaseSafe("listening");
        else setPhaseSafe("idle");
        setAmplitude(0);
      }
    },
    [armCapture, playWav, sessionId, setPhaseSafe]
  );

  useEffect(() => {
    if (!active) {
      setPermission("idle");
      setPhaseSafe("idle");
      setAmplitude(0);
      return;
    }

    let cancelled = false;

    (async () => {
      setPermission("pending");
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        setPermission("granted");

        const ctx = new AudioContext();
        ctxRef.current = ctx;
        sampleRateRef.current = ctx.sampleRate;
        if (ctx.state === "suspended") await ctx.resume().catch(() => {});

        const source = ctx.createMediaStreamSource(stream);
        const processor = ctx.createScriptProcessor(4096, 1, 1);
        processorRef.current = processor;
        source.connect(processor);
        // Route to a muted sink: some browsers only fire onaudioprocess when the
        // node reaches the destination, but we must NOT play the mic back.
        const mute = ctx.createGain();
        mute.gain.value = 0;
        processor.connect(mute);
        mute.connect(ctx.destination);

        armCapture();
        setPhaseSafe("listening");

        processor.onaudioprocess = (e: AudioProcessingEvent) => {
          const input = e.inputBuffer.getChannelData(0);

          let sumSquares = 0;
          for (let i = 0; i < input.length; i++) sumSquares += input[i] * input[i];
          const level = Math.min(1, Math.sqrt(sumSquares / input.length) * 5);

          // Only the human's turn drives the orb + VAD; the AI's turn is handled
          // by playWav so the model doesn't "hear itself" through the speakers.
          if (phaseRef.current !== "listening" || busyRef.current) return;
          setAmplitude(level);

          const now = performance.now();
          if (level > VOICE_THRESHOLD) {
            silenceStartRef.current = null;
            if (speechStartRef.current === null) speechStartRef.current = now;
            if (now - speechStartRef.current > SPEECH_MIN_MS) hasSpokenRef.current = true;
            capturingRef.current = true;
          } else if (capturingRef.current) {
            speechStartRef.current = null;
            if (!hasSpokenRef.current) {
              // Noise blip that never became real speech — discard it.
              capturingRef.current = false;
              chunksRef.current = [];
            } else {
              if (silenceStartRef.current === null) silenceStartRef.current = now;
              if (now - silenceStartRef.current > SILENCE_CONFIRM_MS) {
                const samples = mergeChunks(chunksRef.current);
                capturingRef.current = false;
                chunksRef.current = [];
                const wav = encodeWav(samples, sampleRateRef.current);
                void handleUtterance(wav);
                return;
              }
            }
          }

          if (capturingRef.current) chunksRef.current.push(new Float32Array(input));
        };
      } catch {
        if (!cancelled) setPermission("denied");
      }
    })();

    return () => {
      cancelled = true;
      if (processorRef.current) processorRef.current.onaudioprocess = null;
      processorRef.current?.disconnect();
      streamRef.current?.getTracks().forEach((t) => t.stop());
      ctxRef.current?.close().catch(() => {});
      processorRef.current = null;
      streamRef.current = null;
      ctxRef.current = null;
      busyRef.current = false;
      setAmplitude(0);
      setPhaseSafe("idle");
    };
  }, [active, armCapture, handleUtterance, sessionId, setPhaseSafe]);

  return { permission, phase, amplitude, transcript, partial, decision, error };
}
