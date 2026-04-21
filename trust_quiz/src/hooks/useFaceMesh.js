/**
 * useFaceMesh – custom React hook
 *
 * Responsibilities:
 *  1. Request camera access and feed frames into MediaPipe FaceMesh.
 *  2. Extract nose-tip + eye landmarks → compute horizontal yaw.
 *  3. Apply LEFT / CENTER / RIGHT classification with a 300 ms debounce
 *     so a fleeting glance doesn't accidentally select an answer.
 *  4. Draw a lightweight face-mesh skeleton overlay onto a <canvas>.
 *  5. Expose direction, holdProgress (0–1), faceDetected, and error state.
 *
 * Coordinate system note
 * ----------------------
 * MediaPipe returns landmarks in RAW (un-mirrored) image space where x=0
 * is the LEFT edge of the physical frame.  The video element is displayed
 * mirrored via CSS (transform: scaleX(-1)) so it looks like a selfie mirror.
 *
 * Consequence: when the user tilts their head to THEIR left, their nose moves
 * to the right in raw image space → rawYaw > 0.
 * We negate rawYaw before applying thresholds so that positive yaw = the user's
 * RIGHT tilt and negative yaw = the user's LEFT tilt, matching the visual.
 */

import { useEffect, useRef, useState, useCallback } from 'react';

// MediaPipe is loaded as a global via <script> tags in index.html.
// We access them as window.FaceMesh / window.Camera to avoid Vite trying to
// bundle the WASM-heavy CJS packages (which always breaks in ESM environments).
/* global FaceMesh, Camera */

// ─── Constants ────────────────────────────────────────────────────────────────

const DEBOUNCE_MS = 1000;       // ms a pose must be held before confirming
const YAW_THRESHOLD = 0.27;   // fraction of face width to trigger L/R (higher = less sensitive)
const PROGRESS_INTERVAL_MS = 16; // ~60 fps progress updates

// Nod detection constants
const NOD_DOWN_DIST   = 0.025; // nose.y must drop this far below baseline to register nod-down
const NOD_RETURN_DIST = 0.008; // nose.y must return this close to baseline to confirm nod
const NOD_TIMEOUT_MS  = 1800;  // max ms allowed between nod-down and nod-up

// MediaPipe FaceMesh 468-landmark indices for key points
const LM_NOSE_TIP      = 4;   // tip of nose
const LM_LEFT_EYE_IN   = 133; // inner corner of user's LEFT eye  (camera right)
const LM_RIGHT_EYE_IN  = 362; // inner corner of user's RIGHT eye (camera left)

// CDN base – keeps Vite from having to bundle the WASM/model files
const MEDIAPIPE_CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4';

// ─── Helper: draw mesh overlay ────────────────────────────────────────────────

/**
 * Render a subtle point-cloud face skeleton on the provided canvas element.
 * The canvas sits on top of the video (transparent background) so we only
 * need to draw dots; the video element itself shows the webcam feed.
 *
 * @param {HTMLCanvasElement} canvas
 * @param {Array|null}        landmarks  – normalised [0,1] landmarks or null
 * @param {number}            w          – image width in px
 * @param {number}            h          – image height in px
 */
function drawOverlay(canvas, landmarks, w, h) {
  const ctx = canvas.getContext('2d');

  // Keep canvas dimensions in sync with the video resolution
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width  = w;
    canvas.height = h;
  }

  // Clear canvas only — skeleton drawing intentionally omitted
  ctx.clearRect(0, 0, w, h);
}

// ─── Hook ─────────────────────────────────────────────────────────────────────

/**
 * @param {object}   options
 * @param {React.RefObject<HTMLVideoElement>}  options.videoRef
 * @param {React.RefObject<HTMLCanvasElement>} options.canvasRef
 * @param {(dir: 'LEFT'|'RIGHT') => void}      options.onDirectionConfirmed
 *   Called exactly once when the user holds a non-centre pose for DEBOUNCE_MS.
 *   Will NOT be called again until the user returns to CENTRE and tilts again.
 * @param {boolean} [options.enabled=true]
 *   Set false to pause detection (e.g., between questions or on result screen).
 *
 * @returns {{ direction: string, holdProgress: number, faceDetected: boolean, error: string|null }}
 */
export function useFaceMesh({ videoRef, canvasRef, onDirectionConfirmed, onNodConfirmed, enabled = true }) {
  const [direction,    setDirection]    = useState('CENTER');
  const [holdProgress, setHoldProgress] = useState(0);   // 0–1
  const [faceDetected, setFaceDetected] = useState(false);
  const [error,        setError]        = useState(null);

  // Stable ref for the callbacks so the MediaPipe closure always calls the
  // latest version without triggering a full re-initialisation.
  const onConfirmedRef    = useRef(onDirectionConfirmed);
  const onNodConfirmedRef = useRef(onNodConfirmed);
  useEffect(() => { onConfirmedRef.current    = onDirectionConfirmed; }, [onDirectionConfirmed]);
  useEffect(() => { onNodConfirmedRef.current = onNodConfirmed;       }, [onNodConfirmed]);

  // Refs for debounce state (mutable, no re-render on change)
  const holdTimerRef     = useRef(null);
  const progressTimerRef = useRef(null);
  const holdStartRef     = useRef(null);
  const pendingDirRef    = useRef('CENTER'); // direction currently being held
  const lockedRef        = useRef(false);    // true while waiting for user to return to CENTER

  // Refs for nod detection state machine
  const nodStateRef    = useRef('idle');  // 'idle' | 'peaked'
  const nodBaselineRef = useRef(null);   // slow-EMA of nose.y (null until first face)
  const nodTimerRef    = useRef(null);   // timeout to reset peaked state

  // ── Debounce helpers ────────────────────────────────────────────────────────

  const cancelHold = useCallback(() => {
    clearTimeout(holdTimerRef.current);
    clearInterval(progressTimerRef.current);
    holdTimerRef.current     = null;
    progressTimerRef.current = null;
    holdStartRef.current     = null;
    setHoldProgress(0);
  }, []);

  const startHold = useCallback((dir) => {
    cancelHold();
    holdStartRef.current = Date.now();

    // Tick progress at ~60 fps
    progressTimerRef.current = setInterval(() => {
      const elapsed  = Date.now() - holdStartRef.current;
      const progress = Math.min(elapsed / DEBOUNCE_MS, 1);
      setHoldProgress(progress);
    }, PROGRESS_INTERVAL_MS);

    // Fire after full debounce period
    holdTimerRef.current = setTimeout(() => {
      clearInterval(progressTimerRef.current);
      progressTimerRef.current = null;
      setHoldProgress(1);
      lockedRef.current = true;   // prevent re-trigger until user resets
      onConfirmedRef.current?.(dir);
    }, DEBOUNCE_MS);
  }, [cancelHold]);

  // ── Yaw calculation ─────────────────────────────────────────────────────────

  /**
   * Returns the head-tilt (roll) signal only — eye-slope in "face-width" units.
   * Turning/facing left or right is intentionally ignored so the user can look
   * freely at the panels without accidentally selecting an answer.
   *
   *   < 0  → user tilted LEFT  (left ear toward left shoulder)
   *   > 0  → user tilted RIGHT
   *   0    → level
   */
  const computeYaw = useCallback((landmarks) => {
    const lEye = landmarks[LM_LEFT_EYE_IN];
    const rEye = landmarks[LM_RIGHT_EYE_IN];

    if (!lEye || !rEye) return 0;

    const faceWidth = Math.abs(rEye.x - lEye.x);
    if (faceWidth < 0.01) return 0;

    // Slope of the inner-eye-corner line.
    // When user tilts LEFT: lEye drops (y↑), rEye rises (y↓) → result < 0 → LEFT ✓
    return (rEye.y - lEye.y) / faceWidth;
  }, []);

  // ── MediaPipe initialisation ─────────────────────────────────────────────────

  useEffect(() => {
    if (!enabled) {
      cancelHold();
      return;
    }

    const videoEl  = videoRef.current;
    const canvasEl = canvasRef.current;
    if (!videoEl || !canvasEl) return;

    // --- FaceMesh setup ---
    const faceMesh = new FaceMesh({
      // Point to the CDN so Vite doesn't need to handle the WASM/model files.
      locateFile: (file) => `${MEDIAPIPE_CDN}/${file}`,
    });

    faceMesh.setOptions({
      maxNumFaces:           1,      // we only care about the primary user
      refineLandmarks:       false,  // skip iris refine — saves ~5 ms/frame
      minDetectionConfidence: 0.5,
      minTrackingConfidence:  0.5,
    });

    faceMesh.onResults((results) => {
      const { multiFaceLandmarks, image } = results;
      const imgW = image.width  || 320;
      const imgH = image.height || 240;

      // ── No face detected ──────────────────────────────────────────────────
      if (!multiFaceLandmarks || multiFaceLandmarks.length === 0) {
        setFaceDetected(false);
        setDirection('CENTER');
        pendingDirRef.current = 'CENTER';

        if (!lockedRef.current) cancelHold();

        drawOverlay(canvasEl, null, imgW, imgH);
        return;
      }

      // ── Face detected ─────────────────────────────────────────────────────
      setFaceDetected(true);
      const landmarks = multiFaceLandmarks[0];
      drawOverlay(canvasEl, landmarks, imgW, imgH);

      const yaw = computeYaw(landmarks);

      let newDir = 'CENTER';
      if      (yaw >  YAW_THRESHOLD) newDir = 'LEFT';
      else if (yaw < -YAW_THRESHOLD) newDir = 'RIGHT';

      setDirection(newDir);

      // If we're locked (waiting for user to return to center), only unlock
      // when they come back to CENTER.
      if (lockedRef.current) {
        if (newDir === 'CENTER') lockedRef.current = false;
        return;
      }

      // Direction changed → manage hold timer
      if (newDir !== pendingDirRef.current) {
        pendingDirRef.current = newDir;

        if (newDir === 'CENTER') {
          cancelHold();
        } else {
          startHold(newDir);
        }
      }

      // ── Nod detection ──────────────────────────────────────────────────
      // Only runs when a nod callback is registered (e.g. during explaining state).
      if (onNodConfirmedRef.current) {
        const noseY = landmarks[LM_NOSE_TIP].y;

        // Initialise baseline on first face detection
        if (nodBaselineRef.current === null) nodBaselineRef.current = noseY;

        if (nodStateRef.current === 'idle') {
          // Update baseline slowly so short nods don't shift it
          nodBaselineRef.current = nodBaselineRef.current * 0.97 + noseY * 0.03;

          // Nod down: nose moves meaningfully below baseline (higher Y in image coords)
          if (noseY > nodBaselineRef.current + NOD_DOWN_DIST) {
            nodStateRef.current = 'peaked';
            clearTimeout(nodTimerRef.current);
            nodTimerRef.current = setTimeout(() => {
              nodStateRef.current = 'idle'; // nod didn't return in time → reset
            }, NOD_TIMEOUT_MS);
          }
        } else if (nodStateRef.current === 'peaked') {
          // Nod up: nose returns close to baseline → nod confirmed
          if (noseY < nodBaselineRef.current + NOD_RETURN_DIST) {
            clearTimeout(nodTimerRef.current);
            nodStateRef.current = 'idle';
            onNodConfirmedRef.current?.();
          }
        }
      }
    });

    // --- Camera setup (320×240 @ ≤30 fps) ---
    const camera = new Camera(videoEl, {
      onFrame: async () => {
        try {
          await faceMesh.send({ image: videoEl });
        } catch {
          // Frame send can throw if the element is detached; ignore silently.
        }
      },
      width:  320,
      height: 240,
    });

    camera.start().catch((err) => {
      setError(`Camera error: ${err.message || 'Permission denied or device unavailable'}`);
    });

    // Cleanup: stop camera and close FaceMesh when effect re-runs or unmounts
    return () => {
      camera.stop();
      faceMesh.close();
      cancelHold();
      lockedRef.current     = false;
      pendingDirRef.current = 'CENTER';
      clearTimeout(nodTimerRef.current);
      nodStateRef.current    = 'idle';
      nodBaselineRef.current = null;
    };
  }, [enabled, videoRef, canvasRef, computeYaw, startHold, cancelHold]);

  return { direction, holdProgress, faceDetected, error };
}
