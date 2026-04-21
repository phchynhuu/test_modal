/**
 * QuizEngine
 *
 * Full-screen layout:
 *
 *  ┌──────────────────────────────────────────────┐
 *  │  progress dots · question text (top bar)     │
 *  ├──────────────────┬───────────────────────────┤
 *  │                  │                           │
 *  │  LEFT PANEL      │       RIGHT PANEL         │
 *  │  (large, tall)   │       (large, tall)       │
 *  │                  │                           │
 *  │         [○ 150px camera bubble]              │
 *  ├──────────────────┴───────────────────────────┤
 *  │  [Status Left]         [Status Right]        │
 *  └──────────────────────────────────────────────┘
 *
 * State machine: idle → asking → confirmed → results
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useFaceMesh } from '../hooks/useFaceMesh';
import QUESTIONS from '../data/questions.js';

const CONFIRM_PAUSE_MS  = 900;
const QUESTION_TIME_SEC = 30;

export default function QuizEngine({ videoRef, canvasRef }) {
  const [quizStatus,  setQuizStatus]  = useState('idle');
  const [questionIdx, setQuestionIdx] = useState(0);
  const [score,       setScore]       = useState(0);
  const [lastAnswer,  setLastAnswer]  = useState(null);
  const [lastCorrect, setLastCorrect] = useState(null);
  const [timeLeft,    setTimeLeft]    = useState(QUESTION_TIME_SEC);

  const questionIdxRef = useRef(questionIdx);
  const quizStatusRef  = useRef(quizStatus);
  useEffect(() => { questionIdxRef.current = questionIdx; }, [questionIdx]);
  useEffect(() => { quizStatusRef.current  = quizStatus;  }, [quizStatus]);

  const detectionActive = quizStatus !== 'results';

  const advanceQuestion = useCallback((idx) => {
    const next = idx + 1;
    if (next >= QUESTIONS.length) {
      setQuizStatus('results');
    } else {
      setQuestionIdx(next);
      setLastAnswer(null);
      setLastCorrect(null);
      setTimeLeft(QUESTION_TIME_SEC);
      setQuizStatus('asking');
    }
  }, []);

  const showExplanationOrAdvance = useCallback((idx) => {
    if (QUESTIONS[idx]?.explanation) {
      setQuizStatus('explaining');
    } else {
      advanceQuestion(idx);
    }
  }, [advanceQuestion]);

  const handleDirectionConfirmed = useCallback((dir) => {
    if (quizStatusRef.current !== 'asking') return;
    const idx      = questionIdxRef.current;
    const question = QUESTIONS[idx];
    const side     = dir === 'LEFT' ? 'left' : 'right';
    const correct  = side === question.correct;
    setLastAnswer(side);
    setLastCorrect(correct);
    if (correct) setScore((s) => s + 1);
    setQuizStatus('confirmed');
    setTimeout(() => showExplanationOrAdvance(idx), CONFIRM_PAUSE_MS);
  }, [showExplanationOrAdvance]);

  const handleNodConfirmed = useCallback(() => {
    if (quizStatusRef.current !== 'explaining') return;
    advanceQuestion(questionIdxRef.current);
  }, [advanceQuestion]);

  useEffect(() => {
    if (quizStatus !== 'asking') return;
    setTimeLeft(QUESTION_TIME_SEC);
    const interval = setInterval(() => {
      setTimeLeft((t) => {
        if (t <= 1) {
          clearInterval(interval);
          setQuizStatus('confirmed');
          setLastAnswer(null);
          setLastCorrect(null);
          setTimeout(() => showExplanationOrAdvance(questionIdxRef.current), CONFIRM_PAUSE_MS);
          return 0;
        }
        return t - 1;
      });
    }, 1000);
    return () => clearInterval(interval);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [quizStatus, questionIdx, showExplanationOrAdvance]);

  const { direction, holdProgress, faceDetected, error } = useFaceMesh({
    videoRef,
    canvasRef,
    onDirectionConfirmed: handleDirectionConfirmed,
    onNodConfirmed: quizStatus === 'explaining' ? handleNodConfirmed : undefined,
    enabled: detectionActive,
  });

  useEffect(() => {
    if (quizStatus === 'idle' && faceDetected) setQuizStatus('asking');
  }, [faceDetected, quizStatus]);

  const currentQuestion = QUESTIONS[questionIdx];
  const isAsking      = quizStatus === 'asking';
  const isConfirmed   = quizStatus === 'confirmed';
  const isExplaining  = quizStatus === 'explaining';
  const isResults     = quizStatus === 'results';
  const isIdle        = quizStatus === 'idle';
  const activeDir     = isAsking ? direction : null;
  const isSingle    = currentQuestion?.layout === 'single';
  const isDualAudio = !isSingle
    && currentQuestion?.options?.left?.type  === 'audio'
    && currentQuestion?.options?.right?.type === 'audio';

  // Sequential audio: left plays first, switches to right when done, then loops
  const [audioPlayingSide, setAudioPlayingSide] = useState('left');
  useEffect(() => { setAudioPlayingSide('left'); }, [questionIdx]);
  const handleAudioEnded = useCallback((side) => {
    setAudioPlayingSide(side === 'left' ? 'right' : 'left');
  }, []);

  const handleRestart = () => {
    setQuestionIdx(0);
    setScore(0);
    setLastAnswer(null);
    setLastCorrect(null);
    setQuizStatus('idle');
  };

  return (
    <div className="quiz-layout">

      {/* ── Camera feed — full-screen background ─────────────────── */}
      <video
        ref={videoRef}
        className="camera-bg-video"
        playsInline
        muted
        autoPlay
        aria-hidden="true"
      />
      <canvas
        ref={canvasRef}
        className="camera-bg-canvas"
        aria-hidden="true"
      />
      <div className="camera-bg-overlay" aria-hidden="true" />

      {/* ── Face-not-detected corner badge ───────────────────────── */}
      {isAsking && !faceDetected && (
        <div className="camera-face-warning" title="Face not detected">!</div>
      )}

      {/* ── Error banner ─────────────────────────────────────────── */}
      {error && (
        <div className="error-banner" role="alert">
          <span>⚠ {error}</span>
          <p className="error-hint">Allow camera access and refresh.</p>
        </div>
      )}

      {/* ── Top bar: progress + question ─────────────────────────── */}
      <div className="quiz-top">
        {(isAsking || isConfirmed) && (
          <>
            <div className="progress-dots" aria-label="Question progress">
              {QUESTIONS.map((_, i) => (
                <span
                  key={i}
                  className={`dot ${i < questionIdx ? 'done' : i === questionIdx ? 'current' : ''}`}
                />
              ))}
            </div>
            <div className="question-box">
              <div className="question-header">
                <p className="question-label">Question {questionIdx + 1}/{QUESTIONS.length}</p>
                <CountdownRing
                  timeLeft={timeLeft}
                  total={QUESTION_TIME_SEC}
                  urgent={timeLeft <= 3}
                />
              </div>
              <h2 className="question-text">{currentQuestion.question}</h2>
            </div>
          </>
        )}
      </div>

      {/* ── Main area: two large panels + camera bubble ──────────── */}
      <div className="panels-row">

        {/* ── Single-media layout (real vs fake) ───────────────── */}
        {isSingle ? (
          <SinglePanel
            question={currentQuestion}
            activeDir={activeDir}
            confirmed={isConfirmed || isExplaining}
            lastAnswer={lastAnswer}
            correct={lastCorrect}
            show={isAsking || isConfirmed || isExplaining}
          />
        ) : (
          <>
            {/* Left panel */}
            <LargePanel
              side="left"
              question={currentQuestion}
              active={activeDir === 'LEFT'}
              confirmed={(isConfirmed || isExplaining) && lastAnswer === 'left'}
              correct={(isConfirmed || isExplaining) && lastAnswer === 'left' ? lastCorrect : null}
              dimmed={(isConfirmed || isExplaining) && lastAnswer === 'right'}
              show={isAsking || isConfirmed || isExplaining}
              shouldPlay={isDualAudio ? audioPlayingSide === 'left' : undefined}
              onAudioEnded={isDualAudio ? handleAudioEnded : undefined}
            />

            {/* Right panel */}
            <LargePanel
              side="right"
              question={currentQuestion}
              active={activeDir === 'RIGHT'}
              confirmed={(isConfirmed || isExplaining) && lastAnswer === 'right'}
              correct={(isConfirmed || isExplaining) && lastAnswer === 'right' ? lastCorrect : null}
              dimmed={(isConfirmed || isExplaining) && lastAnswer === 'left'}
              show={isAsking || isConfirmed || isExplaining}
              shouldPlay={isDualAudio ? audioPlayingSide === 'right' : undefined}
              onAudioEnded={isDualAudio ? handleAudioEnded : undefined}
            />
          </>
        )}

        {/* ── Explanation overlay (slides up after answer confirmed) ── */}
        {isExplaining && currentQuestion?.explanation && (
          <ExplanationPanel
            question={currentQuestion}
            isCorrect={lastCorrect}
            onContinue={() => advanceQuestion(questionIdxRef.current)}
          />
        )}

        {/* Idle overlay covers both panels */}
        {isIdle && !error && (
          <div className="panels-idle-overlay">
            <div className="spinner" aria-label="Loading" />
            <p className="status-text">
              {faceDetected ? 'Calibrating…' : 'Looking for your face…'}
            </p>
            <p className="status-hint">Position your face in the camera</p>
          </div>
        )}

        {/* Results — full-screen overlay rendered at quiz-layout level below */}
      </div>

      {/* ── Results screen — full viewport overlay ───────────────── */}
      {isResults && (
        <ResultsScreen
          score={score}
          total={QUESTIONS.length}
          onRestart={handleRestart}
        />
      )}

      {/* ── Status row: tilt-to-answer indicators ────────────────── */}
      <div className={`status-row${isExplaining ? ' status-row--hidden' : ''}`}>
        <StatusBox
          side="left"
          questionType={currentQuestion?.options?.left?.type || 'text'}
          hintLabel={isSingle ? currentQuestion?.options?.left?.label : null}
          active={activeDir === 'LEFT'}
          holdProgress={activeDir === 'LEFT' ? holdProgress : 0}
          confirmed={isConfirmed && lastAnswer === 'left'}
          correct={isConfirmed && lastAnswer === 'left' ? lastCorrect : null}
          dimmed={isConfirmed && lastAnswer === 'right'}
        />
        <StatusBox
          side="right"
          questionType={currentQuestion?.options?.right?.type || 'text'}
          hintLabel={isSingle ? currentQuestion?.options?.right?.label : null}
          active={activeDir === 'RIGHT'}
          holdProgress={activeDir === 'RIGHT' ? holdProgress : 0}
          confirmed={isConfirmed && lastAnswer === 'right'}
          correct={isConfirmed && lastAnswer === 'right' ? lastCorrect : null}
          dimmed={isConfirmed && lastAnswer === 'left'}
        />
      </div>

    </div>
  );
}

// ─── CountdownRing ────────────────────────────────────────────────────────────

function CountdownRing({ timeLeft, total, urgent }) {
  const r = 16;
  const circ = 2 * Math.PI * r;
  const dash = circ * (timeLeft / total);
  return (
    <div className={`countdown-ring ${urgent ? 'countdown-ring--urgent' : ''}`} aria-label={`${timeLeft}s left`}>
      <svg width="44" height="44" viewBox="0 0 44 44">
        <circle cx="22" cy="22" r={r} fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="3" />
        <circle
          cx="22" cy="22" r={r}
          fill="none"
          stroke={urgent ? 'var(--clr-wrong)' : 'var(--clr-accent)'}
          strokeWidth="3"
          strokeDasharray={`${dash} ${circ}`}
          strokeLinecap="round"
          transform="rotate(-90 22 22)"
          style={{ transition: 'stroke-dasharray 0.9s linear, stroke 0.3s' }}
        />
      </svg>
      <span className="countdown-number">{timeLeft}</span>
    </div>
  );
}

// ─── SinglePanel ─────────────────────────────────────────────────────────────

/**
 * Full-width panel for "is this real or fake?" questions.
 * Shows one piece of media; tilt left = question.left, tilt right = question.right.
 *
 * Question shape:
 *   layout:    'single'
 *   mediaType: 'image' | 'video' | 'audio'
 *   mediaSrc:  URL
 *   left:      label for left tilt  (e.g. 'Thật')
 *   right:     label for right tilt (e.g. 'Giả')
 *   correct:   'left' | 'right'
 */
function SinglePanel({ question, activeDir, confirmed, lastAnswer, correct, show }) {
  const audioRef = useRef(null);

  useEffect(() => {
    const el = audioRef.current;
    if (!el) return;
    el.currentTime = 0;
    el.play().catch(() => {});
  }, [question]);

  if (!show || !question) return null;

  const mediaType = question.media?.type;
  const mediaSrc  = question.media?.src;
  const left      = question.options?.left?.label  ?? 'Thật';
  const right     = question.options?.right?.label ?? 'Giả';

  let cls = 'single-panel';
  if (confirmed && correct === true)  cls += ' single-panel--correct';
  if (confirmed && correct === false) cls += ' single-panel--wrong';

  const tiltingLeft  = !confirmed && activeDir === 'LEFT';
  const tiltingRight = !confirmed && activeDir === 'RIGHT';

  return (
    <div className={cls}>

      {/* ── Media ─────────────────────────────────────────────── */}
      <div className="single-panel__media">
        {mediaType === 'image' && (
          <img src={mediaSrc} alt="quiz content" className="panel-media-img" />
        )}
        {mediaType === 'video' && (
          <video src={mediaSrc} className="panel-media-video" autoPlay loop muted playsInline />
        )}
        {mediaType === 'audio' && (
          <div className="panel-audio-placeholder">
            <audio ref={audioRef} src={mediaSrc} loop preload="auto" />
            <div className="audio-speaker">
              <svg viewBox="0 0 24 24" fill="none">
                <path d="M11 5L6 9H2v6h4l5 4V5z" fill="currentColor"/>
                <path d="M15.54 8.46a5 5 0 0 1 0 7.07" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                <path d="M19.07 4.93a10 10 0 0 1 0 14.14" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
              </svg>
            </div>
            <div className="audio-waveform" aria-hidden="true">
              {[0,1,2,3,4,5,6].map(i => (
                <span key={i} className="wave-bar" style={{ animationDelay: `${i * 0.1}s` }} />
              ))}
            </div>
            <p className="audio-label">Nghe và lựa chọn</p>
            <button
              className="audio-replay-btn"
              aria-label="Nghe lại"
              onClick={() => {
                const el = audioRef.current;
                if (!el) return;
                el.currentTime = 0;
                el.play().catch(() => {});
              }}
            >
              ↺ Nghe lại
            </button>
          </div>
        )}
      </div>

      {/* ── Tilt-direction hint overlay ────────────────────────── */}
      {tiltingLeft && (
        <div className="single-panel__hint single-panel__hint--left">
          👈 {left}
        </div>
      )}
      {tiltingRight && (
        <div className="single-panel__hint single-panel__hint--right">
          {right} 👉
        </div>
      )}

      {/* ── Confirmed outcome overlay ──────────────────────────── */}
      {confirmed && correct !== null && (
        <div className="single-panel__outcome">
          <span className="single-panel__outcome-icon">{correct ? '✅' : '❌'}</span>
          <span className="single-panel__outcome-label">
            {correct
              ? (lastAnswer === 'left' ? left : right)
              : (lastAnswer === 'left' ? left : right)}
          </span>
        </div>
      )}

    </div>
  );
}

// ─── LargePanel ──────────────────────────────────────────────────────────────

/**
 * Large tall panel showing the answer content.
 * Renders <h1>, <img>, <video>, or an audio placeholder depending on type.
 *
 * Question shape (extended):
 *   left / right          – label text (always present)
 *   leftType / rightType  – 'text' | 'image' | 'video' | 'audio'  (default: 'text')
 *   leftSrc  / rightSrc   – URL for image, video, or audio
 */
function LargePanel({ side, question, active, confirmed, correct, dimmed, show, shouldPlay, onAudioEnded }) {
  const audioRef = useRef(null);

  useEffect(() => {
    const el = audioRef.current;
    if (!el) return;
    if (shouldPlay === false) {
      // Sequential mode: this panel should be silent right now
      el.pause();
      el.currentTime = 0;
    } else {
      // shouldPlay=true or undefined (single audio / non-sequential): just play
      el.currentTime = 0;
      el.play().catch(() => {});
    }
  }, [shouldPlay, question]);

  if (!show || !question) return <div className={`large-panel large-panel--${side} large-panel--empty`} />;

  const option = question.options?.[side] ?? {};
  const type   = option.type  || 'text';
  const src    = option.src   ?? null;
  const label  = option.label ?? '';
  const badge  = side === 'left' ? 'A' : 'B';

  let cls = `large-panel large-panel--${side}`;
  if (active)                        cls += ' large-panel--active';
  if (confirmed && correct === true)  cls += ' large-panel--correct';
  if (confirmed && correct === false) cls += ' large-panel--wrong';
  if (dimmed)                        cls += ' large-panel--dimmed';

  return (
    <div className={cls} role="option" aria-selected={confirmed}>
      <div className="large-panel__media">

        {type === 'image' && (
          <img src={src} alt={label} className="panel-media-img" />
        )}

        {type === 'video' && (
          <video
            src={src}
            className="panel-media-video"
            autoPlay
            loop
            muted
            playsInline
          />
        )}

        {type === 'audio' && (
          <div className={`panel-audio-placeholder${shouldPlay === false ? ' audio-paused' : ''}`}>
            {/* Hidden audio element — playback controlled via ref */}
            <audio
              ref={audioRef}
              src={src}
              preload="auto"
              onEnded={() => onAudioEnded?.(side)}
            />

            {/* Speaker icon */}
            <div className="audio-speaker">
              <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M11 5L6 9H2v6h4l5 4V5z" fill="currentColor"/>
                <path d="M15.54 8.46a5 5 0 0 1 0 7.07" stroke="currentColor"
                  strokeWidth="2" strokeLinecap="round"/>
                <path d="M19.07 4.93a10 10 0 0 1 0 14.14" stroke="currentColor"
                  strokeWidth="2" strokeLinecap="round"/>
              </svg>
            </div>

            {/* Animated waveform bars */}
            <div className="audio-waveform" aria-hidden="true">
              {[0,1,2,3,4,5,6].map(i => (
                <span key={i} className="wave-bar" style={{ animationDelay: `${i * 0.1}s` }} />
              ))}
            </div>

            <p className="audio-label">{label}</p>
            <button
              className="audio-replay-btn"
              aria-label="Nghe lại"
              onClick={() => {
                const el = audioRef.current;
                if (!el) return;
                el.currentTime = 0;
                el.play().catch(() => {});
              }}
            >
              ↺ Nghe lại
            </button>
          </div>
        )}

        {type === 'text' && (
          <h1 className="panel-media-text">{label}</h1>
        )}

      </div>

      {/* Corner badge */}
      <div className="large-panel__badge">{badge}</div>

      {/* Outcome icon */}
      {confirmed && correct !== null && (
        <div className="large-panel__outcome" aria-label={correct ? 'Correct' : 'Wrong'}>
          {correct ? '✅' : '❌'}
        </div>
      )}
    </div>
  );
}

// ─── ResultsScreen ───────────────────────────────────────────────────────────

function getTopMessage(score, total) {
  const pct = score / total;
  if (pct === 1)   return 'Bạn thuộc top 1%\nngười kháng lừa đảo cao nhất';
  if (pct >= 0.8)  return 'Bạn thuộc top 5%\nngười kháng lừa đảo cao nhất';
  if (pct >= 0.6)  return 'Bạn thuộc top 20%\nngười kháng lừa đảo cao nhất';
  if (pct >= 0.4)  return 'Bạn thuộc top 50%\nngười kháng lừa đảo';
  return 'Hãy luyện tập thêm nhé!';
}

function ResultsScreen({ score, total, onRestart }) {
  const topMsg = getTopMessage(score, total);

  return (
    <div className="results-screen" role="region" aria-label="Kết quả">

      {/* ── Card ──────────────────────────────────────────────────── */}
      <div className="results-card">
        {/* Pin at top */}
        <div className="results-pin" />

        <h2 className="results-congrats">Chúc Mừng!</h2>
        <p className="results-subtitle">Chỉ số kháng lừa đảo:</p>

        {/* Score badge */}
        <div className="results-badge">
          <div className="results-badge__shield">
            <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M12 2L4 6v6c0 5.25 3.5 10.15 8 11.35C16.5 22.15 20 17.25 20 12V6L12 2z"
                fill="#e91e8c" />
              <path d="M9 12l2 2 4-4" stroke="#fff" strokeWidth="2"
                strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </div>
          <span className="results-badge__score">{score}/{total}</span>
        </div>

        <p className="results-topmsg">{topMsg}</p>
      </div>

      {/* ── Briefcase illustration ─────────────────────────────── */}
      <div className="results-briefcase" aria-hidden="true">
        <div className="briefcase">
          <div className="briefcase__handle" />
          <div className="briefcase__body">
            <div className="briefcase__latch" />
            <div className="briefcase__coins">
              {[0,1,2].map(col => (
                <div key={col} className="coin-stack">
                  {[0,1,2,3].map(row => (
                    <div key={row} className="coin" />
                  ))}
                </div>
              ))}
            </div>
            <div className="briefcase__emblem">
              <svg viewBox="0 0 24 24" fill="none">
                <path d="M12 2L4 6v6c0 5.25 3.5 10.15 8 11.35C16.5 22.15 20 17.25 20 12V6L12 2z"
                  fill="#e91e8c" opacity="0.9" />
                <path d="M9 12l2 2 4-4" stroke="#fff" strokeWidth="2.5"
                  strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </div>
          </div>
        </div>
      </div>

      {/* ── Play again button ─────────────────────────────────────── */}
      <button className="results-play-btn" onClick={onRestart}>
        Chơi lại
      </button>

    </div>
  );
}

// ─── ExplanationPanel ─────────────────────────────────────────────────────────

/**
 * Bottom-sheet overlay that slides up after the user answers.
 * Shows the verdict, tactic name, explanation body, and tips.
 * Dismissed by nodding or tapping "Tiếp tục".
 */
function ExplanationPanel({ question, isCorrect, onContinue }) {
  const exp = question.explanation;
  if (!exp) return null;

  let verdictPrefix;
  if (isCorrect === true)  verdictPrefix = 'Chính xác, ';
  else if (isCorrect === false) verdictPrefix = 'Sai rồi, ';
  else verdictPrefix = 'Đáp án: ';

  return (
    <div className="explanation-overlay" role="dialog" aria-label="Giải thích">
      <div className="explanation-card">
        <p className={`explanation-verdict ${isCorrect === true ? 'explanation-verdict--correct' : isCorrect === false ? 'explanation-verdict--wrong' : 'explanation-verdict--neutral'}`}>
          {verdictPrefix}{exp.heading}
        </p>

        {exp.tactic && (
          <p className="explanation-tactic">Chiêu: {exp.tactic}</p>
        )}

        {exp.body && (
          <p className="explanation-body">{exp.body}</p>
        )}

        {exp.tips?.length > 0 && (
          <div className="explanation-tips">
            {exp.tipsLabel && (
              <p className="explanation-tips-label">{exp.tipsLabel}</p>
            )}
            <ul className="explanation-tips-list">
              {exp.tips.map((tip, i) => (
                <li key={i}>{tip}</li>
              ))}
            </ul>
          </div>
        )}

        <p className="explanation-nod-hint">Gật đầu hoặc nhấn nút để tiếp tục</p>
      </div>

      <button className="explanation-continue-btn" onClick={onContinue}>
        Tiếp tục
      </button>
    </div>
  );
}

// ─── StatusBox ───────────────────────────────────────────────────────────────

/**
 * Small status box below each large panel.
 * Shows tilt hint, progress bar while holding, and ✓/✗ icon after confirm.
 */
function StatusBox({ side, questionType, hintLabel, active, holdProgress, confirmed, correct, dimmed }) {
  let hint;
  if (hintLabel) {
    // Single-panel mode: show the custom label (e.g. "Thật" / "Giả")
    hint = side === 'left' ? `👈 ${hintLabel}` : `${hintLabel} 👉`;
  } else {
    const letter = side === 'left' ? 'A' : 'B';
    const prefix = questionType === 'video' ? 'Video '
      : questionType === 'image' ? 'Image '
      : questionType === 'audio' ? 'Audio '
      : '';
    const arrow = side === 'left' ? '👈 ' : ' 👉';
    hint = side === 'left' ? `${arrow}${prefix}${letter}` : `${prefix}${letter}${arrow}`;
  }

  let cls = `status-box status-box--${side}`;
  if (active)                       cls += ' status-box--active';
  if (confirmed && correct === true)  cls += ' status-box--correct';
  if (confirmed && correct === false)  cls += ' status-box--wrong';
  if (dimmed)                        cls += ' status-box--dimmed';

  return (
    <div className={cls}>
      <span className="status-box__hint">{hint}</span>

      {confirmed && correct !== null && (
        <span className="status-box__icon" aria-label={correct ? 'Correct' : 'Wrong'}>
          {correct ? '✓' : '✗'}
        </span>
      )}

      {/* Progress bar that fills during the 300ms debounce hold */}
      <div className="status-bar-track" aria-hidden="true">
        <div
          className="status-bar-fill"
          style={{ width: `${Math.round(holdProgress * 100)}%` }}
        />
      </div>
    </div>
  );
}
