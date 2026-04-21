/**
 * StartPage
 *
 * Full-screen intro screen shown before the quiz begins.
 * Displays the "Thật Giả Quiz" logo and a "Chơi ngay" button.
 */

import React from 'react';

export default function StartPage({ onStart }) {
  return (
    <div className="start-page">
      {/* ── Logo ──────────────────────────────────────────────── */}
      <div className="start-logo">
        <div className="logo-bubbles">
          {/* Left bubble: white bg, pink text */}
          <div className="logo-bubble--left">Thật</div>
          {/* Right bubble: pink bg, white text */}
          <div className="logo-bubble--right">Giả</div>
        </div>
        <div className="logo-quiz-label">Quiz</div>
      </div>

      {/* ── CTA button ────────────────────────────────────────── */}
      <button className="start-btn" onClick={onStart}>
        Chơi ngay
      </button>
    </div>
  );
}
