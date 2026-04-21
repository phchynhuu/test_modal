/**
 * App – manages start screen → quiz transition.
 */

import React, { useRef, useState } from 'react';
import StartPage  from './components/StartPage.jsx';
import QuizEngine from './components/QuizEngine.jsx';

export default function App() {
  const videoRef  = useRef(null);
  const canvasRef = useRef(null);
  const [started, setStarted] = useState(false);

  if (!started) {
    return <StartPage onStart={() => setStarted(true)} />;
  }

  return (
    <div className="app">
      <QuizEngine videoRef={videoRef} canvasRef={canvasRef} />
    </div>
  );
}
