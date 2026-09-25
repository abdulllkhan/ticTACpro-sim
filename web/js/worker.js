/**
 * AI Web Worker — keeps search off the UI thread.
 *
 * In:  { id, type: 'move', state, difficulty }   → { id, type: 'move', move, search?, ms }
 *      { id, type: 'hint', state }               → { id, type: 'hint', move, ms }
 * `state` is a structured-cloned engine game object (reviveGame handles it).
 */
import { reviveGame } from './engine.js';
import { chooseMove, hintMove } from './ai.js';

self.onmessage = (e) => {
  const { id, type, state, difficulty } = e.data;
  const t0 = performance.now();
  try {
    const game = reviveGame(state);
    if (type === 'move') {
      const { move, search } = chooseMove(game, difficulty);
      self.postMessage({ id, type, move, search, ms: performance.now() - t0 });
    } else if (type === 'hint') {
      self.postMessage({ id, type, move: hintMove(game), ms: performance.now() - t0 });
    }
  } catch (err) {
    self.postMessage({ id, type, error: String(err && err.stack || err) });
  }
};
