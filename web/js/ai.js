/**
 * Difficulty dispatcher. Runs inside the worker (or Node for tests).
 *
 * Tiers:
 *  - easy:   takes wins, blocks only half the time, otherwise random.
 *  - medium: win > block > threat heuristic (the MCTS rollout policy) with
 *            an occasional random slip.
 *  - hard:   time-limited alpha-beta search, no opening book.
 *  - expert: deeper search + the opening book distilled from the exact solve.
 */
import { easyMove, mediumMove } from './heuristic.js';
import { minimaxMove } from './minimax.js';

export const DIFFICULTIES = ['easy', 'medium', 'hard', 'expert'];

export function chooseMove(game, difficulty, rng = Math.random) {
  switch (difficulty) {
    case 'easy':
      return { move: easyMove(game, rng) };
    case 'medium':
      return { move: mediumMove(game, rng) };
    case 'hard': {
      const r = minimaxMove(game, { timeLimitMs: 350, maxDepth: 12, useBook: false });
      return { move: r.move, search: { depth: r.depth, nodes: r.nodes, score: r.score } };
    }
    case 'expert':
    default: {
      const r = minimaxMove(game, { timeLimitMs: 1000, maxDepth: 18, useBook: true });
      return { move: r.move, search: { depth: r.depth, nodes: r.nodes, score: r.score } };
    }
  }
}

/** Best move for the hint button — book-aware search, medium budget. */
export function hintMove(game) {
  return minimaxMove(game, { timeLimitMs: 650, maxDepth: 18, useBook: true }).move;
}
