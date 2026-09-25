/**
 * Python↔JS engine parity check.
 *
 * Run:  node tests_web/parity.mjs   (after gen_traces.py has written traces.json)
 *
 * Replays every traced game through web/js/engine.js and at EVERY ply compares:
 * the sorted legal-move set ("r,c,s", s 0-indexed), the 27 board bytes, gameOver,
 * winner, and current player. Both engines keep the mover as `current` when the
 * game ends, so states are compared directly. Exits 1 on first mismatch.
 */
import { readFileSync } from 'node:fs';
import { newGame, legalMoves, makeMove } from '../web/js/engine.js';

const traces = JSON.parse(
  readFileSync(new URL('./traces.json', import.meta.url), 'utf8'));

const moveKeys = (g) =>
  legalMoves(g).map(({ r, c, s }) => `${r},${c},${s}`).sort();

function fail(gi, pi, ply, field, expected, actual, g) {
  console.error(`PARITY FAIL: game ${gi}, ply ${pi}, field "${field}"`);
  console.error(`  move applied: ${JSON.stringify(ply.move)} (r,c,s — s 0-indexed)`);
  console.error(`  expected (python): ${JSON.stringify(expected)}`);
  console.error(`  actual   (js)    : ${JSON.stringify(actual)}`);
  console.error(`  python post-move board: ${JSON.stringify(ply.board)}`);
  console.error(`  js board              : ${JSON.stringify(Array.from(g.board))}`);
  console.error(`  js state: current=${g.current} winner=${g.winner} ` +
                `gameOver=${g.gameOver} piecesLeft=${g.piecesLeft}`);
  process.exit(1);
}

let totalPlies = 0;
for (let gi = 0; gi < traces.games.length; gi++) {
  const game = traces.games[gi];
  const g = newGame();
  for (let pi = 0; pi < game.plies.length; pi++) {
    const ply = game.plies[pi];

    // Pre-move: legal-move sets must match exactly.
    const jsLegal = moveKeys(g);
    if (jsLegal.length !== ply.legal.length ||
        jsLegal.some((m, i) => m !== ply.legal[i])) {
      fail(gi, pi, ply, 'legal moves', ply.legal, jsLegal, g);
    }

    const [r, c, s] = ply.move;
    if (!makeMove(g, r, c, s)) {
      fail(gi, pi, ply, 'makeMove returned false', true, false, g);
    }

    // Post-move state.
    for (let i = 0; i < 27; i++) {
      if (g.board[i] !== ply.board[i]) {
        fail(gi, pi, ply, `board[${i}]`, ply.board[i], g.board[i], g);
      }
    }
    if (g.gameOver !== ply.game_over) fail(gi, pi, ply, 'gameOver', ply.game_over, g.gameOver, g);
    if (g.winner !== ply.winner) fail(gi, pi, ply, 'winner', ply.winner, g.winner, g);
    if (g.current !== ply.current_player) fail(gi, pi, ply, 'current player', ply.current_player, g.current, g);
    totalPlies++;
  }
  if (!g.gameOver || game.plies.length !== game.final_move_count) {
    fail(gi, game.plies.length - 1, game.plies.at(-1), 'final move count / terminal',
         game.final_move_count, game.plies.length, g);
  }
}

console.log(`PARITY OK: ${traces.games.length} games, ${totalPlies} plies`);
