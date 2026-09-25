/**
 * TicTacPro game engine — pure JS port of game/tictacpro.py.
 *
 * Rules:
 *  - 3×3 board; each cell has 3 independent size slots (Small, Medium, Large).
 *  - Each player has 3 pieces of each size (9 total). No gobbling.
 *  - Win: three same-size pieces in a row/col/diagonal, OR a "bullseye"
 *    (all three sizes of your color stacked in one cell).
 *  - Red moves first. All 18 pieces placed with no win → draw.
 *
 * Board indexing matches the Python engine's flat layout: idx = r*9 + c*3 + s
 * where s is 0=Small, 1=Medium, 2=Large. Cell values: 0=empty, 1=Red, 2=Blue.
 */

export const EMPTY = 0;
export const RED = 1;
export const BLUE = 2;

export const SMALL = 0;
export const MEDIUM = 1;
export const LARGE = 2;

export const SIZE_NAMES = ['Small', 'Medium', 'Large'];
export const PLAYER_NAMES = ['', 'Red', 'Blue'];

export const idx = (r, c, s) => r * 9 + c * 3 + s;

// The 8 winning lines as [r, c] cell pairs.
export const LINES = [
  [[0, 0], [0, 1], [0, 2]],
  [[1, 0], [1, 1], [1, 2]],
  [[2, 0], [2, 1], [2, 2]],
  [[0, 0], [1, 0], [2, 0]],
  [[0, 1], [1, 1], [2, 1]],
  [[0, 2], [1, 2], [2, 2]],
  [[0, 0], [1, 1], [2, 2]],
  [[0, 2], [1, 1], [2, 0]],
];

// WIN_PAIRS[s][cellIdx] = array of [i1, i2] flat indices: for each way the slot
// (cell, s) can complete a win, the two OTHER slots that must hold the mover's
// color. Line pairs use the same size at the other two cells of each line
// through the cell; the final pair is the bullseye (other two sizes, same cell).
// Mirrors _WIN_LINES in game/tictacpro.py.
const OTHER_SIZES = [[1, 2], [0, 2], [0, 1]];
export const WIN_PAIRS = (() => {
  const linesThrough = Array.from({ length: 9 }, () => []);
  for (const line of LINES) {
    for (let i = 0; i < 3; i++) {
      const [r, c] = line[i];
      const rest = [];
      for (let j = 0; j < 3; j++) if (j !== i) rest.push(line[j]);
      linesThrough[r * 3 + c].push(rest);
    }
  }
  const table = [];
  for (let s = 0; s < 3; s++) {
    const [o1, o2] = OTHER_SIZES[s];
    const row = [];
    for (let ci = 0; ci < 9; ci++) {
      const r = (ci / 3) | 0, c = ci % 3;
      const pairs = linesThrough[ci].map(([[r1, c1], [r2, c2]]) =>
        [idx(r1, c1, s), idx(r2, c2, s)]);
      pairs.push([idx(r, c, o1), idx(r, c, o2)]);
      row.push(pairs);
    }
    table.push(row);
  }
  return table;
})();

/** Fresh game state. Plain data — safe to postMessage to a worker. */
export function newGame() {
  return {
    board: new Uint8Array(27),
    // pieces[player][size] = remaining count; index 0 unused.
    pieces: [null, [3, 3, 3], [3, 3, 3]],
    current: RED,
    winner: EMPTY,
    gameOver: false,
    piecesLeft: 18,
    history: [],           // [{r, c, s, player}]
  };
}

export function cloneGame(g) {
  return {
    board: new Uint8Array(g.board),
    pieces: [null, g.pieces[1].slice(), g.pieces[2].slice()],
    current: g.current,
    winner: g.winner,
    gameOver: g.gameOver,
    piecesLeft: g.piecesLeft,
    history: g.history.slice(),
  };
}

/** Rebuild a live game (typed-array board) from a structured-cloned or JSON state. */
export function reviveGame(o) {
  return {
    board: o.board instanceof Uint8Array ? new Uint8Array(o.board) : Uint8Array.from(o.board),
    pieces: [null, Array.from(o.pieces[1]), Array.from(o.pieces[2])],
    current: o.current,
    winner: o.winner,
    gameOver: o.gameOver,
    piecesLeft: o.piecesLeft,
    history: (o.history || []).map(m => ({ ...m })),
  };
}

export const otherPlayer = (p) => 3 - p;

/** Would placing `player` at (r,c,s) win immediately? Board unchanged. */
export function wouldWin(board, r, c, s, player) {
  const pairs = WIN_PAIRS[s][r * 3 + c];
  for (let k = 0; k < pairs.length; k++) {
    const [i1, i2] = pairs[k];
    if (board[i1] === player && board[i2] === player) return true;
  }
  return false;
}

export function isLegal(g, r, c, s) {
  return !g.gameOver &&
    r >= 0 && r < 3 && c >= 0 && c < 3 && s >= 0 && s < 3 &&
    g.pieces[g.current][s] > 0 &&
    g.board[idx(r, c, s)] === EMPTY;
}

/** All legal moves for the current player, as {r, c, s}. Size-major order (S, M, L). */
export function legalMoves(g) {
  if (g.gameOver) return [];
  const moves = [];
  const pieces = g.pieces[g.current];
  for (let s = 0; s < 3; s++) {
    if (pieces[s] === 0) continue;
    for (let ci = 0; ci < 9; ci++) {
      const r = (ci / 3) | 0, c = ci % 3;
      if (g.board[r * 9 + c * 3 + s] === EMPTY) moves.push({ r, c, s });
    }
  }
  return moves;
}

/** Sizes the current player can place SOMEWHERE, as booleans [S, M, L]. */
export function placeableSizes(g) {
  const out = [false, false, false];
  if (g.gameOver) return out;
  const pieces = g.pieces[g.current];
  for (let s = 0; s < 3; s++) {
    if (pieces[s] === 0) continue;
    for (let ci = 0; ci < 9 && !out[s]; ci++) {
      if (g.board[ci * 3 + s] === EMPTY) out[s] = true;
    }
  }
  return out;
}

/** Sizes of the current player that fit in cell (r,c): array of s values. */
export function sizesForCell(g, r, c) {
  const out = [];
  if (g.gameOver) return out;
  for (let s = 0; s < 3; s++) {
    if (g.pieces[g.current][s] > 0 && g.board[idx(r, c, s)] === EMPTY) out.push(s);
  }
  return out;
}

/**
 * Apply a move for the current player. Returns false if illegal.
 * Matches Python make_move: on game end the mover stays `current`.
 */
export function makeMove(g, r, c, s) {
  if (!isLegal(g, r, c, s)) return false;
  const player = g.current;
  const wins = wouldWin(g.board, r, c, s, player);
  g.board[idx(r, c, s)] = player;
  g.pieces[player][s] -= 1;
  g.piecesLeft -= 1;
  g.history.push({ r, c, s, player });
  if (wins) {
    g.winner = player;
    g.gameOver = true;
  } else if (g.piecesLeft === 0) {
    g.gameOver = true;         // draw: all 18 pieces placed
  } else {
    g.current = otherPlayer(player);
  }
  return true;
}

/** Undo the last move (inverse of makeMove). Returns false if no history. */
export function undoMove(g) {
  const last = g.history.pop();
  if (!last) return false;
  g.board[idx(last.r, last.c, last.s)] = EMPTY;
  g.pieces[last.player][last.s] += 1;
  g.piecesLeft += 1;
  g.winner = EMPTY;
  g.gameOver = false;
  g.current = last.player;
  return true;
}

/**
 * Search-only push/pop mirroring rl/minimax_agent.py _push/_pop:
 * ALWAYS switches current player (negamax invariant), skips history.
 * pop() must receive push()'s return token, in LIFO order.
 */
export function searchPush(g, r, c, s) {
  const player = g.current;
  const token = { r, c, s, player, winner: g.winner, gameOver: g.gameOver };
  if (wouldWin(g.board, r, c, s, player)) {
    g.winner = player;
    g.gameOver = true;
  }
  g.board[idx(r, c, s)] = player;
  g.pieces[player][s] -= 1;
  g.piecesLeft -= 1;
  if (g.piecesLeft === 0) g.gameOver = true;
  g.current = otherPlayer(player);
  return token;
}

export function searchPop(g, token) {
  g.board[idx(token.r, token.c, token.s)] = EMPTY;
  g.pieces[token.player][token.s] += 1;
  g.piecesLeft += 1;
  g.winner = token.winner;
  g.gameOver = token.gameOver;
  g.current = token.player;
}

/**
 * Describe how `winner` won, for UI highlighting.
 * Returns { type: 'line', size, cells: [[r,c]×3] } or
 *         { type: 'bullseye', cells: [[r,c]] } or null.
 */
export function winInfo(g) {
  const p = g.winner;
  if (!p) return null;
  for (let s = 0; s < 3; s++) {
    for (const line of LINES) {
      if (line.every(([r, c]) => g.board[idx(r, c, s)] === p)) {
        return { type: 'line', size: s, cells: line.map(([r, c]) => [r, c]) };
      }
    }
  }
  for (let ci = 0; ci < 9; ci++) {
    const r = (ci / 3) | 0, c = ci % 3;
    if (g.board[idx(r, c, 0)] === p && g.board[idx(r, c, 1)] === p && g.board[idx(r, c, 2)] === p) {
      return { type: 'bullseye', cells: [[r, c]] };
    }
  }
  return null;
}

/** Top visible piece per cell (largest first): {player, s} or null. 3×3 array. */
export function visibleBoard(g) {
  const out = [];
  for (let r = 0; r < 3; r++) {
    const row = [];
    for (let c = 0; c < 3; c++) {
      let top = null;
      for (let s = 2; s >= 0; s--) {
        const p = g.board[idx(r, c, s)];
        if (p !== EMPTY) { top = { player: p, s }; break; }
      }
      row.push(top);
    }
    out.push(row);
  }
  return out;
}
