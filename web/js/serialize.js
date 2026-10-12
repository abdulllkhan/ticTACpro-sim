/**
 * Board → text, for describing a position to a language model.
 *
 * Cell names match the convention used in FINDINGS.md (TL/TC/TR, ML/CC/MR,
 * BL/BC/BR) so transcripts line up with the research notes. Move IDs are
 * `<CELL>-<S|M|L>`, e.g. "CC-S" = Small at center.
 *
 * Pure and dependency-free: runs in the browser, in Node, and in the Vercel
 * serverless function.
 */
import { idx, legalMoves, RED, SIZE_NAMES } from './engine.js';

export const CELL_NAMES = [
  ['TL', 'TC', 'TR'],
  ['ML', 'CC', 'MR'],
  ['BL', 'BC', 'BR'],
];
const SIZE_CODES = ['S', 'M', 'L'];

/** "CC-S" → {r, c, s}; returns null if the id is malformed. */
export function parseMoveId(id) {
  if (typeof id !== 'string') return null;
  const m = id.trim().toUpperCase().match(/^([A-Z]{2})-([SML])$/);
  if (!m) return null;
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      if (CELL_NAMES[r][c] === m[1]) return { r, c, s: SIZE_CODES.indexOf(m[2]) };
    }
  }
  return null;
}

export const moveId = ({ r, c, s }) => `${CELL_NAMES[r][c]}-${SIZE_CODES[s]}`;

const owner = (v) => (v === 0 ? 'empty' : v === RED ? 'Red' : 'Blue');

/**
 * Describe the position from the mover's point of view.
 * Deliberately verbose and explicit: an ASCII grid alone leaves a model
 * guessing which layer a piece sits in, and the per-cell breakdown is what
 * makes the three independent slots legible.
 */
export function describePosition(game) {
  const me = game.current;
  const lines = [];

  lines.push(`You are playing ${owner(me)}. It is your turn.`);
  lines.push('');
  lines.push('BOARD — each cell has three independent slots (Small, Medium, Large):');
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      const slots = [0, 1, 2]
        .map((s) => `${SIZE_NAMES[s]}=${owner(game.board[idx(r, c, s)])}`)
        .join(', ');
      lines.push(`  ${CELL_NAMES[r][c]}: ${slots}`);
    }
  }

  lines.push('');
  lines.push('PIECES REMAINING:');
  for (const p of [1, 2]) {
    const inv = [0, 1, 2].map((s) => `${SIZE_NAMES[s]}=${game.pieces[p][s]}`).join(', ');
    lines.push(`  ${owner(p)}${p === me ? ' (you)' : ' (opponent)'}: ${inv}`);
  }

  const legal = legalMoves(game).map(moveId);
  lines.push('');
  lines.push(`LEGAL MOVES (${legal.length}) — you must pick exactly one of these:`);
  lines.push(`  ${legal.join(' ')}`);

  return { text: lines.join('\n'), legal };
}

/**
 * Rules and format explanation. Stable across every request in a game, so it
 * is the prompt-cache prefix — keep it byte-identical between calls (nothing
 * position-specific, no timestamps).
 */
export const SYSTEM_PROMPT = `You are playing TicTacPro, a 3x3 board game with stacking pieces. Play to win.

RULES

The board is a 3x3 grid. Cells are named by position:
  TL TC TR
  ML CC MR
  BL BC BR

Each player has 9 pieces: 3 Small, 3 Medium, 3 Large. Red moves first.

Every cell contains THREE INDEPENDENT SLOTS — one for Small, one for Medium,
one for Large. This is the most important rule and the easiest to get wrong:

  - Placing a piece fills only the slot of its own size in that cell.
  - Pieces NEVER cover, capture, or remove each other. There is no gobbling.
  - One cell can simultaneously hold your Small and your opponent's Large.
  - A move is legal if that cell's slot for that size is empty and you still
    have a piece of that size in hand.

WIN CONDITIONS — there are exactly two, and you win the instant either occurs:

  1. SAME-SIZE LINE: three pieces of the SAME size, all yours, in a row,
     column, or diagonal. Three of your Smalls on TR/CC/BL wins. A row
     containing your Small, your Medium and your Large does NOT win — the
     three pieces must all be the same size.

  2. BULLSEYE: all three sizes (Small AND Medium AND Large), all yours,
     stacked in one single cell. Your Small + Medium + Large at CC wins.

The eight lines are: the three rows (TL-TC-TR, ML-CC-MR, BL-BC-BR), the three
columns (TL-ML-BL, TC-CC-BC, TR-MR-BR), and the two diagonals (TL-CC-BR,
TR-CC-BL).

If all 18 pieces are placed with no win, the game is a draw.

HOW TO THINK ABOUT A POSITION

Work through these in order before choosing:

  1. Can you win immediately? Check every legal move for a completed same-size
     line and for a completed bullseye. If one exists, play it.
  2. Can your opponent win on their next move? Find every square where they
     would complete a line or a bullseye, and block it. Remember they can only
     do so if they still hold a piece of that size.
  3. Can you create two threats at once? A move producing two separate winning
     threats cannot be answered, since your opponent can only block one.
  4. Otherwise, build toward a line or a bullseye while denying theirs.
     Occupying a slot in a cell where your opponent already has two pieces
     permanently kills their bullseye there.

Remember that tracking three separate size layers is where mistakes happen.
Before you commit, re-read the slot listing for the cell you chose and confirm
that the slot is genuinely empty and that the move does what you intend.

RESPONDING

Call the submit_move tool exactly once with a move id from the supplied list of
legal moves. A move id is the cell name, a hyphen, and S, M or L — for example
"CC-S" places your Small at the center cell.`;
