/**
 * TicTacPro UI.
 *
 * Interaction model (the core fix over the old UI): there are no dead clicks.
 *  - The tray shows exactly which of your sizes are playable right now.
 *  - Selecting a size highlights every cell that accepts it.
 *  - Tapping any cell with room either places (unique fit / selected size fits)
 *    or opens a mini chooser with just the sizes that fit that cell.
 */
import {
  newGame, makeMove, undoMove, isLegal, legalMoves, placeableSizes, sizesForCell,
  winInfo, otherPlayer, reviveGame, RED, BLUE, SIZE_NAMES, idx,
} from './engine.js';

// ── Constants ────────────────────────────────────────────────────────────────
const POS_NAMES = [
  ['top-left', 'top-center', 'top-right'],
  ['mid-left', 'center', 'mid-right'],
  ['bottom-left', 'bottom-center', 'bottom-right'],
];
const DIFF_NOTES = {
  easy: 'Learns you the ropes — misses half its blocks.',
  medium: 'Solid tactics — wins, blocks and threats.',
  hard: 'Tree search. Punishes loose play.',
  expert: 'Deep search + the solved opening book. Good luck.',
  llm: 'Claude reads the board in words and reasons out a move. Slower, and it plays nothing like a search engine.',
};
const LLM_DIFFICULTY = 'llm';
// SVG radii for S/M/L in a 100×100 cell viewBox: [radius, strokeWidth] (0 = filled)
const PIECE_GEOM = [[13, 0], [26, 9], [40, 10]];

const LS = { settings: 'ttp-settings', scores: 'ttp-scores', theme: 'ttp-theme' };

// ── State ────────────────────────────────────────────────────────────────────
let game = newGame();
let settings = { mode: 'ai', side: RED, difficulty: 'medium' };
let scores = { red: 0, blue: 0, draws: 0 };
let selSize = null;          // currently selected size (0/1/2) or null
let aiBusy = false;
let hintBusy = false;
let reqId = 0;               // invalidates in-flight worker replies on new game
let lastMove = null;         // {r,c,s,player,popped} for the pop/glow animation
let scored = false;          // this game already counted toward the score
let scoredAs = null;         // which counter endGame incremented ('red'|'blue'|'draws')
let winAnimated = false;     // winning-cell blink already played for this game

const $ = (id) => document.getElementById(id);
const store = {
  get(k, fallback) { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : fallback; } catch { return fallback; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode */ } },
};

// ── Worker ───────────────────────────────────────────────────────────────────
// Resilience contract: askWorker ALWAYS settles. A worker that fails to load,
// crashes, garbles a message, or hangs past the timeout rejects the promise
// (and marks itself dead so later asks reject immediately), which routes the
// caller into the main-thread computeLocally fallback — the game never locks
// in "AI is thinking…".
let worker = null;
let workerDead = false;
const pending = new Map();   // id → {resolve, reject, timer}
const WORKER_TIMEOUT_MS = 8000;

function rejectAllPending(err) {
  for (const p of pending.values()) {
    clearTimeout(p.timer);
    p.reject(err);
  }
  pending.clear();
}

function makeWorker() {
  try {
    worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  } catch (e) {
    worker = null;
    workerDead = true;
    return;
  }
  workerDead = false;
  worker.onmessage = (e) => {
    const p = pending.get(e.data.id);
    if (!p) return;
    pending.delete(e.data.id);
    clearTimeout(p.timer);
    e.data.error ? p.reject(new Error(e.data.error)) : p.resolve(e.data);
  };
  const die = (e) => {
    workerDead = true;
    rejectAllPending(new Error(e && e.message || 'worker failed'));
  };
  worker.onerror = die;
  worker.onmessageerror = die;
}
makeWorker();

function askWorker(msg) {
  return new Promise((resolve, reject) => {
    const id = ++reqId;
    if (workerDead || !worker) { reject(new Error('worker unavailable')); return; }
    const timer = setTimeout(() => {
      if (!pending.has(id)) return;
      worker.terminate();               // hung — recycle it for the next ask
      rejectAllPending(new Error('worker timeout'));
      makeWorker();
    }, WORKER_TIMEOUT_MS);
    pending.set(id, { resolve, reject, timer });
    worker.postMessage({ id, ...msg, state: snapshot() });
  });
}

// Abandon in-flight AI work (new game, difficulty switch). Stale replies are
// already ignored via reqId, but terminating also frees the CPU so the next
// request doesn't queue behind an abandoned expert search.
function abandonAiWork() {
  reqId++;
  if (pending.size > 0 && worker) {
    worker.terminate();
    rejectAllPending(new Error('superseded'));
    makeWorker();
  }
}
function snapshot() {
  return {
    board: game.board, pieces: game.pieces, current: game.current,
    winner: game.winner, gameOver: game.gameOver, piecesLeft: game.piecesLeft, history: [],
  };
}

// Main-thread fallback if the worker is unavailable (e.g. a host that blocks
// module workers). Blocks the UI for up to ~1s on expert — acceptable as a fallback.
let localAI = null;
async function computeLocally(type) {
  if (!localAI) localAI = await import('./ai.js');
  const g = reviveGame(snapshot());
  return type === 'hint'
    ? { move: localAI.hintMove(g) }
    : localAI.chooseMove(g, settings.difficulty);
}

// ── Helpers ──────────────────────────────────────────────────────────────────
const isAiMode = () => settings.mode === 'ai';
const humanSide = () => (isAiMode() ? settings.side : game.current);
const aiSide = () => otherPlayer(settings.side);
const humanCanAct = () => !game.gameOver && !aiBusy && (!isAiMode() || game.current === settings.side);
const colorName = (p) => (p === RED ? 'Red' : 'Blue');
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function pieceSVG(player, s, extraClass = '') {
  const [r, sw] = PIECE_GEOM[s];
  const cls = `pc pc-${player === RED ? 'red' : 'blue'} ${extraClass}`;
  return sw === 0
    ? `<circle class="${cls}" data-s="${s}" cx="50" cy="50" r="${r}" fill="currentColor"/>`
    : `<circle class="${cls}" data-s="${s}" cx="50" cy="50" r="${r}" fill="none" stroke="currentColor" stroke-width="${sw}"/>`;
}

// Pointer devices with hover get radial per-slot targeting; touch gets the
// tap flow (selected size → unique fit → chooser).
const FINE_POINTER = window.matchMedia('(hover: hover) and (pointer: fine)').matches;

/** Which size band (0/1/2) the pointer is in, from distance to cell center. */
function bandFromEvent(cell, e) {
  const rect = cell.getBoundingClientRect();
  const x = ((e.clientX - rect.left) / rect.width) * 100 - 50;
  const y = ((e.clientY - rect.top) / rect.height) * 100 - 50;
  const d = Math.hypot(x, y);
  return d < 19.5 ? 0 : d < 33 ? 1 : 2;
}

/** Re-apply the hovered-band highlight after a render or pointer move. */
function updateHot(cell) {
  if (!FINE_POINTER) return;
  let hotShown = false;
  for (const el of cell.querySelectorAll('.slot-open')) {
    const on = cell._hot === +el.dataset.s;
    el.classList.toggle('hot', on);
    if (on) hotShown = true;
  }
  cell.style.cursor = hotShown ? 'pointer' : 'default';
}

let toastTimer = null;
let toastClearTimer = null;
function toast(msg) {
  const el = $('toast');
  // Force a text mutation even for a repeated message so the live region re-announces.
  el.textContent = '';
  el.textContent = msg;
  el.classList.add('show');
  clearTimeout(toastTimer);
  clearTimeout(toastClearTimer);
  toastTimer = setTimeout(() => {
    el.classList.remove('show');
    toastClearTimer = setTimeout(() => { el.textContent = ''; }, 250);
  }, 2200);
}

// ── Modal focus management ───────────────────────────────────────────────────
// The page behind a modal goes inert (unfocusable, unclickable); focus moves
// into the dialog and returns to where it was on close.
let lastFocus = null;
function setBackgroundInert(on) {
  document.querySelector('.layout').inert = on;
  document.querySelector('.topbar').inert = on;
}
function openModal(id, focusSel) {
  const ov = $(id);
  if (!ov.hidden) return;
  lastFocus = document.activeElement;
  ov.hidden = false;
  setBackgroundInert(true);
  ov.querySelector(focusSel)?.focus();
}
function closeModal(id) {
  const ov = $(id);
  if (ov.hidden) return;
  ov.hidden = true;
  setBackgroundInert(false);
  if (lastFocus && document.contains(lastFocus)) lastFocus.focus();
  lastFocus = null;
}

// ── Board DOM ────────────────────────────────────────────────────────────────
const boardEl = $('board');
const cells = [];            // [r][c] → button element
for (let r = 0; r < 3; r++) {
  cells.push([]);
  for (let c = 0; c < 3; c++) {
    const cell = document.createElement('button');
    cell.className = 'cell';
    cell.dataset.r = r;
    cell.dataset.c = c;
    cell._hot = null;
    cell.addEventListener('click', (e) => onCellTap(r, c, e));
    if (FINE_POINTER) {
      cell.addEventListener('pointermove', (e) => {
        const band = bandFromEvent(cell, e);
        if (band !== cell._hot) { cell._hot = band; updateHot(cell); }
      });
      cell.addEventListener('pointerleave', () => { cell._hot = null; updateHot(cell); });
    }
    boardEl.appendChild(cell);
    cells[r].push(cell);
  }
}

// ── Rendering ────────────────────────────────────────────────────────────────
function render() {
  renderBoard();
  renderTrays();
  renderStatus();
  renderHistory();
  renderScores();
}

function renderBoard() {
  const canAct = humanCanAct();
  const mover = game.current;
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
    const cell = cells[r][c];
    const parts = [];
    const openNames = [];
    const haveNames = [];
    // Draw large → small so the dot sits on top.
    for (let s = 2; s >= 0; s--) {
      const p = game.board[idx(r, c, s)];
      if (p) {
        const isLast = lastMove && lastMove.r === r && lastMove.c === c && lastMove.s === s;
        // pc-pop only on the render right after placement — a re-render (hint,
        // tray tap, resize) must not replay the entrance animation.
        parts.push(pieceSVG(p, s, isLast ? (lastMove.popped ? 'pc-last' : 'pc-pop pc-last') : ''));
        haveNames.push(`${colorName(p)} ${SIZE_NAMES[s]}`);
      }
    }
    // Open slots the acting player can actually fill: faint guides, hover-lit.
    const fits = [];
    if (canAct) {
      for (let s = 0; s < 3; s++) {
        if (game.pieces[mover][s] > 0 && game.board[idx(r, c, s)] === 0) fits.push(s);
      }
    }
    for (const s of fits) {
      parts.push(pieceSVG(mover, s, 'slot-open' + (!FINE_POINTER && selSize === s ? ' sel' : '')));
    }
    cell._open = fits;
    cell.innerHTML = `<svg viewBox="0 0 100 100">${parts.join('')}</svg>`;
    cell.classList.toggle('can-place', fits.length > 0);
    cell.classList.toggle('p1', mover === RED);
    cell.classList.toggle('p2', mover === BLUE);
    updateHot(cell);
    for (let s = 0; s < 3; s++) if (game.board[idx(r, c, s)] === 0) openNames.push(SIZE_NAMES[s]);
    cell.setAttribute('aria-label',
      `${POS_NAMES[r][c]} cell. ${haveNames.length ? 'Has ' + haveNames.join(', ') + '.' : 'Empty.'}` +
      ` Open slots: ${openNames.length ? openNames.join(', ') : 'none'}.`);
    // Win highlight
    cell.classList.remove('win', 'win-blink');
  }
  const info = game.gameOver ? winInfo(game) : null;
  if (info) {
    document.documentElement.style.setProperty('--win-color', game.winner === RED ? 'var(--red)' : 'var(--blue)');
    // The exact winning cells blink in the winner's color (3 for a line,
    // 1 for a bullseye) — only on the first render after the win, so later
    // re-renders (theme toggle, review) keep the steady tint without replaying.
    for (const [r, c] of info.cells) {
      cells[r][c].classList.add('win');
      if (!winAnimated) cells[r][c].classList.add('win-blink');
    }
    winAnimated = true;
  }
  if (lastMove) lastMove.popped = true;
}

function trayChip(player, s, count, { interactive }) {
  const chip = document.createElement('button');
  chip.type = 'button';
  chip.className = `size-chip ${player === RED ? 'p1' : 'p2'}`;
  chip.dataset.s = s;
  const playable = humanCanAct() && game.current === player && count > 0 && legalCellCount(s) > 0;
  const canSelect = interactive && !FINE_POINTER && playable;
  // No `disabled` attribute: unplayable chips stay tappable so they can
  // explain themselves (a disabled button swallows the tap silently).
  if (interactive && !FINE_POINTER && !canSelect) {
    chip.classList.add('is-disabled');
    chip.setAttribute('aria-disabled', 'true');
  }
  if (interactive && FINE_POINTER && count > 0) chip.classList.add('informational');
  if (count === 0) chip.classList.add('exhausted');
  if (canSelect && selSize === s) chip.classList.add('selected');
  chip.innerHTML =
    `<svg viewBox="0 0 100 100">${pieceSVG(player, s)}</svg>` +
    `<span class="chip-count"><b>${count}</b>×</span>` +
    `<span class="chip-name">${SIZE_NAMES[s]}</span>`;
  chip.setAttribute('aria-label', `${SIZE_NAMES[s]}: ${count} left`);
  if (interactive) {
    chip.addEventListener('click', () => {
      if (FINE_POINTER) { toast('Hover the board — every faint ring is a spot you can play'); return; }
      if (aiBusy) { toast('AI is thinking…'); return; }
      if (!humanCanAct() || game.current !== player) return;
      if (count === 0) { toast(`No ${SIZE_NAMES[s]} pieces left`); return; }
      if (legalCellCount(s) === 0) { toast(`No open ${SIZE_NAMES[s]} slot on the board`); return; }
      selSize = s;
      closeChooser();
      render();
    });
  }
  return chip;
}

function legalCellCount(s) {
  if (game.pieces[game.current][s] === 0) return 0;
  let n = 0;
  for (let ci = 0; ci < 9; ci++) if (game.board[ci * 3 + s] === 0) n++;
  return n;
}

function renderTrays() {
  const top = $('tray-top');
  const bottom = $('tray-bottom');
  // Bottom tray belongs to the human (vs AI) or Red (2P); top is the other player.
  const bottomP = isAiMode() ? settings.side : RED;
  const topP = otherPlayer(bottomP);
  for (const [el, player, interactive] of [[top, topP, !isAiMode()], [bottom, bottomP, true]]) {
    el.innerHTML = '';
    el.classList.toggle('tray-passive', !interactive);
    el.setAttribute('aria-label', interactive ? `${colorName(player)} pieces` : 'AI pieces');
    const label = document.createElement('div');
    label.className = 'tray-label';
    const who = isAiMode()
      ? (player === settings.side ? 'You' : 'AI')
      : (player === RED ? 'Red' : 'Blue');
    label.innerHTML = `<span class="who ${player === RED ? 'red' : 'blue'}">${who}</span>${colorName(player) === who ? '' : colorName(player)}`;
    el.appendChild(label);
    const groups = document.createElement('div');
    groups.className = 'tray-groups';
    for (let s = 0; s < 3; s++) groups.appendChild(trayChip(player, s, game.pieces[player][s], { interactive }));
    el.appendChild(groups);
    el.classList.toggle('inactive', !game.gameOver && game.current !== player);
  }
}

function renderStatus() {
  const dot = $('status-dot');
  const txt = $('status-text');
  dot.className = 'status-dot';
  if (game.gameOver) {
    if (game.winner === 0) {
      dot.classList.add('draw');
      txt.textContent = 'Draw — all pieces placed';
    } else {
      dot.classList.add(game.winner === BLUE ? 'p2' : 'p1', 'done');
      txt.textContent = isAiMode()
        ? (game.winner === settings.side ? 'You win!' : 'AI wins')
        : `${colorName(game.winner)} wins!`;
    }
  } else if (aiBusy) {
    dot.classList.add('thinking');
    if (aiSide() === BLUE) dot.classList.add('p2');
    txt.textContent = 'AI is thinking…';
  } else {
    const p = game.current;
    dot.classList.add('pulse');
    if (p === BLUE) dot.classList.add('p2');
    txt.textContent = isAiMode()
      ? (p === settings.side ? 'Your turn' : 'AI to move')
      : `${colorName(p)}'s turn`;
  }
}

function renderHistory() {
  const el = $('history');
  if (game.history.length === 0) {
    el.innerHTML = '<li class="history-empty">No moves yet.</li>';
    return;
  }
  el.innerHTML = game.history.map((m, i) => {
    const who = isAiMode() ? (m.player === settings.side ? 'You' : 'AI') : colorName(m.player);
    return `<li><i class="dot ${m.player === RED ? 'dot-red' : 'dot-blue'}"></i>` +
      `<span class="h-who">${i + 1}. ${who}</span>` +
      `<span class="h-what">${SIZE_NAMES[m.s]} → ${POS_NAMES[m.r][m.c]}</span></li>`;
  }).reverse().join('');
}

function renderScores() {
  $('score-red').textContent = scores.red;
  $('score-blue').textContent = scores.blue;
  $('score-draws').textContent = scores.draws;
}

function renderControls() {
  for (const b of document.querySelectorAll('#seg-mode .seg-btn'))
    b.setAttribute('aria-pressed', String(b.dataset.mode === settings.mode));
  for (const b of document.querySelectorAll('#seg-side .seg-btn'))
    b.setAttribute('aria-pressed', String(+b.dataset.side === settings.side));
  for (const b of document.querySelectorAll('#seg-difficulty .seg-btn'))
    b.setAttribute('aria-pressed', String(b.dataset.diff === settings.difficulty));
  $('field-side').style.display = isAiMode() ? '' : 'none';
  $('field-difficulty').style.display = isAiMode() ? '' : 'none';
  $('btn-hint').style.display = isAiMode() ? '' : 'none';
  $('diff-note').textContent = DIFF_NOTES[settings.difficulty];
  updateActionButtons();
}

function updateActionButtons() {
  const undoable = !aiBusy && game.history.some((m) => !isAiMode() || m.player === settings.side);
  $('btn-undo').disabled = !undoable;
  $('btn-hint').disabled = !humanCanAct() || hintBusy;
}

// ── Size auto-selection: there is always a valid selection on your turn ─────
function autoSelectSize() {
  if (FINE_POINTER) { selSize = null; return; }   // hover targeting: nothing to select
  if (!humanCanAct()) { selSize = null; return; }
  const can = placeableSizes(game);
  if (selSize !== null && can[selSize]) return;
  selSize = can.findIndex(Boolean);
  if (selSize === -1) selSize = null;      // no legal move at all (never happens pre-end)
}

// ── Cell interaction ─────────────────────────────────────────────────────────
function shakeCell(r, c) {
  cells[r][c].classList.add('shake');
  setTimeout(() => cells[r][c].classList.remove('shake'), 350);
}

function onCellTap(r, c, e) {
  closeChooser();
  if (game.gameOver) { showOverlay(); return; }
  if (aiBusy) { toast('AI is thinking…'); return; }
  if (isAiMode() && game.current !== settings.side) { toast("It's the AI's move"); return; }

  const fits = sizesForCell(game, r, c);
  if (fits.length === 0) {
    shakeCell(r, c);
    const anyOpen = [0, 1, 2].some((s) => game.board[idx(r, c, s)] === 0);
    toast(anyOpen ? 'You have no pieces left for the open slots here' : 'That cell is full');
    return;
  }

  if (FINE_POINTER && e && e.detail > 0) {   // detail 0 = keyboard-activated click
    // Radial targeting: the clicked ring decides the size. No selection step.
    const s = bandFromEvent(cells[r][c], e);
    if (fits.includes(s)) return place(r, c, s);
    shakeCell(r, c);
    if (game.board[idx(r, c, s)] !== 0) toast(`The ${SIZE_NAMES[s]} spot here is already taken`);
    else if (game.pieces[game.current][s] === 0) toast(`No ${SIZE_NAMES[s]} pieces left`);
    return;
  }

  // Touch flow: selected size → unique fit → in-cell chooser.
  if (selSize !== null && fits.includes(selSize)) return place(r, c, selSize);
  if (fits.length === 1) return place(r, c, fits[0]);
  openChooser(r, c, fits);
}

function place(r, c, s) {
  if (!makeMove(game, r, c, s)) return;
  lastMove = { r, c, s, player: game.history[game.history.length - 1].player };
  autoSelectSize();
  render();
  updateActionButtons();
  if (game.gameOver) return endGame();
  if (isAiMode() && game.current !== settings.side) aiTurn();
}

// ── Mini chooser (cell offers 2-3 of your sizes) ────────────────────────────
function openChooser(r, c, fits) {
  const chooser = $('cell-chooser');
  const player = game.current;
  chooser.innerHTML = '';
  for (const s of fits) {
    const b = document.createElement('button');
    b.className = 'chooser-btn';
    b.innerHTML = `<svg viewBox="0 0 100 100">${pieceSVG(player, s)}</svg>`;
    b.setAttribute('aria-label', `Place ${SIZE_NAMES[s]}`);
    b.addEventListener('click', (e) => {
      e.stopPropagation();
      closeChooser();
      place(r, c, s);
    });
    chooser.appendChild(b);
  }
  chooser.hidden = false;
  const wrap = $('board-wrap').getBoundingClientRect();
  const cell = cells[r][c].getBoundingClientRect();
  const ch = chooser.getBoundingClientRect();
  let x = cell.left - wrap.left + cell.width / 2 - ch.width / 2;
  x = Math.max(4, Math.min(x, wrap.width - ch.width - 4));
  let y = cell.top - wrap.top - ch.height - 8;
  if (y < 2) y = cell.bottom - wrap.top + 8;
  chooser.style.left = `${x}px`;
  chooser.style.top = `${y}px`;
  chooserCell = cells[r][c];
  chooser.querySelector('button')?.focus();
}
let chooserCell = null;
function closeChooser() {
  const chooser = $('cell-chooser');
  if (chooser.hidden) return;
  const hadFocus = chooser.contains(document.activeElement);
  chooser.hidden = true;
  if (hadFocus && chooserCell && document.contains(chooserCell)) chooserCell.focus();
  chooserCell = null;
}
document.addEventListener('click', (e) => {
  if (!e.target.closest('.cell-chooser') && !e.target.closest('.cell')) closeChooser();
});

// ── AI turn ──────────────────────────────────────────────────────────────────
async function aiTurn() {
  aiBusy = true;
  selSize = null;
  renderStatus();
  renderTrays();
  updateActionButtons();
  // The LLM tier goes over the network to a serverless function instead of
  // the local search worker, so it gets its own path (and no local fallback —
  // silently substituting minimax would misrepresent the experiment).
  if (settings.difficulty === LLM_DIFFICULTY) return llmTurn(reqId + 1);

  const myReq = reqId + 1;   // askWorker increments reqId synchronously
  try {
    const [reply] = await Promise.all([
      askWorker({ type: 'move', difficulty: settings.difficulty }),
      sleep(420),            // let "thinking" read as thinking
    ]);
    if (myReq !== reqId) return;   // superseded by a new game
    applyAiMove(reply.move);
  } catch (err) {
    if (myReq !== reqId) return;   // a new game already superseded this turn
    console.warn('AI worker failed, using main-thread fallback:', err);
    try {
      const { move } = await computeLocally('move');
      if (myReq !== reqId) return;
      applyAiMove(move);
    } catch (err2) {
      console.error(err2);
      aiBusy = false;
      toast('The AI hit an error — press New game');
      autoSelectSize();
      render();
      updateActionButtons();
    }
  }
}

/** LLM tier: POST the position to /api/llm-move and play what comes back. */
async function llmTurn(myReq) {
  try {
    const res = await fetch('/api/llm-move', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ state: snapshot(), effort: 'medium' }),
    });
    const data = await res.json().catch(() => ({}));
    if (myReq !== reqId) return;   // superseded by a new game
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    lastLlmNote = data.reasoning || '';
    if (data.retried) console.warn('LLM needed a correction to find a legal move');
    applyAiMove(data.move);
    if (lastLlmNote) toast(lastLlmNote);
  } catch (err) {
    if (myReq !== reqId) return;
    console.error('LLM move failed:', err);
    aiBusy = false;
    toast(`LLM unavailable: ${err.message}`);
    autoSelectSize();
    render();
    updateActionButtons();
  }
}
let lastLlmNote = '';

function applyAiMove(move) {
  aiBusy = false;
  if (move && makeMove(game, move.r, move.c, move.s)) {
    lastMove = { ...move, player: game.history[game.history.length - 1].player };
  }
  autoSelectSize();
  render();
  updateActionButtons();
  if (game.gameOver) endGame();
}
// A reply is stale if a newer request id has been handed out since.
function pendingInvalid(id) { return id !== reqId; }

// ── Game end ─────────────────────────────────────────────────────────────────
function endGame() {
  if (!scored) {
    scored = true;
    scoredAs = game.winner === RED ? 'red' : game.winner === BLUE ? 'blue' : 'draws';
    scores[scoredAs]++;
    store.set(LS.scores, scores);
  }
  renderScores();
  updateActionButtons();
  setTimeout(showOverlay, 950);   // let the win line draw first
}

function showOverlay() {
  if (!game.gameOver) return;
  const info = winInfo(game);
  const mark = $('result-mark');
  const title = $('result-title');
  const sub = $('result-sub');
  if (game.winner === 0) {
    mark.innerHTML = `<svg viewBox="0 0 100 100">${pieceSVG(RED, 2)}${pieceSVG(BLUE, 1)}${pieceSVG(RED, 0)}</svg>`;
    title.textContent = 'Draw';
    title.style.color = '';
    sub.textContent = 'All 18 pieces placed — nobody connected.';
  } else {
    const w = game.winner;
    mark.innerHTML = `<svg viewBox="0 0 100 100">${pieceSVG(w, 2)}${pieceSVG(w, 1)}${pieceSVG(w, 0)}</svg>`;
    title.textContent = isAiMode()
      ? (w === settings.side ? 'You win!' : 'AI wins')
      : `${colorName(w)} wins!`;
    title.style.color = w === RED ? 'var(--red)' : 'var(--blue)';
    sub.textContent = info?.type === 'bullseye'
      ? `Bullseye — all three sizes stacked at ${POS_NAMES[info.cells[0][0]][info.cells[0][1]]}.`
      : info
        ? `Three ${SIZE_NAMES[info.size]}s through ${POS_NAMES[info.cells[1][0]][info.cells[1][1]]}.`
        : '';
  }
  openModal('overlay', '#btn-rematch');
}

// ── Game lifecycle ───────────────────────────────────────────────────────────
function startNewGame() {
  abandonAiWork();           // invalidate + cancel any in-flight AI search
  aiBusy = false;
  hintBusy = false;
  game = newGame();
  lastMove = null;
  scored = false;
  scoredAs = null;
  winAnimated = false;
  selSize = null;
  closeChooser();
  closeModal('overlay');
  autoSelectSize();
  render();
  renderControls();
  if (isAiMode() && settings.side === BLUE) aiTurn();
}

function undo() {
  if (aiBusy || game.history.length === 0) return;
  if (isAiMode() && !game.history.some((m) => m.player === settings.side)) return;
  closeChooser();
  closeModal('overlay');
  // Undoing out of a finished game takes its recorded result back off the board.
  if (scored && scoredAs) {
    scores[scoredAs] = Math.max(0, scores[scoredAs] - 1);
    store.set(LS.scores, scores);
    renderScores();
  }
  if (isAiMode()) {
    // Rewind through AI replies to the human's last move, then remove it too.
    while (game.history.length && game.history[game.history.length - 1].player !== settings.side) undoMove(game);
    if (game.history.length) undoMove(game);
  } else {
    undoMove(game);
  }
  scored = false;
  scoredAs = null;
  winAnimated = false;
  const prev = game.history[game.history.length - 1] ?? null;
  lastMove = prev ? { ...prev, popped: true } : null;
  autoSelectSize();
  render();
  updateActionButtons();
}

async function hint() {
  if (!humanCanAct() || hintBusy) return;
  hintBusy = true;
  updateActionButtons();
  const plyAtAsk = game.history.length;
  try {
    let reply;
    try {
      reply = await askWorker({ type: 'hint' });
      if (pendingInvalid(reply.id)) return;
    } catch {
      reply = await computeLocally('hint');
    }
    if (!humanCanAct() || game.history.length !== plyAtAsk) return;
    const m = reply.move;
    if (!m) return;
    if (!FINE_POINTER) {
      selSize = m.s;
      render();
      const chip = document.querySelector(`.tray-bottom .size-chip[data-s="${m.s}"]`);
      chip?.classList.add('hinted');
      setTimeout(() => chip?.classList.remove('hinted'), 2900);
    }
    // Glow the exact suggested slot (it carries the player's color class already).
    const slot = cells[m.r][m.c].querySelector(`.slot-open[data-s="${m.s}"]`);
    slot?.classList.add('hint-glow');
    setTimeout(() => slot?.classList.remove('hint-glow'), 2900);
  } catch (err) {
    console.error(err);
    toast('Hint unavailable');
  } finally {
    hintBusy = false;
    updateActionButtons();
  }
}

// ── Settings wiring ──────────────────────────────────────────────────────────
$('seg-mode').addEventListener('click', (e) => {
  const b = e.target.closest('[data-mode]');
  if (!b || b.dataset.mode === settings.mode) return;
  settings.mode = b.dataset.mode;
  store.set(LS.settings, settings);
  startNewGame();
});
$('seg-side').addEventListener('click', (e) => {
  const b = e.target.closest('[data-side]');
  if (!b || +b.dataset.side === settings.side) return;
  settings.side = +b.dataset.side;
  store.set(LS.settings, settings);
  startNewGame();
});
$('seg-difficulty').addEventListener('click', (e) => {
  const b = e.target.closest('[data-diff]');
  if (!b) return;
  settings.difficulty = b.dataset.diff;
  store.set(LS.settings, settings);
  renderControls();
  if (aiBusy) {
    // Supersede the in-flight search so the move really comes at the new level.
    abandonAiWork();
    aiBusy = false;
    aiTurn();
  }
  toast(`Difficulty: ${b.textContent}`);
});

$('btn-new').addEventListener('click', startNewGame);
$('btn-rematch').addEventListener('click', startNewGame);
$('btn-review').addEventListener('click', () => { closeModal('overlay'); });
$('btn-undo').addEventListener('click', undo);
$('btn-hint').addEventListener('click', hint);
$('btn-reset-score').addEventListener('click', () => {
  scores = { red: 0, blue: 0, draws: 0 };
  store.set(LS.scores, scores);
  renderScores();
  toast('Score reset');
});

$('btn-rules').addEventListener('click', () => { openModal('rules-overlay', '#btn-rules-close'); });
$('btn-rules-close').addEventListener('click', () => { closeModal('rules-overlay'); });
for (const ovId of ['overlay', 'rules-overlay']) {
  $(ovId).addEventListener('click', (e) => { if (e.target.id === ovId) closeModal(ovId); });
}

// ── Theme ────────────────────────────────────────────────────────────────────
function applyTheme(t) {
  if (t === 'light' || t === 'dark') document.documentElement.setAttribute('data-theme', t);
  else document.documentElement.removeAttribute('data-theme');
}
$('btn-theme').addEventListener('click', () => {
  // Light is the default everywhere; dark is an explicit opt-in toggle.
  const next = store.get(LS.theme, null) === 'dark' ? 'light' : 'dark';
  store.set(LS.theme, next);
  applyTheme(next);
  toast(`Theme: ${next}`);
});

// ── Keyboard ─────────────────────────────────────────────────────────────────
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.metaKey || e.ctrlKey) return;
  if (e.key === 'Escape') {
    closeChooser();
    closeModal('rules-overlay');
    if (game.gameOver) closeModal('overlay');
    return;
  }
  // Game shortcuts must not fire behind an open dialog ('n' still works on the
  // result overlay, where it reads as rematch).
  if (!$('rules-overlay').hidden) return;
  if (!$('overlay').hidden && e.key.toLowerCase() !== 'n') return;
  if (['1', '2', '3'].includes(e.key)) {
    if (FINE_POINTER) return;   // hover targeting: no size selection
    const s = +e.key - 1;
    if (humanCanAct() && game.pieces[game.current][s] > 0 && legalCellCount(s) > 0) {
      selSize = s;
      render();
    }
    return;
  }
  const k = e.key.toLowerCase();
  if (k === 'n') startNewGame();
  else if (k === 'u') undo();
  else if (k === 'h' && isAiMode()) hint();
});

// ── Boot ─────────────────────────────────────────────────────────────────────
$('tip').textContent = FINE_POINTER
  ? 'Every faint ring is a spot you can play — hover a cell and click the ring you want.'
  : 'Tap a cell with room — if more than one of your pieces fits, pick from the popup.';
applyTheme(store.get(LS.theme, null));
settings = { ...settings, ...store.get(LS.settings, {}) };
if (!['ai', '2p'].includes(settings.mode)) settings.mode = 'ai';
if (![RED, BLUE].includes(settings.side)) settings.side = RED;
if (!Object.hasOwn(DIFF_NOTES, settings.difficulty)) settings.difficulty = 'medium';
scores = { ...scores, ...store.get(LS.scores, {}) };
renderControls();
startNewGame();
