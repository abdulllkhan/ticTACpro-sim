/**
 * Local dev server for trying the "Claude (LLM)" opponent — LOCAL ONLY.
 *
 *   npm install                       # once, for @anthropic-ai/sdk
 *   node dev-server.mjs               # then open http://localhost:8080
 *
 * Why this exists: the Anthropic API cannot be called straight from a browser
 * (no supported CORS path, and a key in page JS is exposed to the page), so
 * this serves web/ and proxies one endpoint, POST /api/llm-move.
 *
 * The key is whatever you supply — either ANTHROPIC_API_KEY in the environment,
 * or pasted into the field in the app, which sends it to THIS local process
 * only. It is never written to disk here and never leaves your machine except
 * in the call to api.anthropic.com.
 *
 * Deliberately not part of the deployed site: the public build on Vercel has no
 * API route and no LLM tier.
 */
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';
import Anthropic from '@anthropic-ai/sdk';

import { reviveGame, isLegal } from './web/js/engine.js';
import { describePosition, parseMoveId, moveId, SYSTEM_PROMPT } from './web/js/serialize.js';

const ROOT = join(fileURLToPath(new URL('.', import.meta.url)), 'web');
const PORT = Number(process.env.PORT) || 8080;
const MODEL = 'claude-opus-5';

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
};

const MOVE_TOOL = {
  name: 'submit_move',
  description: 'Submit your chosen move. Call this exactly once.',
  strict: true,
  input_schema: {
    type: 'object',
    properties: {
      move: { type: 'string', description: 'A legal move id, e.g. "CC-S".' },
      reasoning: { type: 'string', description: 'One sentence on why.' },
    },
    required: ['move', 'reasoning'],
    additionalProperties: false,
  },
};

const readMove = (r) =>
  r.content.find((b) => b.type === 'tool_use' && b.name === 'submit_move')?.input ?? null;

async function pickMove(game, apiKey, effort) {
  const client = new Anthropic({ apiKey });
  const { text, legal } = describePosition(game);
  const request = {
    model: MODEL,
    max_tokens: 8000,
    output_config: { effort },
    // Stable prefix, so later moves in a game re-read the rules from cache.
    system: [{ type: 'text', text: SYSTEM_PROMPT, cache_control: { type: 'ephemeral' } }],
    tools: [MOVE_TOOL],
    tool_choice: { type: 'tool', name: 'submit_move' },
    messages: [{ role: 'user', content: text }],
  };

  let response = await client.messages.create(request);
  let picked = readMove(response);
  let chosen = picked && parseMoveId(picked.move);
  let retried = false;

  if (!chosen || !isLegal(game, chosen.r, chosen.c, chosen.s)) {
    retried = true;
    response = await client.messages.create({
      ...request,
      messages: [
        ...request.messages,
        { role: 'assistant', content: response.content },
        {
          role: 'user',
          content: [{
            type: 'tool_result',
            tool_use_id: response.content.find((b) => b.type === 'tool_use')?.id,
            is_error: true,
            content: `"${picked?.move ?? '(none)'}" is not legal. Pick one of: ${legal.join(' ')}`,
          }],
        },
      ],
    });
    picked = readMove(response);
    chosen = picked && parseMoveId(picked.move);
  }

  if (!chosen || !isLegal(game, chosen.r, chosen.c, chosen.s)) {
    throw new Error(`No legal move produced (last attempt: ${picked?.move ?? 'none'})`);
  }

  return {
    move: chosen,
    moveId: moveId(chosen),
    reasoning: picked.reasoning ?? '',
    retried,
    usage: {
      input: response.usage.input_tokens,
      output: response.usage.output_tokens,
      cacheRead: response.usage.cache_read_input_tokens ?? 0,
      cacheWrite: response.usage.cache_creation_input_tokens ?? 0,
    },
  };
}

/** SDK errors stringify to raw JSON; turn the common ones into a readable line. */
function humanError(err) {
  switch (err?.status) {
    case 401: return 'API key rejected — check the key you saved';
    case 403: return 'API key lacks permission for this model';
    case 429: return 'Rate limited by the API — wait a moment and retry';
    case 400: return `Request rejected: ${err?.error?.error?.message ?? 'bad request'}`;
    default:
      if (err?.status >= 500) return 'Anthropic API error — retry shortly';
      return err?.message?.slice(0, 120) || 'LLM request failed';
  }
}

const json = (res, code, body) => {
  res.writeHead(code, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(body));
};

const server = createServer(async (req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);

  if (url.pathname === '/api/llm-move') {
    if (req.method !== 'POST') return json(res, 405, { error: 'Use POST' });
    let raw = '';
    for await (const chunk of req) raw += chunk;
    try {
      const body = JSON.parse(raw || '{}');
      const apiKey = body.apiKey || process.env.ANTHROPIC_API_KEY;
      if (!apiKey) {
        return json(res, 400, {
          error: 'No API key. Paste one in the app, or start the server with ANTHROPIC_API_KEY set.',
        });
      }
      const game = reviveGame(body.state);
      if (game.gameOver) return json(res, 400, { error: 'Game is already over' });
      const t0 = Date.now();
      const result = await pickMove(game, apiKey, body.effort || 'medium');
      console.log(`  ${result.moveId}  ${Date.now() - t0}ms`
        + `  in ${result.usage.input} out ${result.usage.output}`
        + `  cache r${result.usage.cacheRead}/w${result.usage.cacheWrite}`
        + `${result.retried ? '  [needed a retry]' : ''}`);
      return json(res, 200, result);
    } catch (err) {
      console.error('  llm-move failed:', err.message);
      return json(res, err?.status >= 400 && err.status < 600 ? err.status : 502,
        { error: humanError(err) });
    }
  }

  // Static files out of web/, with traversal blocked.
  const rel = normalize(decodeURIComponent(url.pathname)).replace(/^(\.\.[/\\])+/, '');
  const file = join(ROOT, rel === '/' || rel === '\\' ? 'index.html' : rel);
  if (!file.startsWith(ROOT)) { res.writeHead(403); return res.end('Forbidden'); }
  try {
    const data = await readFile(file);
    res.writeHead(200, { 'Content-Type': MIME[extname(file)] || 'application/octet-stream' });
    res.end(data);
  } catch {
    res.writeHead(404, { 'Content-Type': 'text/plain' });
    res.end('Not found');
  }
});

server.listen(PORT, () => {
  console.log(`TicTacPro dev server → http://localhost:${PORT}`);
  console.log(process.env.ANTHROPIC_API_KEY
    ? 'ANTHROPIC_API_KEY found in the environment — the LLM tier is ready.'
    : 'No ANTHROPIC_API_KEY set — paste a key in the app to use the LLM tier.');
});
