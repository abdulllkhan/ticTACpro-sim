/**
 * Shared Claude move-picker. Imported by the HTTP route (api/llm-move.js) and
 * by the benchmark (tests_web/llm_bench.mjs) so both exercise the identical
 * prompt, tool schema and retry policy — benchmark numbers describe what the
 * deployed game actually does.
 *
 * Files under api/ whose name starts with "_" are not routed by Vercel.
 */
import Anthropic from '@anthropic-ai/sdk';
import { isLegal } from '../web/js/engine.js';
import { describePosition, parseMoveId, moveId, SYSTEM_PROMPT } from '../web/js/serialize.js';

export const DEFAULT_MODEL = 'claude-opus-5';
export const EFFORTS = ['low', 'medium', 'high', 'xhigh', 'max'];

const MOVE_TOOL = {
  name: 'submit_move',
  description: 'Submit your chosen move. Call this exactly once.',
  strict: true,
  input_schema: {
    type: 'object',
    properties: {
      move: {
        type: 'string',
        description: 'A move id from the legal move list, e.g. "CC-S" or "TL-M".',
      },
      reasoning: {
        type: 'string',
        description: 'One sentence explaining why this move was chosen.',
      },
    },
    required: ['move', 'reasoning'],
    additionalProperties: false,
  },
};

let client = null;
const getClient = () => (client ??= new Anthropic());   // reads ANTHROPIC_API_KEY

const readMove = (response) =>
  response.content.find((b) => b.type === 'tool_use' && b.name === 'submit_move')?.input ?? null;

/**
 * Ask Claude for a move. Throws on API failure or if no legal move is produced
 * after one corrective round-trip.
 *
 * @returns {{move, moveId, reasoning, retried, usage, ms}}
 */
export async function chooseLlmMove(game, { effort = 'medium', model = DEFAULT_MODEL } = {}) {
  const t0 = Date.now();
  const { text, legal } = describePosition(game);

  const request = {
    model,
    max_tokens: 8000,
    output_config: { effort },
    // Stable prefix — cached, so later moves in a game re-read it at ~0.1x price.
    system: [{ type: 'text', text: SYSTEM_PROMPT, cache_control: { type: 'ephemeral' } }],
    tools: [MOVE_TOOL],
    tool_choice: { type: 'tool', name: 'submit_move' },
    messages: [{ role: 'user', content: text }],
  };

  let response = await getClient().messages.create(request);
  let picked = readMove(response);
  let chosen = picked && parseMoveId(picked.move);
  let retried = false;

  // The illegal-move rate is a result worth measuring, so it is counted rather
  // than silently repaired.
  if (!chosen || !isLegal(game, chosen.r, chosen.c, chosen.s)) {
    retried = true;
    response = await getClient().messages.create({
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
            content: `"${picked?.move ?? '(none)'}" is not a legal move here. `
              + `Choose one of exactly these: ${legal.join(' ')}`,
          }],
        },
      ],
    });
    picked = readMove(response);
    chosen = picked && parseMoveId(picked.move);
  }

  if (!chosen || !isLegal(game, chosen.r, chosen.c, chosen.s)) {
    const err = new Error(`Model produced no legal move (last attempt: ${picked?.move ?? 'none'})`);
    err.attempted = picked?.move ?? null;
    err.legal = legal;
    throw err;
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
    ms: Date.now() - t0,
  };
}
