import assert from 'node:assert/strict'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { buildCodexExecArgs, classifyCodexFinalOutcome } from './codex-command.mjs'

const directory = dirname(fileURLToPath(import.meta.url))
const projectRoot = resolve(directory, '../..')
const finalPath = resolve(directory, 'final-message.md')
const args = buildCodexExecArgs({ projectRoot, finalPath })

assert.ok(args.includes('--approve-for-me'), 'bridge must use reviewed workspace-write execution')
assert.ok(!args.includes('--disable'), 'bridge must keep the required code-mode host enabled')
assert.ok(
  !args.includes('--sandbox'),
  '--approve-for-me and --sandbox are mutually exclusive in codex-cli 0.147',
)
assert.deepEqual(args.slice(-3), ['--output-last-message', finalPath, '-'])

assert.equal(
  classifyCodexFinalOutcome('Outcome: `blocked_tooling`.\nNo geometry changed.'),
  'failed',
)
assert.equal(
  classifyCodexFinalOutcome('Outcome: **blocked—no geometry changes made**.'),
  'failed',
)
assert.equal(
  classifyCodexFinalOutcome('Outcome: completed\nNew immutable experiment created.'),
  'completed',
)

console.log('annotation bridge CLI argument contract passed')
