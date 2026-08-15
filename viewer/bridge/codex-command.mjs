export function buildCodexExecArgs({ projectRoot, finalPath }) {
  return [
    'exec', '--cd', projectRoot,
    '--approve-for-me',
    '--skip-git-repo-check',
    '--json',
    '--output-last-message', finalPath,
    '-',
  ]
}

export function classifyCodexFinalOutcome(finalMessage) {
  const outcome = String(finalMessage ?? '').match(
    /(?:^|\n)\s*(?:[-*>]\s*)?(?:#{1,6}\s*)?Outcome\s*:\s*[`*_~]*([a-z][a-z0-9_-]*)/im,
  )?.[1]?.toLowerCase()

  return ['blocked', 'blocked_tooling', 'failed', 'failure', 'cancelled'].includes(outcome)
    ? 'failed'
    : 'completed'
}
