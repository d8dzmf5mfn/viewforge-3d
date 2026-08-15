import assert from 'node:assert/strict'
import {
  applyEventProgress,
  finishProgress,
  initialProgress,
  submittedProgress,
  timingFor,
} from './job-progress.mjs'

const submittedJob = {
  state: 'submitted',
  createdAt: new Date().toISOString(),
  ...submittedProgress([{ intent: 'lower' }]),
}
assert.equal(submittedJob.stage, 'queued')
assert.equal(submittedJob.stageLabel, '等待接手')
assert.equal(submittedJob.progressDetail, '标注已保存，等待当前任务接手')
assert.equal(timingFor(submittedJob).estimatedSecondsRemaining, null)

const job = {
  state: 'running',
  startedAt: new Date(Date.now() - 120_000).toISOString(),
  ...initialProgress([{ intent: 'polish' }]),
}

applyEventProgress(job, { type: 'thread.started' })
assert.equal(job.stage, 'intake')
applyEventProgress(job, { type: 'item.completed', item: { type: 'agent_message', text: '[PROGRESS stage=edit percent=48 detail=局部抛光中]' } })
assert.equal(job.stage, 'edit')
assert.equal(job.progress, 48)
assert.equal(job.progressDetail, '局部抛光中')
applyEventProgress(job, { type: 'item.started', item: { type: 'command_execution', command: 'sed -n 1,20p file' } })
assert.equal(job.stage, 'edit', 'progress must not regress to inspection')

const inspectingRenderCode = {
  state: 'running',
  ...initialProgress([]),
}
applyEventProgress(inspectingRenderCode, { type: 'turn.started' })
applyEventProgress(inspectingRenderCode, {
  type: 'item.started',
  item: {
    type: 'command_execution',
    command: "rg -n 'render|fixed.?view' viewer/bridge/job-progress.mjs",
  },
})
assert.equal(inspectingRenderCode.stage, 'inspect', 'search text must not be classified as rendering')
assert.equal(inspectingRenderCode.progress, 14)

applyEventProgress(inspectingRenderCode, {
  type: 'item.started',
  item: {
    type: 'command_execution',
    command: '.venv/bin/python scripts/render_glb_neutral_turntable.py --help',
  },
})
assert.equal(inspectingRenderCode.stage, 'render', 'an invoked renderer must still be detected')
assert.equal(inspectingRenderCode.progress, 84)

assert.ok(timingFor(job).estimatedSecondsRemaining > 0)
finishProgress(job, 'completed')
job.state = 'completed'
assert.equal(job.progress, 100)
assert.equal(timingFor(job).estimatedSecondsRemaining, 0)

console.log('annotation bridge progress contract passed')
