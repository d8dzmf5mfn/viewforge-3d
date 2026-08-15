import { spawn } from 'node:child_process'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const bridgeDirectory = dirname(fileURLToPath(import.meta.url))
const viewerRoot = resolve(bridgeDirectory, '..')
const bridge = spawn(process.execPath, [resolve(bridgeDirectory, 'annotation-bridge.mjs')], {
  cwd: viewerRoot,
  env: process.env,
  stdio: 'inherit',
  shell: false,
})
const vite = spawn(process.execPath, [resolve(viewerRoot, 'node_modules/vite/bin/vite.js'), '--host', '127.0.0.1'], {
  cwd: viewerRoot,
  env: process.env,
  stdio: 'inherit',
  shell: false,
})

let shuttingDown = false
function shutdown(code = 0) {
  if (shuttingDown) return
  shuttingDown = true
  bridge.kill('SIGTERM')
  vite.kill('SIGTERM')
  setTimeout(() => process.exit(code), 300).unref()
}

bridge.on('exit', (code) => shutdown(code ?? 1))
vite.on('exit', (code) => shutdown(code ?? 1))
process.on('SIGINT', () => shutdown(0))
process.on('SIGTERM', () => shutdown(0))
