/**
 * winnow's hooks: a Claude Code function-hook module ("Claude Mods", early access).
 *
 * Claude Code loads this file when it runs with CLAUDE_CODE_ENABLE_FUNCTION_HOOKS=1;
 * without the flag winnow does nothing, which `winnow doctor` reports. The module
 * wraps every Read, Bash and Grep call in-process: the result goes to the resident
 * sidecar (`winnow serve`) and the sidecar's rewrite comes back as the tool's
 * result. At prompt time it asks the sidecar which context files the prompt needs
 * and appends them to the prompt's context.
 *
 * The sidecar reads the task a call serves from the tail of the session's
 * transcript, a subagent's own by its agent id, so a large result never waits on
 * the engine copying the whole conversation into the module. When the sidecar has
 * stopped answering (it exits after 45 idle minutes), the module starts it again
 * and resends, instead of passing everything through until the next session start.
 */
import type { EngineInterface, Register } from 'claude-code'

/** The sidecar's port. Keep it equal to WINNOW_PORT. */
export const PORT = 47311
export const TOOLS = ['Read', 'Bash', 'Grep'] as const
/** Results shorter than this are never rewritten (the sidecar's WINNOW_MIN_CHARS default); skip the round trip. */
const MIN_CHARS = 1500
/** After the sidecar could not be reached or started, results pass through without a POST for this long. */
export const RETRY_MS = 60_000
/** The same, when it was stopped on purpose with `winnow serve --stop`. */
export const STOPPED_RETRY_MS = 10 * 60_000
/** `winnow serve --revive` exit codes; keep them equal to serve.py's REVIVE_* values. */
const REVIVE_STARTED = 0
const REVIVE_STOPPED = 3
const REVIVE_RUNNING = 4

/** What the sidecar puts in its X-Winnow response header when it rewrote a result. */
export type Meta = { hidden?: number; blocks?: number; before?: number; after?: number; key?: string }

type HookOutput = {
  hookSpecificOutput?: { updatedToolOutput?: unknown; additionalContext?: string }
  systemMessage?: string // the sidecar's once-per-session notice when its judge cannot start
}

type ToolCallEvent = { tool: string; tool_use_id?: string; agentId?: string } & Record<string, unknown>

export function describeMeta(tool: string, meta: Meta): string {
  const k = (n: number | undefined) =>
    n === undefined ? '?' : n >= 10_000 ? `${Math.round(n / 1000)}k` : n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n)
  const key = meta.key === undefined ? '' : `; winnow_recall ${meta.key}`
  return `winnow: hid ${meta.hidden ?? '?'} of ${meta.blocks ?? '?'} blocks of ${tool} (${k(meta.before)} to ${k(meta.after)} chars${key})`
}

type Answer = { output?: HookOutput; meta?: Meta }

/** What one POST came to: nobody answered, or the sidecar's answer (an empty one means pass through). */
export type Sent = { reached: false } | ({ reached: true } & Answer)

/**
 * What `winnow serve --revive` came to: a sidecar started, one that was answering
 * all along (so the POST failed for some other reason), one stopped on purpose
 * with `winnow serve --stop`, or nothing that could run.
 */
export type Revival = 'started' | 'running' | 'stopped' | 'failed'

/**
 * What the module knows about reaching the sidecar, apart from the engine: whether
 * to try at all right now, the restart already under way, and whether a failed
 * restart has been reported. A stopped sidecar must cost nothing per call: a
 * refused connection is not free (on Windows the stack retries the SYN first).
 */
export class Link {
  private quietUntil = 0
  private starting: Promise<Revival> | undefined
  private warned = false
  private readonly now: () => number

  constructor(now: () => number = () => Date.now()) {
    this.now = now
  }

  /** True while results pass through without trying the sidecar. */
  quiet(): boolean {
    return this.now() < this.quietUntil
  }

  /** The restart a call that failed alongside this one already started, if any. */
  pending(): Promise<Revival> | undefined {
    return this.starting
  }

  /** Makes `start` the restart every call that fails while it runs waits on. */
  track(start: Promise<Revival>): Promise<Revival> {
    const shared: Promise<Revival> = start
      .catch((): Revival => 'failed')
      .finally(() => {
        if (this.starting === shared) this.starting = undefined
      })
    this.starting = shared
    return shared
  }

  /** After a restart that did not bring the sidecar back: pass results through untried for a while. */
  settle(revival: Revival): void {
    if (revival === 'running') return // it answers; that POST failed for some other reason
    this.quietUntil = this.now() + (revival === 'stopped' ? STOPPED_RETRY_MS : RETRY_MS)
  }

  /** True the first time a restart fails, so the person hears about it once, not on every call. */
  firstFailure(revival: Revival): boolean {
    if (revival !== 'failed' || this.warned) return false
    this.warned = true
    return true
  }
}

async function send($: EngineInterface, path: string, payload: unknown): Promise<Sent> {
  const url = `http://127.0.0.1:${PORT}${path}`
  let res
  try {
    res = await $.http.fetch(url, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(payload),
    })
  } catch (err) {
    $.ui.log(`winnow: sidecar not answering at ${url}: ${String(err)}`, { to: 'debug' })
    return { reached: false }
  }
  if (!res.ok) {
    $.ui.log(`winnow: sidecar answered ${res.status} at ${url}`, { to: 'debug' })
    return { reached: true }
  }
  let meta: Meta | undefined
  const raw = res.headers['x-winnow']
  if (raw !== undefined) {
    try {
      meta = JSON.parse(raw) as Meta
    } catch {
      meta = undefined
    }
  }
  if (res.text.trim() === '') return { reached: true, meta } // empty 2xx: pass-through
  try {
    return { reached: true, output: JSON.parse(res.text) as HookOutput, meta }
  } catch {
    $.ui.log('winnow: sidecar sent something that is not JSON', { to: 'debug' })
    return { reached: true }
  }
}

/**
 * Runs `winnow serve --revive` with the sidecar's own interpreter: the venv the
 * SessionStart hook's `uv run` made beside the plugin, or `uv run` itself when
 * there is none yet. No shell, so nothing depends on what a shell adds to PATH.
 */
async function startSidecar($: EngineInterface): Promise<Revival> {
  const project = `${$.plugin.root}/sidecar`
  const revive = ['-m', 'winnow', 'serve', '--revive']
  let argv = ['uv', 'run', '-q', '--project', project, 'python', ...revive]
  for (const python of [`${project}/.venv/Scripts/python.exe`, `${project}/.venv/bin/python`]) {
    if (await $.fs.stat(python).then(() => true, () => false)) {
      argv = [python, ...revive]
      break
    }
  }
  try {
    const { exitCode } = await $.process.run(argv, { timeoutMs: 30_000 })
    if (exitCode === REVIVE_STARTED) return 'started'
    if (exitCode === REVIVE_RUNNING) return 'running'
    if (exitCode === REVIVE_STOPPED) return 'stopped'
    $.ui.log(`winnow: serve --revive exited ${exitCode}`, { to: 'debug' })
    return 'failed'
  } catch (err) {
    $.ui.log(`winnow: could not run ${argv[0]}: ${String(err)}`, { to: 'debug' })
    return 'failed'
  }
}

async function whereami($: EngineInterface): Promise<{ session_id: string; cwd: string }> {
  const [session_id, cwd] = await Promise.all([$.session.id().catch(() => ''), $.session.cwd().catch(() => '')])
  return { session_id, cwd }
}

/**
 * One request to the sidecar. A POST that reaches nobody starts the sidecar again,
 * once for every call that failed alongside it, and is sent once more; when that
 * does not bring it back, results pass through untried for a while.
 */
async function ask($: EngineInterface, link: Link, path: string, payload: unknown): Promise<Answer | undefined> {
  if (link.quiet()) return undefined
  const first = await send($, path, payload)
  if (first.reached) return first
  const revival = await (link.pending() ?? link.track(startSidecar($)))
  if (link.firstFailure(revival)) {
    $.ui.toast('winnow: the sidecar is not running and could not be started, so results pass through untouched. Run `winnow doctor` to see why.')
  }
  if (revival === 'started') {
    const again = await send($, path, payload)
    if (again.reached) return again
  }
  link.settle(revival)
  return undefined
}

export const register: Register = (on) => {
  const link = new Link()

  for (const tool of TOOLS) {
    on('tool.call', { tool }, async ($, e, next) => {
      const answer = await next(e)
      if (!('result' in answer) || answer.result === undefined) return answer
      if (JSON.stringify(answer.result).length < MIN_CHARS) return answer
      if (link.quiet()) return answer

      const { tool: toolName, tool_use_id, agentId, ...input } = e as ToolCallEvent
      // No task here: the sidecar reads it from the transcript's tail. `$.session.messages()` would
      // copy the whole conversation on every call (seconds, in a long session), and with no
      // argument it is the main conversation even inside a subagent.
      const res = await ask($, link, '/hook/post-tool-use', {
        hook_event_name: 'PostToolUse',
        source: 'function-hook',
        ...(await whereami($)),
        agent_id: agentId,
        tool_name: toolName,
        tool_input: input,
        tool_response: answer.result,
        tool_use_id,
      })
      if (typeof res?.output?.systemMessage === 'string') $.ui.toast(res.output.systemMessage)
      const updated = res?.output?.hookSpecificOutput?.updatedToolOutput
      if (updated === undefined) return answer
      if (res?.meta !== undefined) $.ui.toast(describeMeta(toolName, res.meta))
      return { ...answer, result: updated }
    })
  }

  on('prompt.submit', async ($, e, next) => {
    if (e.text.trim().length < 12 || link.quiet()) return next(e)
    const res = await ask($, link, '/hook/user-prompt-submit', {
      hook_event_name: 'UserPromptSubmit',
      source: 'function-hook',
      ...(await whereami($)),
      prompt: e.text,
    })
    if (typeof res?.output?.systemMessage === 'string') $.ui.toast(res.output.systemMessage)
    const extra = res?.output?.hookSpecificOutput?.additionalContext
    if (typeof extra !== 'string' || extra === '') return next(e)
    return next({ ...e, context: [...(e.context ?? []), extra] })
  })
}
