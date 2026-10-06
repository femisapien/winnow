import { describe, expect, test } from 'claude-code/testing'
import type { PromptSubmitInput } from 'claude-code'
import { describeMeta, Link, RETRY_MS, STOPPED_RETRY_MS, type Revival } from '../hooks/winnow'

describe('describeMeta', () => {
  test('names the tool, the counts and the recall key', () => {
    const text = describeMeta('Read', { hidden: 3, blocks: 8, before: 5120, after: 1800, key: 'ab12' })
    expect(text).toBe('winnow: hid 3 of 8 blocks of Read (5.1k to 1.8k chars; winnow_recall ab12)')
  })
})

describe('Link', () => {
  test('after a restart that did not help, results pass through untried for a minute', () => {
    let now = 1_000
    const link = new Link(() => now)
    expect(link.quiet()).toBe(false)
    link.settle('failed')
    expect(link.quiet()).toBe(true)
    now += RETRY_MS - 1
    expect(link.quiet()).toBe(true)
    now += 1
    expect(link.quiet()).toBe(false)
  })

  test('a sidecar stopped on purpose is left alone for longer', () => {
    let now = 0
    const link = new Link(() => now)
    link.settle('stopped')
    now = RETRY_MS
    expect(link.quiet()).toBe(true)
    now = STOPPED_RETRY_MS
    expect(link.quiet()).toBe(false)
  })

  test('a sidecar that was answering all along is no reason to stop trying', () => {
    const link = new Link(() => 0)
    link.settle('running')
    expect(link.quiet()).toBe(false)
  })

  test('calls that fail together share one restart', async () => {
    const link = new Link(() => 0)
    let starts = 0
    let finish: (revival: Revival) => void = () => undefined
    const start = (): Promise<Revival> => {
      starts += 1
      return new Promise<Revival>((resolve) => {
        finish = resolve
      })
    }
    const first = link.pending() ?? link.track(start())
    const second = link.pending() ?? link.track(start())
    expect(starts).toBe(1)
    finish('started')
    expect(await first).toBe('started')
    expect(await second).toBe('started')
    expect(link.pending()).toBe(undefined) // the next failure starts a fresh restart
  })

  test('a restart that throws counts as failed, and is reported once', async () => {
    const link = new Link(() => 0)
    const revival = await link.track(Promise.reject(new Error('no interpreter')))
    expect(revival).toBe('failed')
    expect(link.firstFailure(revival)).toBe(true)
    expect(link.firstFailure(revival)).toBe(false)
    expect(link.firstFailure('stopped')).toBe(false)
  })
})

describe('tool.call', () => {
  test('a large result passes through untouched when the sidecar is not reachable and cannot be started', async ($, on) => {
    const big = { stdout: 'ok\n'.repeat(2000), stderr: '', interrupted: false, isImage: false }
    on('tool.call', { tool: 'Bash' }, () => ({ result: big }))
    const answer = await $.tool.call({ tool: 'Bash', command: 'true' })
    expect(answer.result).toEqual(big)
    // The failed restart quiets the link: a second large result passes through without trying.
    const again = await $.tool.call({ tool: 'Bash', command: 'true' })
    expect(again.result).toEqual(big)
  })

  test('a small result is returned as is', async ($, on) => {
    const small = { type: 'text', file: { filePath: 'a.py', content: 'x = 1\n', numLines: 1, startLine: 1, totalLines: 1 } }
    on('tool.call', { tool: 'Read' }, () => ({ result: small }))
    const answer = await $.tool.call({ tool: 'Read', file_path: 'a.py' })
    expect(answer.result).toEqual(small)
  })

  test('a denied call is left alone', async ($, on) => {
    on('tool.call', { tool: 'Grep' }, () => ({ deny: 'not in tests' }))
    const answer = await $.tool.call({ tool: 'Grep', pattern: 'x' })
    expect(answer).toMatchObject({ deny: 'not in tests' })
  })
})

describe('prompt.submit', () => {
  test('the prompt goes through unchanged when the sidecar is not reachable', async ($, on) => {
    on('prompt.submit', (_$, e) => ({ text: e.text, context: e.context }))
    // The engine fills wait and origin for a plugin's own submission; the declared type asks for them anyway.
    const out = await $.prompt.submit({ text: 'What does the sidecar do when it is down?' } as PromptSubmitInput)
    expect(out).toMatchObject({ text: 'What does the sidecar do when it is down?' })
  })
})
