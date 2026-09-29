// gemini-checkpoint.test.mjs — Gemini CLI 新检查点 JSONL（~/.gemini/tmp/<slot>/chats/session-*.jsonl）
// 覆盖：旧单对象向后兼容、新 JSONL 开轮/工具配对去重、session_context 不开轮、
// 占位→富化同 id 取富者、发现层双根与 tmp 布局、单文件 .jsonl 路由到 gemini。
import { test, beforeEach } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import {
  convertGeminiJson, convertGeminiJsonl, parseGeminiCheckpoint, isGeminiSessionFile,
} from '../lib/convert/gemini.mjs'
import {
  discoverSessions, clearScanCache, clearInflightScans, defaultRoots, layoutProject,
} from '../lib/discovery.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const load = (n) => readFileSync(join(HERE, 'fixtures', n), 'utf8')

beforeEach(() => {
  clearScanCache()
  clearInflightScans()
})

const HOME = join('C:', 'Users', 'tester')
const j = (o) => JSON.stringify(o)

function mockHost(files) {
  const host = {
    async stat(path) {
      const v = files.get(path)
      if (!v) return null
      return v.type === 'dir' ? { type: 'directory' } : { type: 'file', size: v.text.length, mtimeMs: v.mtimeMs }
    },
    async readText(path) {
      const v = files.get(path)
      return v && v.type === 'file' ? v.text : null
    },
    async readHead(path, maxBytes) {
      const v = files.get(path)
      return v && v.type === 'file' ? v.text.slice(0, maxBytes) : null
    },
    async readTail(path, maxBytes) {
      const v = files.get(path)
      return v && v.type === 'file' ? v.text.slice(-maxBytes) : null
    },
    async readDir(path) {
      const sep = String(path).includes('\\') ? '\\' : '/'
      const prefix = String(path).endsWith(sep) ? String(path) : String(path) + sep
      const out = []
      for (const [p, v] of files) {
        if (!p.startsWith(prefix) || p === prefix) continue
        const rest = p.slice(prefix.length)
        if (rest.includes('\\') || rest.includes('/')) continue
        out.push({ name: rest, type: v.type === 'dir' ? 'directory' : 'file', path: p })
      }
      return out.sort((a, b) => a.name.localeCompare(b.name))
    },
    async readSessions() { return null },
  }
  return host
}

test('isGeminiSessionFile：新旧两代 session-* 命中，logs.json 排除', () => {
  assert.equal(isGeminiSessionFile('session-2026-09-25T18-28-abc.json'), true)
  assert.equal(isGeminiSessionFile('session-2026-09-25T18-28-abc.jsonl'), true)
  assert.equal(isGeminiSessionFile('logs.json'), false)
  assert.equal(isGeminiSessionFile('.project_root'), false)
})

test('convertGeminiJsonl：检查点开轮、session_context 不开轮、内联+独立结果去重', () => {
  const out = convertGeminiJsonl(load('gemini-checkpoint.jsonl'), { sourcePath: 'C:\\demo\\session.jsonl', cwd: 'C:\\work\\proj' })
  assert.equal(out.turns.length, 1)
  assert.equal(out.turns[0].prompt, 'analyse the current project')
  assert.equal(out.toolCalls, 1)
  assert.equal(out.title, 'Analyse project layout')
  assert.equal(out.meta.cwd, 'C:\\work\\proj')
  assert.equal(out.meta.sourceId, 'aaaaaaaa-1111-4222-8333-444444444444')
  // 工具调用与结果配对唯一（内联 + 独立 user functionResponse 同 id 去重）
  const calls = out.events.filter((e) => e.type === 'tool/call')
  const results = out.events.filter((e) => e.type === 'tool/result')
  assert.equal(calls.length, 1)
  assert.equal(results.length, 1)
  assert.equal(calls[0].data.name, 'list_directory')
})

test('convertGeminiJson：新 JSONL 自动分流（扩展名误判仍有确定答案）', () => {
  const raw = load('gemini-checkpoint.jsonl')
  const out = convertGeminiJson(raw, { cwd: '/tmp/proj' })
  assert.equal(out.turns.length, 1)
  assert.equal(out.toolCalls, 1)
})

test('parseGeminiCheckpoint：同 id 占位→富化取富者（空 gemini 后带 toolCalls）', () => {
  const raw = [
    j({ sessionId: 's-1', startTime: '2026-09-25T00:00:00.000Z' }),
    j({ id: 'g-1', timestamp: '2026-09-25T00:00:01.000Z', type: 'gemini', content: '' }),
    j({ id: 'g-1', timestamp: '2026-09-25T00:00:01.000Z', type: 'gemini', content: '', toolCalls: [{ id: 'c-1', name: 'read_file', args: {}, status: 'success', result: [{ functionResponse: { id: 'c-1', name: 'read_file', response: { output: 'hi' } } }] }], model: 'm' }),
  ].join('\n')
  const { messages } = parseGeminiCheckpoint(raw)
  assert.equal(messages.length, 1)
  assert.equal(messages[0].toolCalls.length, 1)
})

test('convertGeminiJson：旧单对象向后兼容（directories→cwd）', () => {
  const out = convertGeminiJson(load('gemini-simple.json'), {})
  assert.equal(out.turns.length, 1)
  assert.equal(out.meta.cwd, 'D:\\demo\\gemini-proj')
})

test('discovery：gemini 默认双根（history + tmp），tmp 布局 project/cwd', async () => {
  const roots = defaultRoots({ home: HOME })
  assert.ok(Array.isArray(roots.gemini))
  assert.equal(roots.gemini.length, 2)
  assert.equal(layoutProject('/home/u/.gemini/tmp/slot-a/chats/session-1.jsonl', 'gemini'), 'slot-a')
  assert.equal(layoutProject('/home/u/.gemini/history/slot-a/chats/session-1.json', 'gemini'), 'slot-a')

  const slotDir = join(HOME, '.gemini', 'tmp', 'slot-a', 'chats')
  const sessPath = join(slotDir, 'session-2026-09-25T18-28-abc.jsonl')
  const histRoot = join(HOME, '.gemini', 'history', 'slot-a', '.project_root')
  const files = new Map([
    [join(HOME, '.gemini', 'tmp'), { type: 'dir' }],
    [join(HOME, '.gemini', 'tmp', 'slot-a'), { type: 'dir' }],
    [join(HOME, '.gemini', 'tmp', 'slot-a', 'chats'), { type: 'dir' }],
    [sessPath, { type: 'file', mtimeMs: 1786000000000, text: load('gemini-checkpoint.jsonl') }],
    [histRoot, { type: 'file', mtimeMs: 1786000000000, text: 'C:\\work\\proj' }],
  ])
  const host = mockHost(files)
  const { sessions, total } = await discoverSessions({ path: join(HOME, '.gemini', 'tmp'), format: 'gemini', host, imports: {} })
  assert.equal(total, 1)
  const s = sessions[0]
  assert.equal(s.format, 'gemini')
  assert.equal(s.project, 'proj')
  assert.equal(s.cwd, 'C:\\work\\proj')
  assert.ok(s.title && s.title.length > 0)
})

test('discovery：单文件 session-*.jsonl 路由到 gemini（不落通用 JSONL 回退）', async () => {
  const sessPath = join(HOME, '.gemini', 'tmp', 'slot-a', 'chats', 'session-2026-09-25T18-28-abc.jsonl')
  const files = new Map([
    [sessPath, { type: 'file', mtimeMs: 1786000000000, text: load('gemini-checkpoint.jsonl') }],
  ])
  const host = mockHost(files)
  const { sessions, total } = await discoverSessions({ path: sessPath, host, imports: {} })
  assert.equal(total, 1)
  assert.equal(sessions[0].format, 'gemini')
})
