// lib/convert/gemini.mjs — Gemini CLI 会话 → DSH 会话（纯函数）

import {
  SESSION_FORMAT_VERSION,
  applyBudgetTrim,
  mintSessionId,
  parseJsonlLines,
  parseTime,
  synthesizeSession,
} from './core.mjs'

// 标题归一统一规则：去首尾空白、折叠内部空白；超 80 字符截断加省略号；
// 空白返回空串。core.mjs 属禁改面，各源按文件内联同款（改规则需同步 5 处）。
const TITLE_MAX_LEN = 80
const TITLE_ELLIPSIS = '…'
function normalizeTitle(text) {
  const t = String(text ?? '').trim().replace(/\s+/g, ' ')
  if (!t) return ''
  return t.length <= TITLE_MAX_LEN ? t : t.slice(0, TITLE_MAX_LEN - TITLE_ELLIPSIS.length) + TITLE_ELLIPSIS
}

// 会话文件名签名：新旧两代都是 session-*（旧 .json 单对象、新 .jsonl 检查点）。
// logs.json / .project_root 等伴生不是会话，调用方按此过滤。
export function isGeminiSessionFile(name) {
  return /^session-.+\.jsonl?$/i.test(String(name ?? ''))
}

// <session_context> 是 CLI 注入的环境基座（工作区目录、目录树），不是用户提问：
// 标题与开轮都要跳过它（与 inject.mjs 的注入过滤同语义，本源内联避免依赖链）。
function isSessionContext(text) {
  return /<session_context>/i.test(String(text ?? ''))
}

// user parts → 纯文本（text 块拼接；functionResponse / functionCall 块跳过）。
// 返回 '' 表示无人类文本（工具结果载体或空）。
export function geminiPartsText(content) {
  if (!Array.isArray(content)) return ''
  const parts = []
  for (const p of content) {
    if (p && typeof p === 'object' && typeof p.text === 'string' && p.text.trim()) {
      if (isSessionContext(p.text)) continue
      parts.push(p.text.trim())
    }
  }
  return parts.join('\n')
}

// user 消息里的 functionResponse 载体 → [{ id, name, output }]（无则 []）。
// 新检查点格式把工具结果拆成独立 user 消息（与 toolCalls 内联结果同 id 并存，去重由调用方按 callId 判定）。
function geminiFunctionResponses(content) {
  const out = []
  if (!Array.isArray(content)) return out
  for (const part of content) {
    if (!part || typeof part !== 'object') continue
    const fr = part.functionResponse
    if (!fr || typeof fr !== 'object') continue
    const id = typeof fr.id === 'string' && fr.id ? fr.id : null
    if (!id) continue
    const resp = fr.response && typeof fr.response === 'object' ? fr.response.output : undefined
    out.push({ id, name: typeof fr.name === 'string' ? fr.name : 'unknown', output: typeof resp === 'string' ? resp : null })
  }
  return out
}

// gemini 消息 content 为 [{ functionCall }] 列表时的调用提取（无内联结果，结果随后由
// 独立 user functionResponse 消息到达；thoughtSignature 无语义直接丢弃）。
function geminiFunctionCalls(content) {
  const out = []
  if (!Array.isArray(content)) return out
  for (const part of content) {
    if (!part || typeof part !== 'object') continue
    const fc = part.functionCall
    if (!fc || typeof fc !== 'object') continue
    const id = typeof fc.id === 'string' && fc.id ? fc.id : null
    if (!id) continue
    out.push({ id, name: typeof fc.name === 'string' && fc.name ? fc.name : 'unknown', args: fc.args ?? {} })
  }
  return out
}

// 检查点 JSONL → { header, messages, summary, skipped, skippedLines, secrets }。
// 行形态（实测 ~/.gemini/tmp/<slot>/chats/session-*.jsonl）：
//   头行 { sessionId, projectHash, startTime, lastUpdated, kind }（无 messages 数组）；
//   { $set: { messages: [...] } } 首批全量（含 <session_context> 基座）；
//   单消息 { id, timestamp, type, content, ... }（user 文本 / user functionResponse /
//     gemini 字符串 / gemini functionCall 列表 / info）；
//   { $set: { lastUpdated } } 水位（跳过）；{ $set: { summary } } 会话摘要（标题候选）。
// 同一 callId / 消息 id 在压缩重放后会重复出现（session_context 重注后的全量回放），
// 按 id 首次出现去重（与 zed 的重复 callId 口径一致）。
export function parseGeminiCheckpoint(raw) {
  const { recs, skipped, skippedLines, secrets } = parseJsonlLines(raw)
  let header = null
  let summary = ''
  const messages = []
  const indexById = new Map()
  // 同 id 占位 → 富化（先空后带 toolCalls，实测同 id 连发两行）取富者；
  // 压缩重放的完全重复取首个。富度按 toolCalls / 内容 / thoughts / model 加权。
  const richness = (m) => {
    if (!m || typeof m !== 'object') return 0
    let n = 0
    if (Array.isArray(m.toolCalls)) n += m.toolCalls.length * 10
    if (typeof m.content === 'string' && m.content.trim()) n += 3
    if (Array.isArray(m.content)) n += m.content.length
    if (Array.isArray(m.thoughts)) n += m.thoughts.length
    if (typeof m.model === 'string' && m.model) n += 1
    return n
  }
  const pushMsg = (m) => {
    if (!m || typeof m !== 'object' || typeof m.id !== 'string' || !m.id) {
      messages.push(m)
      return
    }
    const at = indexById.get(m.id)
    if (at === undefined) {
      indexById.set(m.id, messages.length)
      messages.push(m)
      return
    }
    if (richness(m) > richness(messages[at])) messages[at] = m
  }
  for (const rec of recs) {
    if (!rec || typeof rec !== 'object') continue
    if (rec.$set && typeof rec.$set === 'object') {
      const s = rec.$set
      if (Array.isArray(s.messages)) {
        if (!header) header = null
        for (const m of s.messages) pushMsg(m)
      }
      if (typeof s.summary === 'string' && s.summary.trim() && !summary) summary = s.summary.trim()
      continue
    }
    if (rec.sessionId && !rec.type && !rec.id && !Array.isArray(rec.messages)) {
      if (!header) header = rec
      continue
    }
    if (typeof rec.type === 'string') {
      pushMsg(rec)
      continue
    }
    // 未知行：忽略（不计入 skipped，parseJsonlLines 已统计畸形行）
  }
  return { header, messages, summary, skipped, skippedLines, secrets }
}

// 共享轮次构建：messages（新旧两代同形：{ type, content, thoughts, toolCalls, model }）→
// { turns, model, toolCalls, records }。调用方负责 header/meta/标题。
//   - user 文本 → 开新轮（<session_context> 与空文本不开轮）；
//   - user functionResponse → 挂到当前轮最后一步的同 id 调用上（无匹配时挂本轮末步，
//     仍无轮则丢弃计数）；与 toolCalls 内联结果同 id 时去重（只保留一份）；
//   - gemini 字符串正文 + thoughts + toolCalls（含内联结果）→ 当前轮一步；
//   - gemini functionCall 列表 → 同一步内多个 tool-call（结果随后到达）；
//   - info 与未知类型跳过；无 user 先行的 gemini 悬空步丢弃（与旧逻辑一致）。
function buildGeminiTurns(messages) {
  let model = null
  const turns = []
  let cur = null
  let toolCalls = 0
  let orphanToolResults = 0
  let duplicateToolResults = 0
  // callId → { step, hasResult }：user functionResponse 到达时挂回
  const callIndex = new Map()
  for (const msg of messages) {
    if (!msg || typeof msg !== 'object') continue
    if (msg.type === 'user') {
      const prompt = geminiPartsText(msg.content)
      const responses = geminiFunctionResponses(msg.content)
      if (prompt) {
        cur = { prompt, steps: [] }
        turns.push(cur)
        // 新轮开始：旧轮的调用索引保留（结果可能跨轮到达？实测紧随到达，保留更稳妥）
      } else if (responses.length > 0 && cur) {
        const step = cur.steps.length > 0 ? cur.steps[cur.steps.length - 1] : null
        if (!step) {
          orphanToolResults += responses.length
          continue
        }
        for (const r of responses) {
          const hit = callIndex.get(r.id)
          if (hit && hit.hasResult) {
            duplicateToolResults++
            continue
          }
          if (hit) {
            hit.hasResult = true
            step.toolResults.push({
              toolCallId: r.id,
              content: [{ type: 'text', text: r.output ?? '' }],
              isError: false,
            })
          } else {
            // 调用声明在 functionCall 列表中但索引未建（乱序）→ 仍挂本步，记名未知
            orphanToolResults++
            step.toolResults.push({
              toolCallId: r.id,
              content: [{ type: 'text', text: r.output ?? '' }],
              isError: false,
            })
          }
        }
      }
      // 纯 session_context / 空 user：不开轮、不挂结果
    } else if (msg.type === 'gemini' && cur) {
      const step = { content: [], toolCalls: [], toolResults: [] }
      if (!model && typeof msg.model === 'string') model = msg.model
      if (typeof msg.content === 'string' && msg.content.trim()) {
        step.content.push({ type: 'text', text: msg.content.trim() })
      }
      if (Array.isArray(msg.thoughts)) {
        for (const t of msg.thoughts) {
          if (t && typeof t === 'object' && (t.description || t.subject)) {
            step.content.push({
              type: 'reasoning',
              text: [t.subject, t.description].filter(Boolean).join('：'),
            })
          }
        }
      }
      // 新式 functionCall 列表（无内联结果）
      const fcs = geminiFunctionCalls(msg.content)
      for (const fc of fcs) {
        const mapped = { id: fc.id, name: fc.name, arguments: JSON.stringify(fc.args ?? {}) }
        step.content.push({ type: 'tool-call', ...mapped })
        step.toolCalls.push(mapped)
        toolCalls++
        callIndex.set(fc.id, { step, hasResult: false })
      }
      // 旧式 toolCalls（含内联结果，与随后 user functionResponse 同 id）
      if (Array.isArray(msg.toolCalls)) {
        for (const tc of msg.toolCalls) {
          if (!tc || typeof tc !== 'object') continue
          const callId = String(tc.id || 'gemini-' + turns.length + '-' + (cur.steps.length + 1))
          const mapped = {
            id: callId,
            name: tc.name || 'unknown',
            arguments: JSON.stringify(tc.args ?? {}),
          }
          step.content.push({ type: 'tool-call', ...mapped })
          step.toolCalls.push(mapped)
          toolCalls++
          const text = geminiToolResultText(tc)
          const hasInline = text !== null
          callIndex.set(callId, { step, hasResult: hasInline })
          if (hasInline) {
            step.toolResults.push({
              toolCallId: callId,
              content: [{ type: 'text', text }],
              isError: tc.status === 'error',
            })
          }
        }
      }
      // 空步（无文本/推理/调用：纯 thoughts 空或空字符串）仍保留？旧逻辑 push 空步；
// 新逻辑同样保留（synthesizeSession 会处理空步），但全空且无调用时跳过以减噪。
      if (step.content.length > 0 || step.toolCalls.length > 0) {
        cur.steps.push(step)
      } else if (Array.isArray(msg.thoughts) && msg.thoughts.length > 0) {
        cur.steps.push(step)
      }
    }
    // info 与未知类型跳过；无 cur 的 gemini 悬空步丢弃
  }
  return { turns, model, toolCalls, orphanToolResults, duplicateToolResults }
}

// Gemini CLI 会话 JSON → DSH 会话（旧单对象格式，向后兼容）。
//
// 存储（旧）：~/.gemini/history/<slot>/chats/session-*.json（一文件一 JSON 对象）。
// 顶层：{ sessionId, projectHash, startTime, directories, kind, messages: [...] }。
// 本函数同时接受新检查点 JSONL（见 convertGeminiJsonl）：含换行且首行无 messages
// 数组时自动分流，保证单文件导入按扩展名误判时仍有确定答案。
export function convertGeminiJson(raw, args = {}) {
  const text = String(raw ?? '')
  // 新格式嗅探：多行且首个非空行是头行（含 sessionId 但无 messages 数组）→ 走 JSONL
  const firstLine = text.split('\n').map((l) => l.trim()).find((l) => l.length > 0) || ''
  if (firstLine && text.includes('\n')) {
    try {
      const head = JSON.parse(firstLine)
      if (head && typeof head === 'object' && typeof head.sessionId === 'string' && !Array.isArray(head.messages)) {
        return convertGeminiJsonl(text, args)
      }
    } catch {
      // 首行非 JSON → 走旧路径（返回 skipped=1）
    }
    // 首行是旧消息数组中的一条？旧格式是单对象无换行（或 pretty 换行但顶层含 messages），
    // pretty 旧 JSON 同样多行——需与新 JSONL 区分：顶层解析出 messages 数组即旧格式。
    try {
      const top = JSON.parse(text)
      if (top && typeof top === 'object' && Array.isArray(top.messages)) {
        return convertGeminiObject(top, args, { records: 1, skipped: 0, skippedLines: [], secrets: [] })
      }
    } catch {
      // 既非新头行也非旧顶层 → 可能是新 JSONL（头行解析失败的畸形）→ 走 JSONL 计数路径
      return convertGeminiJsonl(text, args)
    }
  }
  let chat
  try {
    chat = JSON.parse(text)
  } catch {
    return { meta: null, events: [], turns: [], title: undefined, messages: 0, toolCalls: 0, skipped: 1, records: 0, skippedLines: [], secrets: [] }
  }
  if (!chat || typeof chat !== 'object' || !Array.isArray(chat.messages)) {
    return { meta: null, events: [], turns: [], title: undefined, messages: 0, toolCalls: 0, skipped: 1, records: 0, skippedLines: [], secrets: [] }
  }
  return convertGeminiObject(chat, args, { records: chat.messages.length, skipped: 0, skippedLines: [], secrets: [] })
}

// 旧顶层对象 → DSH 会话（convertGeminiJson 的内层；测试可直调）。
function convertGeminiObject(chat, args = {}, counts = {}) {
  const { turns, model } = buildGeminiTurns(chat.messages)

  const sessionId = args.sessionId || mintSessionId(chat.sessionId)
  const meta = { version: SESSION_FORMAT_VERSION, id: sessionId, createdAt: parseTime(chat.startTime) }
  if (chat.sessionId) meta.sourceId = chat.sessionId
  const cwd = args.cwd || (Array.isArray(chat.directories) && chat.directories[0])
  if (cwd) meta.cwd = cwd
  // 标题兜底：Gemini 无显式标题源 → 首问兜底（只回填 out.title，不钉事件）
  const finalTitle = normalizeTitle(turns.length > 0 ? turns[0].prompt : '')
  const { turns: seedTurns, trimmed } = applyBudgetTrim(turns, args.budget)
  const out = synthesizeSession({ meta, turns: seedTurns, title: undefined, provider: 'gemini', model, skipped: 0, records: counts.records ?? chat.messages.length, skippedLines: counts.skippedLines ?? [], secrets: counts.secrets ?? [], imported: { sourcePath: args.sourcePath } })
  const result = trimmed ? { ...out, trimmed } : out
  return { ...result, title: finalTitle }
}

// Gemini CLI 检查点 JSONL → DSH 会话（新格式，~/.gemini/tmp/<slot>/chats/session-*.jsonl）。
//
// 存储（新）：一文件多行 JSONL，头行 { sessionId, projectHash, startTime } + $set 行 +
// 单消息行（详见 parseGeminiCheckpoint）。directories 字段已取消，cwd 由调用方经
// args.cwd 传入（发现层按 slot 回查 history/<slot>/.project_root，单文件导入按源文件
// 目录回退）；缺 cwd 时 meta 不带 cwd（发现层 project 回退 slot 名）。
export function convertGeminiJsonl(raw, args = {}) {
  const { header, messages, summary, skipped, skippedLines, secrets } = parseGeminiCheckpoint(raw)
  if (!header && messages.length === 0) {
    return { meta: null, events: [], turns: [], title: undefined, messages: 0, toolCalls: 0, skipped: 1, records: 0, skippedLines, secrets }
  }
  const { turns, model } = buildGeminiTurns(messages)
  const sessionId = args.sessionId || mintSessionId(header && header.sessionId)
  const meta = { version: SESSION_FORMAT_VERSION, id: sessionId, createdAt: parseTime(header && (header.startTime || header.createdAt)) }
  if (header && header.sessionId) meta.sourceId = header.sessionId
  const cwd = args.cwd || (Array.isArray(header && header.directories) && header.directories[0])
  if (cwd) meta.cwd = cwd
  // 标题：$set.summary 优先（CLI 生成的摘要），回退首问（只回填 out.title，不钉事件）
  const finalTitle = normalizeTitle(summary || (turns.length > 0 ? turns[0].prompt : ''))
  const { turns: seedTurns, trimmed } = applyBudgetTrim(turns, args.budget)
  const records = messages.length + (header ? 1 : 0)
  const out = synthesizeSession({ meta, turns: seedTurns, title: undefined, provider: 'gemini', model, skipped, records, skippedLines, secrets, imported: { sourcePath: args.sourcePath } })
  const result = trimmed ? { ...out, trimmed } : out
  return { ...result, title: finalTitle }
}

// 提取 Gemini 内联工具结果文本；无结果返回 null。
function geminiToolResultText(tc) {
  if (Array.isArray(tc.result)) {
    for (const entry of tc.result) {
      if (!entry || typeof entry !== 'object') continue
      const fr = entry.functionResponse
      const out = fr && fr.response ? fr.response.output : undefined
      if (typeof out === 'string') return out
    }
  }
  if (typeof tc.resultDisplay === 'string') return tc.resultDisplay
  return null
}
