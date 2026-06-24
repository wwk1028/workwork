import axios from 'axios'

const api = axios.create({ baseURL: '/api/v1', timeout: 30000 })

// Attach JWT token
api.interceptors.request.use(config => {
  const token = localStorage.getItem('access_token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

// ── Auth ──────────────────────────────────────────────────────

export function login(username, password) {
  return api.post('/auth/login', { username, password })
}

export function register(username, password, email, fullName) {
  return api.post('/auth/register', { username, password, email, full_name: fullName })
}

// ── Translate ─────────────────────────────────────────────────

export function translate(text, direction = 'zh', beamSize = 5) {
  return api.post('/translate', { text, direction, beam_size: beamSize })
}

export function getHistory(page = 1, pageSize = 20, direction = '') {
  return api.get('/history', { params: { page, page_size: pageSize, direction } })
}

export function deleteHistory(id) {
  return api.delete(`/history/${id}`)
}

// ── Terms ─────────────────────────────────────────────────────

export function getTerms(page = 1, pageSize = 50, domain = '', sourceLang = '', targetLang = '') {
  return api.get('/terms', { params: { page, page_size: pageSize, domain, source_lang: sourceLang, target_lang: targetLang } })
}

export function createTerm(data) {
  return api.post('/terms', data)
}

export function updateTerm(id, data) {
  return api.put(`/terms/${id}`, data)
}

export function deleteTerm(id) {
  return api.delete(`/terms/${id}`)
}

// ── Tasks ─────────────────────────────────────────────────────

export function getTasks(page = 1, status = '') {
  return api.get('/tasks', { params: { page, status } })
}

export function getTask(uuid) {
  return api.get(`/tasks/${uuid}`)
}

export function createTask(file, taskType, sourceLang, targetLang) {
  const fd = new FormData()
  fd.append('file', file)
  fd.append('task_type', taskType)
  fd.append('source_lang', sourceLang)
  fd.append('target_lang', targetLang)
  return api.post('/tasks', fd, { headers: { 'Content-Type': 'multipart/form-data' } })
}

export default api
