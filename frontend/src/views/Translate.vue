<template>
  <div class="translate-container">
    <!-- 头部导航 -->
    <div class="header">
      <div class="logo">
        <h2>🌐 智能翻译系统</h2>
      </div>
      <div class="nav">
        <el-button :type="$route.path === '/' ? 'primary' : 'text'" @click="goToTranslate">
          翻译
        </el-button>
        <el-button :type="$route.path === '/history' ? 'primary' : 'text'" @click="goToHistory">
          历史记录
        </el-button>
      </div>
      <div class="user-info">
        <el-dropdown @command="handleCommand">
          <span class="user-name">
            {{ username }} <el-icon><ArrowDown /></el-icon>
          </span>
          <template #dropdown>
            <el-dropdown-menu>
              <el-dropdown-item command="logout">退出登录</el-dropdown-item>
            </el-dropdown-menu>
          </template>
        </el-dropdown>
      </div>
    </div>

    <!-- 翻译区域 -->
    <div class="translation-box">
      <div class="input-area">
        <div class="area-header">
          <div class="language-selector">
            <el-select v-model="sourceLang" size="large" style="width: 120px">
              <el-option label="中文" value="zh" />
              <el-option label="English" value="en" />
            </el-select>
            <el-button :icon="Switch" circle @click="swapLanguages" />
            <el-select v-model="targetLang" size="large" style="width: 120px">
              <el-option label="中文" value="zh" />
              <el-option label="English" value="en" />
            </el-select>
          </div>
          <el-button :icon="Delete" circle @click="clearInput" />
        </div>
        <el-input
          v-model="sourceText"
          type="textarea"
          :rows="12"
          placeholder="请输入要翻译的文本..."
          @keydown.ctrl.enter="handleTranslate"
        />
        <div class="footer-info">
          <span class="char-count">{{ sourceText.length }} 字符</span>
          <el-checkbox v-model="useTerminology">使用术语库</el-checkbox>
        </div>
      </div>

      <div class="output-area">
        <div class="area-header">
          <span class="result-label">翻译结果</span>
          <div>
            <el-button :icon="CopyDocument" circle @click="copyResult" />
            <el-button :icon="Refresh" circle @click="resetTranslation" />
          </div>
        </div>
        <div class="result-text">
          <div v-if="translating" class="loading-wrapper">
            <el-icon class="is-loading"><Loading /></el-icon>
            <span>翻译中...</span>
          </div>
          <div v-else class="translated-text">{{ targetText }}</div>
        </div>
        <div class="footer-info" v-if="processingTime">
          <span>耗时: {{ processingTime }} 秒</span>
        </div>
      </div>
    </div>

    <!-- 翻译按钮 -->
    <div class="action-bar">
      <el-button type="primary" size="large" @click="handleTranslate" :loading="translating">
        <el-icon><Promotion /></el-icon>
        翻译 (Ctrl+Enter)
      </el-button>
    </div>

    <!-- 术语库管理 -->
    <el-card class="term-section" shadow="never">
      <template #header>
        <div class="term-header">
          <span>📚 术语库管理</span>
          <el-button type="primary" size="small" @click="showAddTerm = true">
            <el-icon><Plus /></el-icon>
            添加术语
          </el-button>
        </div>
      </template>
      
      <el-table :data="terms" stripe style="width: 100%">
        <el-table-column prop="source_term" label="源语言术语" width="200" />
        <el-table-column prop="target_term" label="目标语言术语" width="200" />
        <el-table-column prop="created_at" label="添加时间" />
        <el-table-column label="操作" width="100">
          <template #default="{ row }">
            <el-button type="danger" size="small" link @click="deleteTerm(row.id)">
              删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <!-- 添加术语对话框 -->
    <el-dialog v-model="showAddTerm" title="添加术语" width="500px">
      <el-form :model="newTerm" label-width="100px">
        <el-form-item label="源语言术语">
          <el-input v-model="newTerm.source_term" placeholder="例如：深度学习" />
        </el-form-item>
        <el-form-item label="目标语言术语">
          <el-input v-model="newTerm.target_term" placeholder="例如：Deep Learning" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showAddTerm = false">取消</el-button>
        <el-button type="primary" @click="addTerm">确定</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Switch, Delete, CopyDocument, Refresh, Loading, Promotion, Plus, ArrowDown } from '@element-plus/icons-vue'
import axios from 'axios'

const router = useRouter()
const username = ref(localStorage.getItem('username') || '')

// 翻译状态
const sourceLang = ref('zh')
const targetLang = ref('en')
const sourceText = ref('')
const targetText = ref('')
const translating = ref(false)
const processingTime = ref(null)
const useTerminology = ref(true)

// 术语库
const terms = ref([])
const showAddTerm = ref(false)
const newTerm = ref({ source_term: '', target_term: '' })

// API 配置
const api = axios.create({
  baseURL: 'http://localhost:8000/api',
  headers: {
    'Authorization': `Bearer ${localStorage.getItem('token')}`
  }
})

// 切换页面
function goToTranslate() {
  router.push('/')
}

function goToHistory() {
  router.push('/history')
}

// 交换语言
function swapLanguages() {
  const temp = sourceLang.value
  sourceLang.value = targetLang.value
  targetLang.value = temp
  
  // 交换文本
  const tempText = sourceText.value
  sourceText.value = targetText.value
  targetText.value = tempText
}

// 清空输入
function clearInput() {
  sourceText.value = ''
  targetText.value = ''
  processingTime.value = null
}

// 重置翻译
function resetTranslation() {
  targetText.value = ''
  processingTime.value = null
}

// 复制结果
async function copyResult() {
  if (targetText.value) {
    try {
      await navigator.clipboard.writeText(targetText.value)
      ElMessage.success('已复制到剪贴板')
    } catch {
      ElMessage.error('复制失败')
    }
  }
}

// 执行翻译
async function handleTranslate() {
  if (!sourceText.value.trim()) {
    ElMessage.warning('请输入要翻译的文本')
    return
  }

  translating.value = true
  try {
    const response = await api.post('/translate/', {
      text: sourceText.value,
      source_lang: sourceLang.value,
      target_lang: targetLang.value,
      use_terminology: useTerminology.value
    })
    
    targetText.value = response.data.target_text
    processingTime.value = response.data.processing_time.toFixed(2)
    ElMessage.success('翻译完成')
  } catch (error) {
    if (error.response?.status === 401) {
      ElMessage.error('登录已过期，请重新登录')
      router.push('/login')
    } else {
      ElMessage.error(error.response?.data?.detail || '翻译失败')
    }
  } finally {
    translating.value = false
  }
}

// 加载术语库
async function loadTerms() {
  try {
    const response = await api.get('/terms/')
    terms.value = response.data
  } catch (error) {
    console.error('加载术语库失败', error)
  }
}

// 添加术语
async function addTerm() {
  if (!newTerm.value.source_term || !newTerm.value.target_term) {
    ElMessage.warning('请填写完整')
    return
  }
  
  try {
    await api.post('/terms/', newTerm.value)
    ElMessage.success('术语添加成功')
    showAddTerm.value = false
    newTerm.value = { source_term: '', target_term: '' }
    loadTerms()
  } catch (error) {
    ElMessage.error(error.response?.data?.detail || '添加失败')
  }
}

// 删除术语
async function deleteTerm(id) {
  try {
    await ElMessageBox.confirm('确定要删除这个术语吗？', '提示', { type: 'warning' })
    await api.delete(`/terms/${id}`)
    ElMessage.success('删除成功')
    loadTerms()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('删除失败')
    }
  }
}

// 退出登录
function handleCommand(command) {
  if (command === 'logout') {
    localStorage.removeItem('token')
    localStorage.removeItem('username')
    router.push('/login')
    ElMessage.success('已退出登录')
  }
}

onMounted(() => {
  loadTerms()
})
</script>

<style scoped>
.translate-container {
  max-width: 1400px;
  margin: 0 auto;
  padding: 20px;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 30px;
  padding-bottom: 20px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.2);
}

.logo h2 {
  color: white;
  margin: 0;
}

.nav {
  display: flex;
  gap: 10px;
}

.nav .el-button {
  color: white;
}

.nav .el-button--primary {
  background: rgba(255, 255, 255, 0.2);
  border-color: transparent;
}

.user-info {
  color: white;
}

.user-name {
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 5px;
}

.translation-box {
  display: flex;
  gap: 20px;
  margin-bottom: 20px;
}

.input-area, .output-area {
  flex: 1;
  background: white;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.area-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  background: #f5f7fa;
  border-bottom: 1px solid #e4e7ed;
}

.language-selector {
  display: flex;
  align-items: center;
  gap: 10px;
}

.result-label {
  font-weight: 600;
  color: #333;
}

.result-text {
  min-height: 280px;
  padding: 16px;
  background: white;
}

.translated-text {
  line-height: 1.6;
  font-size: 16px;
  white-space: pre-wrap;
  word-break: break-word;
}

.loading-wrapper {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 250px;
  gap: 10px;
  color: #909399;
}

.footer-info {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 16px;
  background: #f5f7fa;
  border-top: 1px solid #e4e7ed;
  font-size: 12px;
  color: #909399;
}

.action-bar {
  text-align: center;
  margin-bottom: 30px;
}

.term-section {
  margin-top: 20px;
  border-radius: 12px;
}

.term-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

:deep(.el-textarea__inner) {
  font-size: 16px;
  line-height: 1.6;
  min-height: 280px;
}
</style>