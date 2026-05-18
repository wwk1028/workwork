<template>
  <div class="history-container">
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

    <!-- 历史记录列表 -->
    <el-card class="history-card" shadow="never">
      <template #header>
        <div class="card-header">
          <span>📋 翻译历史记录</span>
          <el-button type="danger" plain size="small" @click="clearAll" :disabled="historyList.length === 0">
            清空全部
          </el-button>
        </div>
      </template>

      <el-table :data="historyList" stripe v-loading="loading">
        <el-table-column prop="created_at" label="时间" width="160" />
        <el-table-column prop="source_lang" label="源语言" width="80">
          <template #default="{ row }">
            <el-tag :type="row.source_lang === 'zh' ? 'danger' : 'primary'" size="small">
              {{ row.source_lang === 'zh' ? '中文' : 'English' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="target_lang" label="目标语言" width="80">
          <template #default="{ row }">
            <el-tag :type="row.target_lang === 'zh' ? 'danger' : 'primary'" size="small">
              {{ row.target_lang === 'zh' ? '中文' : 'English' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="source_text" label="原文" min-width="200" show-overflow-tooltip />
        <el-table-column prop="target_text" label="译文" min-width="200" show-overflow-tooltip />
        <el-table-column label="操作" width="150" fixed="right">
          <template #default="{ row }">
            <el-button type="primary" size="small" link @click="reTranslate(row)">
              重新翻译
            </el-button>
            <el-button type="danger" size="small" link @click="deleteHistory(row.id)">
              删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>

      <!-- 分页 -->
      <div class="pagination">
        <el-pagination
          v-model:current-page="currentPage"
          v-model:page-size="pageSize"
          :page-sizes="[10, 20, 50]"
          :total="total"
          layout="total, sizes, prev, pager, next, jumper"
          @size-change="loadHistory"
          @current-change="loadHistory"
        />
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { ArrowDown } from '@element-plus/icons-vue'
import axios from 'axios'

const router = useRouter()
const username = ref(localStorage.getItem('username') || '')

const historyList = ref([])
const loading = ref(false)
const currentPage = ref(1)
const pageSize = ref(10)
const total = ref(0)

// API 配置
const api = axios.create({
  baseURL: 'http://localhost:8000/api',
  headers: {
    'Authorization': `Bearer ${localStorage.getItem('token')}`
  }
})

function goToTranslate() {
  router.push('/')
}

function goToHistory() {
  router.push('/history')
}

// 加载历史记录
async function loadHistory() {
  loading.value = true
  try {
    const response = await api.get('/history/', {
      params: {
        skip: (currentPage.value - 1) * pageSize.value,
        limit: pageSize.value
      }
    })
    historyList.value = response.data
    total.value = response.data.length < pageSize.value ? 
      (currentPage.value - 1) * pageSize.value + response.data.length : 
      currentPage.value * pageSize.value + 1
  } catch (error) {
    if (error.response?.status === 401) {
      ElMessage.error('登录已过期，请重新登录')
      router.push('/login')
    } else {
      ElMessage.error('加载历史记录失败')
    }
  } finally {
    loading.value = false
  }
}

// 删除单条记录
async function deleteHistory(id) {
  try {
    await ElMessageBox.confirm('确定要删除这条记录吗？', '提示', { type: 'warning' })
    await api.delete(`/history/${id}`)
    ElMessage.success('删除成功')
    loadHistory()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('删除失败')
    }
  }
}

// 清空所有记录
async function clearAll() {
  try {
    await ElMessageBox.confirm('确定要清空所有历史记录吗？此操作不可恢复！', '警告', { 
      type: 'warning',
      confirmButtonText: '确定',
      cancelButtonText: '取消'
    })
    for (const item of historyList.value) {
      await api.delete(`/history/${item.id}`)
    }
    ElMessage.success('清空成功')
    loadHistory()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('清空失败')
    }
  }
}

// 重新翻译
function reTranslate(row) {
  sessionStorage.setItem('retranslate_text', row.source_text)
  sessionStorage.setItem('retranslate_source', row.source_lang)
  sessionStorage.setItem('retranslate_target', row.target_lang)
  router.push('/')
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
  loadHistory()
})
</script>

<style scoped>
.history-container {
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

.history-card {
  border-radius: 12px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.pagination {
  margin-top: 20px;
  display: flex;
  justify-content: flex-end;
}
</style>