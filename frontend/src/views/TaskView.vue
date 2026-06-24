<template>
  <div class="task-page">
    <h2>翻译任务</h2>

    <form @submit.prevent="submitTask" class="upload-form">
      <div class="form-row">
        <label>
          文件
          <input type="file" @change="e => file = e.target.files[0]" required />
        </label>
        <label>
          方向
          <select v-model="direction">
            <option value="en-zh">英 → 中</option>
            <option value="zh-en">中 → 英</option>
          </select>
        </label>
        <button type="submit" :disabled="uploading">
          {{ uploading ? '上传中...' : '提交翻译任务' }}
        </button>
      </div>
      <p v-if="uploadMsg" class="upload-msg">{{ uploadMsg }}</p>
    </form>

    <div class="filters">
      <select v-model="filterStatus" @change="loadTasks">
        <option value="">全部状态</option>
        <option value="pending">等待中</option>
        <option value="processing">处理中</option>
        <option value="completed">已完成</option>
        <option value="failed">失败</option>
      </select>
      <button @click="loadTasks" class="refresh-btn">刷新</button>
    </div>

    <div v-if="tasks.length">
      <div v-for="t in tasks" :key="t.id" class="task-item">
        <div class="task-header">
          <span :class="'status ' + t.status">{{ statusText(t.status) }}</span>
          <span class="uuid">{{ t.task_uuid?.slice(0, 8) }}...</span>
          <span class="type">{{ t.task_type }}</span>
        </div>
        <div class="task-body">
          {{ t.source_lang }} → {{ t.target_lang }}
          <template v-if="t.input_file_path">
            <br />输入: {{ t.input_file_path }}
          </template>
          <template v-if="t.output_file_path">
            <br />输出: {{ t.output_file_path }}
          </template>
          <span v-if="t.error_message" class="error">{{ t.error_message }}</span>
        </div>
        <div class="task-time">{{ formatTime(t.created_at) }}</div>
      </div>
    </div>
    <p v-else class="empty">暂无任务</p>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { getTasks, createTask } from '../api'

const tasks = ref([])
const filterStatus = ref('')
const file = ref(null)
const direction = ref('en-zh')
const uploading = ref(false)
const uploadMsg = ref('')

onMounted(() => loadTasks())

async function loadTasks() {
  const { data } = await getTasks(1, filterStatus.value)
  tasks.value = data.items
}

async function submitTask() {
  if (!file.value) return
  uploading.value = true
  uploadMsg.value = ''
  try {
    const [sl, tl] = direction.value === 'en-zh' ? ['en', 'zh'] : ['zh', 'en']
    await createTask(file.value, 'document', sl, tl)
    uploadMsg.value = '任务已提交'
    file.value = null
    setTimeout(() => loadTasks(), 500)
  } catch (e) {
    uploadMsg.value = '提交失败: ' + (e.response?.data?.detail || e.message)
  } finally {
    uploading.value = false
  }
}

function statusText(s) {
  return { pending: '等待中', processing: '处理中', completed: '已完成', failed: '失败' }[s] || s
}

function formatTime(iso) {
  return iso ? new Date(iso).toLocaleString('zh-CN') : ''
}
</script>

<style scoped>
.task-page { max-width: 800px; margin: 0 auto; }
h2 { margin-bottom: 16px; }
.upload-form {
  padding: 16px; margin-bottom: 20px; border: 1px solid #eee; border-radius: 8px;
}
.form-row {
  display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
}
.form-row label { font-size: 14px; }
.form-row input[type="file"] { margin-left: 4px; }
.form-row select { margin-left: 4px; padding: 4px 8px; border-radius: 4px; }
.form-row button {
  padding: 8px 20px; background: #1a1a2e; color: #fff;
  border: none; border-radius: 4px; cursor: pointer;
}
.form-row button:disabled { opacity: 0.5; cursor: not-allowed; }
.upload-msg { margin-top: 8px; font-size: 13px; color: #27ae60; }
.filters { display: flex; gap: 8px; margin-bottom: 16px; }
.filters select { padding: 6px; border-radius: 4px; }
.refresh-btn {
  padding: 6px 16px; border: 1px solid #ddd; border-radius: 4px;
  background: #fff; cursor: pointer;
}
.task-item {
  padding: 14px; margin-bottom: 10px; border: 1px solid #eee; border-radius: 8px;
}
.task-header { display: flex; gap: 12px; align-items: center; margin-bottom: 6px; }
.status { padding: 2px 8px; border-radius: 10px; font-size: 12px; color: #fff; }
.status.pending { background: #f39c12; }
.status.processing { background: #3498db; }
.status.completed { background: #27ae60; }
.status.failed { background: #e74c3c; }
.uuid { font-size: 12px; color: #999; }
.type { font-size: 12px; color: #666; }
.task-body { font-size: 14px; color: #555; line-height: 1.6; }
.task-body .error { color: #e74c3c; font-size: 12px; margin-left: 8px; }
.task-time { font-size: 12px; color: #999; margin-top: 6px; }
.empty { color: #999; text-align: center; padding: 48px 0; }
</style>
