<template>
  <div class="history-page">
    <div class="history-header">
      <h2>翻译历史</h2>
      <label>
        筛选
        <select v-model="filterDirection" @change="loadHistory">
          <option value="">全部</option>
          <option value="en2zh">英 → 中</option>
          <option value="zh2en">中 → 英</option>
        </select>
      </label>
    </div>

    <div v-if="store.historyLoading" class="loading">加载中...</div>

    <div v-else-if="store.history.length === 0" class="empty">
      暂无记录
    </div>

    <div v-else class="history-list">
      <div v-for="item in store.history" :key="item.id" class="history-item">
        <div class="item-source">{{ item.source_text }}</div>
        <div class="item-arrow">→</div>
        <div class="item-target">{{ item.target_text }}</div>
        <div class="item-meta">
          {{ item.source_lang }} → {{ item.target_lang }}
          &middot; {{ item.request_duration_ms ?? '-' }}ms
          &middot; {{ formatTime(item.created_at) }}
        </div>
        <button class="item-delete" @click="store.removeHistory(item.id)">删除</button>
      </div>
    </div>

    <div class="pagination" v-if="store.historyTotal > 20">
      <button :disabled="store.historyPage <= 1" @click="prevPage">上一页</button>
      <span>{{ store.historyPage }} / {{ totalPages }}</span>
      <button :disabled="store.historyPage >= totalPages" @click="nextPage">下一页</button>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useTranslationStore } from '../stores/translation'

const store = useTranslationStore()
const filterDirection = ref('')

const totalPages = computed(() => Math.max(1, Math.ceil(store.historyTotal / 20)))

onMounted(() => loadHistory())

async function loadHistory() {
  await store.loadHistory(store.historyPage, filterDirection.value)
}

function prevPage() {
  if (store.historyPage <= 1) return
  store.historyPage--
  loadHistory()
}

function nextPage() {
  store.historyPage++
  loadHistory()
}

function formatTime(iso) {
  if (!iso) return ''
  return new Date(iso).toLocaleString('zh-CN')
}
</script>

<style scoped>
.history-header {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 16px;
}
.history-header select { margin-left: 6px; padding: 4px 8px; border-radius: 4px; }
.loading, .empty { text-align: center; color: #999; padding: 48px 0; }
.history-item {
  position: relative; padding: 16px; margin-bottom: 12px;
  border: 1px solid #eee; border-radius: 8px;
}
.item-source { color: #555; font-size: 14px; margin-bottom: 6px; }
.item-arrow { color: #1a1a2e; font-weight: bold; margin: 4px 0; }
.item-target { color: #111; font-size: 16px; font-weight: 500; }
.item-meta { font-size: 12px; color: #999; margin-top: 8px; }
.item-delete {
  position: absolute; top: 8px; right: 8px; background: none; border: none;
  color: #ccc; cursor: pointer; font-size: 12px;
}
.item-delete:hover { color: #e74c3c; }
.pagination {
  display: flex; align-items: center; justify-content: center; gap: 16px;
  margin-top: 24px;
}
.pagination button {
  padding: 6px 16px; border: 1px solid #ddd; border-radius: 4px;
  background: #fff; cursor: pointer;
}
.pagination button:disabled { opacity: 0.4; cursor: not-allowed; }
</style>
