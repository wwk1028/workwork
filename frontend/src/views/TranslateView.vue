<template>
  <div class="translate-page">
    <!-- Controls -->
    <div class="controls">
      <label class="direction-label">
        方向
        <select v-model="store.direction">
          <option value="zh">英 → 中</option>
          <option value="en">中 → 英</option>
        </select>
      </label>
      <label class="beam-label">
        Beam
        <select v-model="store.beamSize">
          <option :value="1">1</option>
          <option :value="3">3</option>
          <option :value="5">5</option>
          <option :value="10">10</option>
        </select>
      </label>
      <span class="hint">按 Enter 翻译，Shift+Enter 换行</span>
    </div>

    <!-- Input -->
    <div class="io-area">
      <textarea
        ref="inputEl"
        v-model="inputText"
        class="input-box"
        :placeholder="store.direction === 'zh' ? '输入英文...' : '输入中文...'"
        @keydown.enter.exact.prevent="handleTranslate"
        rows="6"
      ></textarea>

    <!-- Output -->
      <div class="output-box" :class="{ loading: store.translating }">
        <span v-if="store.translating" class="loading-text">翻译中...</span>
        <span v-else-if="store.lastResult">
          <div class="result-text">{{ store.lastResult.translation }}</div>
          <div class="result-meta">
            评分 {{ store.lastResult.score.toFixed(3) }}
            &middot; {{ store.lastResult.latency_ms }}ms
          </div>
        </span>
        <span v-else class="placeholder">翻译结果</span>
      </div>
    </div>

    <!-- Button -->
    <button class="btn-translate" :disabled="!inputText.trim() || store.translating"
            @click="handleTranslate">
      {{ store.translating ? '...' : '翻 译' }}
    </button>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useTranslationStore } from '../stores/translation'

const store = useTranslationStore()
const inputEl = ref(null)
const inputText = ref('')

async function handleTranslate() {
  if (!inputText.value.trim() || store.translating) return
  await store.doTranslate(inputText.value)
}
</script>

<style scoped>
.controls {
  display: flex; align-items: center; gap: 16px; margin-bottom: 16px;
}
.controls select {
  margin-left: 4px; padding: 4px 8px; border: 1px solid #ccc; border-radius: 4px;
}
.hint { color: #999; font-size: 12px; margin-left: auto; }
.io-area { display: flex; gap: 16px; }
.input-box, .output-box {
  flex: 1; min-height: 180px; padding: 16px; border: 1px solid #ddd;
  border-radius: 8px; font-size: 15px; line-height: 1.6;
}
.input-box { resize: vertical; font-family: inherit; }
.output-box {
  background: #f9f9f9; white-space: pre-wrap; word-break: break-word;
  display: flex; align-items: center; justify-content: center;
}
.output-box.loading { opacity: 0.6; }
.placeholder { color: #ccc; }
.result-text { font-size: 16px; }
.result-meta { font-size: 12px; color: #999; margin-top: 8px; }
.btn-translate {
  margin-top: 16px; width: 100%; padding: 12px; border: none;
  border-radius: 8px; background: #1a1a2e; color: #fff; font-size: 16px;
  cursor: pointer; transition: 0.2s;
}
.btn-translate:disabled { opacity: 0.5; cursor: not-allowed; }
.btn-translate:hover:not(:disabled) { background: #16213e; }
</style>
