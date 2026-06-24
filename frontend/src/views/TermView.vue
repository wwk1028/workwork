<template>
  <div class="term-page">
    <div class="header">
      <h2>术语库</h2>
      <button @click="showForm = !showForm">+ 新增</button>
    </div>

    <form v-if="showForm" @submit.prevent="handleCreate" class="term-form">
      <input v-model="form.source_term" placeholder="源术语" required />
      <input v-model="form.target_term" placeholder="目标术语" required />
      <select v-model="form.domain">
        <option value="">通用</option><option value="medical">医疗</option>
        <option value="legal">法律</option><option value="tech">技术</option>
      </select>
      <button type="submit">保存</button>
      <button type="button" @click="showForm = false">取消</button>
    </form>

    <div class="filters">
      <input v-model="filterDomain" placeholder="领域" @input="loadTerms" />
    </div>

    <table v-if="terms.length">
      <thead>
        <tr><th>源术语</th><th>目标术语</th><th>领域</th><th>操作</th></tr>
      </thead>
      <tbody>
        <tr v-for="t in terms" :key="t.id">
          <td>{{ t.source_term }}</td>
          <td>{{ t.target_term }}</td>
          <td>{{ t.domain || '-' }}</td>
          <td><button @click="handleDelete(t.id)" class="del">删除</button></td>
        </tr>
      </tbody>
    </table>
    <p v-else class="empty">暂无术语</p>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { getTerms, createTerm, deleteTerm } from '../api'

const terms = ref([])
const showForm = ref(false)
const form = ref({ source_term: '', target_term: '', source_lang: 'en', target_lang: 'zh', domain: '' })
const filterDomain = ref('')

onMounted(() => loadTerms())

async function loadTerms() {
  const { data } = await getTerms(1, 50, filterDomain.value, 'en', 'zh')
  terms.value = data.items
}

async function handleCreate() {
  await createTerm(form.value)
  form.value = { source_term: '', target_term: '', source_lang: 'en', target_lang: 'zh', domain: '' }
  showForm.value = false
  loadTerms()
}

async function handleDelete(id) {
  await deleteTerm(id)
  loadTerms()
}
</script>

<style scoped>
.term-page { max-width: 800px; margin: 0 auto; }
.header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }
.header button { padding: 6px 16px; background: #1a1a2e; color: #fff; border: none; border-radius: 4px; cursor: pointer; }
.term-form { display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }
.term-form input, .term-form select { padding: 6px; border: 1px solid #ccc; border-radius: 4px; }
.term-form button { padding: 6px 12px; border: none; border-radius: 4px; cursor: pointer; }
.filters { margin-bottom: 12px; }
.filters input { padding: 6px; border: 1px solid #ccc; border-radius: 4px; }
table { width: 100%; border-collapse: collapse; }
th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid #eee; }
th { font-size: 13px; color: #666; }
.del { background: none; border: none; color: #e74c3c; cursor: pointer; }
.empty { color: #999; text-align: center; padding: 48px 0; }
</style>
