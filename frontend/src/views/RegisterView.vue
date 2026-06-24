<template>
  <div class="register-page">
    <h2>注册</h2>
    <form @submit.prevent="handleRegister">
      <input v-model="username" placeholder="用户名" required />
      <input v-model="email" placeholder="邮箱" />
      <input v-model="fullName" placeholder="姓名" />
      <input v-model="password" type="password" placeholder="密码(6位以上)" required />
      <p v-if="error" class="error">{{ error }}</p>
      <p v-if="ok" class="ok">注册成功！<router-link to="/login">去登录</router-link></p>
      <button type="submit" :disabled="loading">注册</button>
    </form>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useAuthStore } from '../stores/auth'

const auth = useAuthStore()
const username = ref('')
const email = ref('')
const fullName = ref('')
const password = ref('')
const loading = ref(false)
const error = ref('')
const ok = ref(false)

async function handleRegister() {
  error.value = ''
  ok.value = false
  loading.value = true
  try {
    await auth.register(username.value, password.value, email.value, fullName.value)
    ok.value = true
  } catch (e) {
    error.value = e.response?.data?.detail || '注册失败'
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.register-page { max-width: 360px; margin: 80px auto; }
h2 { text-align: center; margin-bottom: 24px; }
input { width: 100%; padding: 10px; margin-bottom: 12px; border: 1px solid #ccc; border-radius: 6px; }
button { width: 100%; padding: 10px; background: #1a1a2e; color: #fff; border: none; border-radius: 6px; cursor: pointer; }
button:disabled { opacity: 0.5; }
.error { color: #e74c3c; font-size: 13px; margin-bottom: 8px; }
.ok { color: #27ae60; font-size: 13px; margin-bottom: 8px; }
</style>
