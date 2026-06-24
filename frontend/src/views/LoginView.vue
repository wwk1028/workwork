<template>
  <div class="login-page">
    <h2>登录</h2>
    <form @submit.prevent="handleLogin">
      <input v-model="username" placeholder="用户名" required />
      <input v-model="password" type="password" placeholder="密码" required />
      <p v-if="error" class="error">{{ error }}</p>
      <button type="submit" :disabled="loading">登录</button>
    </form>
    <p class="link">没有账号？<router-link to="/register">注册</router-link></p>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '../stores/auth'

const router = useRouter()
const auth = useAuthStore()
const username = ref('')
const password = ref('')
const loading = ref(false)
const error = ref('')

async function handleLogin() {
  error.value = ''
  loading.value = true
  try {
    await auth.login(username.value, password.value)
    router.push('/')
  } catch (e) {
    error.value = e.response?.data?.detail || '登录失败'
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.login-page { max-width: 360px; margin: 80px auto; }
h2 { text-align: center; margin-bottom: 24px; }
input { width: 100%; padding: 10px; margin-bottom: 12px; border: 1px solid #ccc; border-radius: 6px; }
button { width: 100%; padding: 10px; background: #1a1a2e; color: #fff; border: none; border-radius: 6px; cursor: pointer; }
button:disabled { opacity: 0.5; }
.error { color: #e74c3c; font-size: 13px; margin-bottom: 8px; }
.link { text-align: center; margin-top: 16px; font-size: 14px; }
</style>
