<template>
  <div class="app">
    <header class="app-header" v-if="auth.isLoggedIn">
      <h1>MT 机器翻译</h1>
      <nav>
        <router-link to="/">即时翻译</router-link>
        <router-link to="/history">历史记录</router-link>
        <router-link to="/terms">术语库</router-link>
        <router-link to="/tasks">任务</router-link>
      </nav>
      <div class="user-area">
        <span v-if="auth.user" class="username">{{ auth.user.username }}</span>
        <button @click="handleLogout" class="logout-btn">退出</button>
      </div>
    </header>
    <main class="app-main">
      <router-view />
    </main>
  </div>
</template>

<script setup>
import { useAuthStore } from './stores/auth'
import { useRouter } from 'vue-router'

const auth = useAuthStore()
const router = useRouter()

function handleLogout() {
  auth.logout()
  router.push('/login')
}
</script>

<style scoped>
.app-header {
  display: flex; align-items: center; gap: 24px;
  padding: 0 24px; height: 56px; background: #1a1a2e; color: #eee;
}
.app-header h1 { font-size: 18px; white-space: nowrap; }
.app-header nav { display: flex; gap: 12px; flex: 1; }
.app-header nav a {
  color: #aaa; text-decoration: none; font-size: 14px;
  padding: 4px 12px; border-radius: 4px; transition: 0.2s;
}
.app-header nav a:hover,
.app-header nav a.router-link-active { color: #fff; background: rgba(255,255,255,0.1); }
.user-area { display: flex; align-items: center; gap: 12px; }
.username { font-size: 14px; color: #ccc; }
.logout-btn {
  padding: 4px 12px; background: transparent; color: #aaa;
  border: 1px solid #555; border-radius: 4px; cursor: pointer; font-size: 13px;
}
.logout-btn:hover { color: #fff; border-color: #888; }
.app-main { max-width: 960px; margin: 24px auto; padding: 0 16px; }
</style>
