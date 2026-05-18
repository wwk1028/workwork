<template>
  <div class="login-container">
    <el-card class="login-card">
      <div class="logo">
        <h1>🌐 中英翻译系统</h1>
        <p>基于Transformer的智能机器翻译</p>
      </div>
      
      <el-tabs v-model="activeTab" class="login-tabs">
        <el-tab-pane label="登录" name="login">
          <el-form :model="loginForm" :rules="loginRules" ref="loginFormRef">
            <el-form-item prop="username">
              <el-input 
                v-model="loginForm.username" 
                placeholder="用户名"
                :prefix-icon="User"
                size="large"
              />
            </el-form-item>
            <el-form-item prop="password">
              <el-input 
                v-model="loginForm.password" 
                type="password" 
                placeholder="密码"
                :prefix-icon="Lock"
                size="large"
                @keyup.enter="handleLogin"
              />
            </el-form-item>
            <el-button 
              type="primary" 
              size="large" 
              @click="handleLogin" 
              :loading="loginLoading"
              block
            >
              登录
            </el-button>
          </el-form>
        </el-tab-pane>
        
        <el-tab-pane label="注册" name="register">
          <el-form :model="registerForm" :rules="registerRules" ref="registerFormRef">
            <el-form-item prop="username">
              <el-input 
                v-model="registerForm.username" 
                placeholder="用户名"
                :prefix-icon="User"
                size="large"
              />
            </el-form-item>
            <el-form-item prop="password">
              <el-input 
                v-model="registerForm.password" 
                type="password" 
                placeholder="密码"
                :prefix-icon="Lock"
                size="large"
              />
            </el-form-item>
            <el-form-item prop="confirmPassword">
              <el-input 
                v-model="registerForm.confirmPassword" 
                type="password" 
                placeholder="确认密码"
                :prefix-icon="Lock"
                size="large"
              />
            </el-form-item>
            <el-form-item prop="email">
              <el-input 
                v-model="registerForm.email" 
                placeholder="邮箱（可选）"
                :prefix-icon="Message"
                size="large"
              />
            </el-form-item>
            <el-button 
              type="primary" 
              size="large" 
              @click="handleRegister" 
              :loading="registerLoading"
              block
            >
              注册
            </el-button>
          </el-form>
        </el-tab-pane>
      </el-tabs>
    </el-card>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { User, Lock, Message } from '@element-plus/icons-vue'
import axios from 'axios'

const router = useRouter()
const activeTab = ref('login')

// 登录相关
const loginForm = ref({ username: '', password: '' })
const loginLoading = ref(false)
const loginFormRef = ref()

const loginRules = {
  username: [{ required: true, message: '请输入用户名', trigger: 'blur' }],
  password: [{ required: true, message: '请输入密码', trigger: 'blur' }]
}

// 注册相关
const registerForm = ref({ username: '', password: '', confirmPassword: '', email: '' })
const registerLoading = ref(false)
const registerFormRef = ref()

const registerRules = {
  username: [{ required: true, message: '请输入用户名', trigger: 'blur' }],
  password: [
    { required: true, message: '请输入密码', trigger: 'blur' },
    { min: 6, message: '密码长度至少6位', trigger: 'blur' }
  ],
  confirmPassword: [
    { required: true, message: '请确认密码', trigger: 'blur' },
    {
      validator: (rule, value, callback) => {
        if (value !== registerForm.value.password) {
          callback(new Error('两次输入的密码不一致'))
        } else {
          callback()
        }
      },
      trigger: 'blur'
    }
  ]
}

const API_BASE = '/api'

async function handleLogin() {
  if (!loginFormRef.value) return
  try {
    await loginFormRef.value.validate()
    loginLoading.value = true
    const response = await axios.post(`${API_BASE}/auth/login`, loginForm.value)
    localStorage.setItem('token', response.data.access_token)
    localStorage.setItem('username', response.data.username)
    ElMessage.success('登录成功')
    router.push('/')
  } catch (error) {
    if (error.response) {
      ElMessage.error(error.response?.data?.detail || '登录失败')
    }
  } finally {
    loginLoading.value = false
  }
}

async function handleRegister() {
  if (!registerFormRef.value) return
  try {
    await registerFormRef.value.validate()
    registerLoading.value = true
    await axios.post(`${API_BASE}/auth/register`, {
      username: registerForm.value.username,
      password: registerForm.value.password,
      email: registerForm.value.email
    })
    ElMessage.success('注册成功，请登录')
    activeTab.value = 'login'
    loginForm.value = { username: registerForm.value.username, password: '' }
  } catch (error) {
    if (error.response) {
      ElMessage.error(error.response?.data?.detail || '注册失败')
    }
  } finally {
    registerLoading.value = false
  }
}
</script>

<style scoped>
.login-container {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 100vh;
  padding: 20px;
}

.login-card {
  width: 450px;
  border-radius: 16px;
  box-shadow: 0 20px 35px rgba(0, 0, 0, 0.2);
}

.logo {
  text-align: center;
  margin-bottom: 30px;
}

.logo h1 {
  font-size: 28px;
  color: #333;
  margin-bottom: 10px;
}

.logo p {
  color: #666;
  font-size: 14px;
}

.login-tabs :deep(.el-tabs__header) {
  margin-bottom: 25px;
}

.login-tabs :deep(.el-tabs__item) {
  font-size: 16px;
  font-weight: 500;
}

:deep(.el-button--block) {
  width: 100%;
}
</style>