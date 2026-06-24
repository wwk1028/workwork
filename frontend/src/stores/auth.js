import { defineStore } from 'pinia'
import { login as apiLogin, register as apiRegister } from '../api'

export const useAuthStore = defineStore('auth', {
  state: () => ({
    user: null,
    token: localStorage.getItem('access_token') || '',
  }),

  getters: {
    isLoggedIn: (state) => !!state.token,
    isAdmin: (state) => state.user?.roles?.includes('admin'),
  },

  actions: {
    async login(username, password) {
      const { data } = await apiLogin(username, password)
      this.token = data.access_token
      this.user = data.user
      localStorage.setItem('access_token', data.access_token)
    },

    async register(username, password, email, fullName) {
      await apiRegister(username, password, email, fullName)
    },

    logout() {
      this.token = ''
      this.user = null
      localStorage.removeItem('access_token')
    },
  },
})
