import { createRouter, createWebHistory } from 'vue-router'

function authGuard() {
  const token = localStorage.getItem('access_token')
  if (!token) return '/login'
}

const routes = [
  {
    path: '/',
    name: 'translate',
    component: () => import('../views/TranslateView.vue'),
    beforeEnter: authGuard,
  },
  {
    path: '/history',
    name: 'history',
    component: () => import('../views/HistoryView.vue'),
    beforeEnter: authGuard,
  },
  {
    path: '/terms',
    name: 'terms',
    component: () => import('../views/TermView.vue'),
    beforeEnter: authGuard,
  },
  {
    path: '/tasks',
    name: 'tasks',
    component: () => import('../views/TaskView.vue'),
    beforeEnter: authGuard,
  },
  {
    path: '/login',
    name: 'login',
    component: () => import('../views/LoginView.vue'),
  },
  {
    path: '/register',
    name: 'register',
    component: () => import('../views/RegisterView.vue'),
  },
]

export default createRouter({
  history: createWebHistory(),
  routes,
})
