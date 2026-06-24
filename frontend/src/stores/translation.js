import { defineStore } from 'pinia'
import { translate, getHistory, deleteHistory } from '../api'

export const useTranslationStore = defineStore('translation', {
  state: () => ({
    direction: 'zh',          // 'zh' → translate to Chinese
    beamSize: 5,
    translating: false,
    lastResult: null,
    // History
    history: [],
    historyTotal: 0,
    historyPage: 1,
    historyLoading: false,
  }),

  actions: {
    async doTranslate(text) {
      this.translating = true
      try {
        const { data } = await translate(text, this.direction, this.beamSize)
        this.lastResult = data
        return data
      } finally {
        this.translating = false
      }
    },

    async loadHistory(page = 1, direction = '') {
      this.historyLoading = true
      try {
        const { data } = await getHistory(page, 20, direction)
        this.history = data.items
        this.historyTotal = data.total
        this.historyPage = data.page
      } finally {
        this.historyLoading = false
      }
    },

    async removeHistory(id) {
      await deleteHistory(id)
      await this.loadHistory(this.historyPage)
    },
  },
})
