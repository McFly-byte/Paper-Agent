import axios from 'axios'

const api = axios.create({
  baseURL: '/api/reports',
  headers: {
    'Content-Type': 'application/json'
  }
})

api.interceptors.response.use(
  response => response,
  error => {
    console.error('Reports API Error:', error)
    return Promise.reject(error)
  }
)

export const reportsApi = {
  /** @returns {Promise<import('axios').AxiosResponse<Array>>} */
  list() {
    return api.get('')
  },

  /** @returns {Promise<import('axios').AxiosResponse<{ id: string, title: string, query: string, status: string, createdAt: string, knowledgeBase?: string, content: string }>>} */
  getById(reportId) {
    return api.get(`/${encodeURIComponent(reportId)}`)
  },

  delete(reportId) {
    return api.delete(`/${encodeURIComponent(reportId)}`)
  }
}
