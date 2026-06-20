const elements = {
  camera: document.querySelector('#camera'), gallery: document.querySelector('#gallery'),
  preview: document.querySelector('#preview'), empty: document.querySelector('#empty'),
  primary: document.querySelector('#primary'), status: document.querySelector('#status'),
  progressWrap: document.querySelector('#progress-wrap'), progressText: document.querySelector('#progress-text')
};

let selectedImage = null;
let searchResult = null;
let previewUrl = null;

function openDatabase() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('face-search', 1);
    request.onupgradeneeded = () => request.result.createObjectStore('state');
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function store(key, value) {
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction('state', 'readwrite');
    transaction.objectStore('state').put(value, key);
    transaction.oncomplete = resolve;
    transaction.onerror = () => reject(transaction.error);
  });
}

async function load(key) {
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const request = db.transaction('state').objectStore('state').get(key);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

function setStatus(message = '', kind = '') {
  elements.status.textContent = message;
  elements.status.className = `status ${kind}`;
}

function showImage(file) {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  elements.preview.src = previewUrl;
  elements.preview.hidden = false;
  elements.empty.hidden = true;
}

function updateButton() {
  elements.primary.textContent = searchResult?.data?.download_url ? '下载' : '匹配';
  elements.primary.disabled = !selectedImage;
}

async function selectImage(file) {
  if (!file) return;
  if (!file.type.startsWith('image/')) {
    setStatus('请选择 JPG、PNG 或 WebP 图片。', 'error');
    return;
  }
  selectedImage = file;
  searchResult = null;
  showImage(file);
  updateButton();
  setStatus('图片已准备好，可以开始匹配。');
  try { await Promise.all([store('image', file), store('result', null)]); } catch (_) {}
}

async function search() {
  elements.primary.disabled = true;
  elements.progressWrap.hidden = false;
  elements.progressText.textContent = '正在匹配并打包照片...';
  setStatus('处理时间取决于照片数量，请保持页面开启。');
  const body = new FormData();
  body.append('image', selectedImage, selectedImage.name || 'photo.jpg');
  try {
    const response = await fetch('/api/v1/face-searches', { method: 'POST', body });
    const result = await response.json().catch(() => null);
    if (!response.ok || !result) throw new Error(result?.message || '请求失败，请稍后重试。');
    if (result.code === 'NO_MATCH_FOUND') {
      searchResult = null;
      await store('result', null);
      setStatus('没有找到您出场的照片，请尝试更清晰的正面照片。', 'error');
      return;
    }
    if (result.code !== 'OK' || !result.data?.download_url) throw new Error(result.message || '匹配失败。');
    searchResult = result;
    await store('result', result);
    setStatus(`已找到 ${result.data.matched_image_count} 张照片，可以下载。`, 'success');
  } catch (error) {
    setStatus(error.message || '请求失败，请稍后重试。', 'error');
  } finally {
    elements.progressWrap.hidden = true;
    updateButton();
  }
}

function download() {
  const url = searchResult?.data?.download_url;
  if (!url) return;
  const link = document.createElement('a');
  link.href = url;
  link.download = '';
  document.body.appendChild(link);
  link.click();
  link.remove();
}

elements.camera.addEventListener('change', event => selectImage(event.target.files[0]));
elements.gallery.addEventListener('change', event => selectImage(event.target.files[0]));
elements.primary.addEventListener('click', () => searchResult?.data?.download_url ? download() : search());

(async function restore() {
  try {
    [selectedImage, searchResult] = await Promise.all([load('image'), load('result')]);
    if (selectedImage) showImage(selectedImage);
    if (searchResult?.data?.download_url) {
      setStatus(`已找到 ${searchResult.data.matched_image_count} 张照片，可以下载。`, 'success');
    }
    updateButton();
  } catch (_) {
    updateButton();
  }
})();
