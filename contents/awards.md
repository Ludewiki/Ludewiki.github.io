## 技术文档
<a href="#" onclick="loadLSTMGAN()">点击查看LSTM+GAN详细技术文档</a>
<div id="lstm-content" style="display:none;"></div>

<script>
async function loadLSTMGAN() {
    try {
        // 如果在线，从GitHub加载；如果在本地，从本地加载
        const url = window.location.hostname.includes('github.io') 
            ? 'https://github.com/Ludewiki/Ludewiki.github.io/tree/main/contents/lstm+gan.md'
            : 'lstm+gan.md';
        
        const response = await fetch(url);
        const text = await response.text();
        
        document.getElementById('lstm-content').innerHTML = 
            marked.parse(text);
        document.getElementById('lstm-content').style.display = 'block';
        
        if (typeof MathJax !== 'undefined') {
            MathJax.typeset();
        }
    } catch (error) {
        document.getElementById('lstm-content').innerHTML = 
            '<div class="alert alert-danger">文档加载失败</div>';
        document.getElementById('lstm-content').style.display = 'block';
    }
}
</script>

