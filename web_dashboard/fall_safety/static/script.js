document.addEventListener('DOMContentLoaded', () => {
    // Tab Switching Logic
    const navBtns = document.querySelectorAll('.nav-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    navBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Remove active class from all buttons and contents
            navBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            // Add active class to clicked button and target content
            btn.classList.add('active');
            const targetId = btn.getAttribute('data-tab');
            document.getElementById(targetId).classList.add('active');
        });
    });

    // Start Button Logic
    const startBtn = document.getElementById('startBtn');
    const sysStartBtn = document.getElementById('sysStartBtn');
    const sysStopBtn = document.getElementById('sysStopBtn');
    const statusBadge = document.querySelector('.status-badge');

    function startSystem() {
        statusBadge.textContent = '가동중';
        statusBadge.classList.remove('stopped');
        statusBadge.classList.add('running');
        startBtn.innerHTML = '<i class="fa-solid fa-stop"></i> 중지';
        startBtn.classList.replace('btn-primary', 'btn-danger');
    }

    function stopSystem() {
        statusBadge.textContent = '중지됨';
        statusBadge.classList.remove('running');
        statusBadge.classList.add('stopped');
        startBtn.innerHTML = '<i class="fa-solid fa-play"></i> 시작';
        startBtn.classList.replace('btn-danger', 'btn-primary');
    }

    startBtn.addEventListener('click', () => {
        if (statusBadge.classList.contains('stopped')) {
            startSystem();
        } else {
            stopSystem();
        }
    });

    sysStartBtn.addEventListener('click', startSystem);
    sysStopBtn.addEventListener('click', stopSystem);
});
