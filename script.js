// ==========================================================================
// AHMED NOURELDIN MOHAMED - INTERACTIVE PORTFOLIO & VISUAL ENGINE
// ==========================================================================

document.addEventListener('DOMContentLoaded', () => {
    // 1. Dynamic Typewriter Effect
    const typewriterElement = document.getElementById('typewriter');
    const titles = [
        "Financial Data Analyst",
        "Ex-CIB Data Analytics Intern",
        "Accounting & Quantitative Finance Scholar",
        "Credit Risk & Insolvency Researcher",
        "Algorithmic Trading & Time-Series Modeler"
    ];

    let titleIndex = 0;
    let charIndex = 0;
    let isDeleting = false;
    let typingSpeed = 90;

    function type() {
        const currentTitle = titles[titleIndex];
        
        if (isDeleting) {
            typewriterElement.textContent = currentTitle.substring(0, charIndex - 1);
            charIndex--;
            typingSpeed = 40;
        } else {
            typewriterElement.textContent = currentTitle.substring(0, charIndex + 1);
            charIndex++;
            typingSpeed = 80;
        }

        if (!isDeleting && charIndex === currentTitle.length) {
            isDeleting = true;
            typingSpeed = 1800;
        } else if (isDeleting && charIndex === 0) {
            isDeleting = false;
            titleIndex = (titleIndex + 1) % titles.length;
            typingSpeed = 400;
        }

        setTimeout(type, typingSpeed);
    }

    if (typewriterElement) {
        type();
    }

    // 2. Interactive Canvas Particle Constellation Background
    const canvas = document.getElementById('bg-canvas');
    if (canvas) {
        const ctx = canvas.getContext('2d');
        let width = canvas.width = window.innerWidth;
        let height = canvas.height = window.innerHeight;

        window.addEventListener('resize', () => {
            width = canvas.width = window.innerWidth;
            height = canvas.height = window.innerHeight;
        });

        const particles = [];
        const particleCount = Math.min(width > 768 ? 60 : 30, 70);

        for (let i = 0; i < particleCount; i++) {
            particles.push({
                x: Math.random() * width,
                y: Math.random() * height,
                vx: (Math.random() - 0.5) * 0.4,
                vy: (Math.random() - 0.5) * 0.4,
                radius: Math.random() * 1.8 + 0.8
            });
        }

        function drawParticles() {
            ctx.clearRect(0, 0, width, height);

            for (let i = 0; i < particles.length; i++) {
                const p = particles[i];
                p.x += p.vx;
                p.y += p.vy;

                if (p.x < 0 || p.x > width) p.vx *= -1;
                if (p.y < 0 || p.y > height) p.vy *= -1;

                ctx.beginPath();
                ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(212, 175, 55, 0.4)';
                ctx.fill();

                for (let j = i + 1; j < particles.length; j++) {
                    const p2 = particles[j];
                    const dist = Math.hypot(p.x - p2.x, p.y - p2.y);
                    if (dist < 130) {
                        ctx.beginPath();
                        ctx.moveTo(p.x, p.y);
                        ctx.lineTo(p2.x, p2.y);
                        ctx.strokeStyle = `rgba(212, 175, 55, ${0.18 * (1 - dist / 130)})`;
                        ctx.lineWidth = 0.6;
                        ctx.stroke();
                    }
                }
            }
            requestAnimationFrame(drawParticles);
        }
        drawParticles();
    }

    // 3. Cursor Glow Follower
    const cursorGlow = document.getElementById('cursor-glow');
    if (cursorGlow && window.innerWidth > 900) {
        window.addEventListener('mousemove', (e) => {
            cursorGlow.style.left = `${e.clientX}px`;
            cursorGlow.style.top = `${e.clientY}px`;
        });
    }

    // 4. Animated Number Counters
    const counters = document.querySelectorAll('.stat-num[data-target]');
    let counted = false;

    function runCounters() {
        counters.forEach(counter => {
            const target = +counter.getAttribute('data-target');
            let count = 0;
            const speed = target / 25;

            const updateCount = () => {
                count += speed;
                if (count < target) {
                    counter.innerHTML = Math.ceil(count) + (target === 100 ? '<span>k+</span>' : '+');
                    requestAnimationFrame(updateCount);
                } else {
                    counter.innerHTML = target + (target === 100 ? '<span>k+</span>' : '+');
                }
            };
            updateCount();
        });
    }

    const heroSection = document.getElementById('hero');
    if (heroSection) {
        const counterObserver = new IntersectionObserver((entries) => {
            if (entries[0].isIntersecting && !counted) {
                counted = true;
                runCounters();
            }
        }, { threshold: 0.2 });
        counterObserver.observe(heroSection);
    }

    // 5. Interactive Project Filter Tabs
    const filterBtns = document.querySelectorAll('.filter-btn');
    const projectCards = document.querySelectorAll('.project-card[data-category]');

    filterBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            filterBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const filter = btn.getAttribute('data-filter');

            projectCards.forEach(card => {
                if (filter === 'all' || card.getAttribute('data-category') === filter) {
                    card.style.display = 'flex';
                    setTimeout(() => {
                        card.style.opacity = '1';
                        card.style.transform = 'translateY(0)';
                    }, 50);
                } else {
                    card.style.opacity = '0';
                    card.style.transform = 'translateY(20px)';
                    setTimeout(() => {
                        card.style.display = 'none';
                    }, 300);
                }
            });
        });
    });

    // 6. Mobile Navigation Toggle
    const mobileToggle = document.getElementById('mobile-toggle');
    const navLinks = document.querySelector('.nav-links');

    if (mobileToggle && navLinks) {
        mobileToggle.addEventListener('click', () => {
            navLinks.classList.toggle('active');
            const icon = mobileToggle.querySelector('i');
            if (icon) {
                icon.classList.toggle('fa-bars');
                icon.classList.toggle('fa-xmark');
            }
        });

        navLinks.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', () => {
                navLinks.classList.remove('active');
                const icon = mobileToggle.querySelector('i');
                if (icon) {
                    icon.classList.add('fa-bars');
                    icon.classList.remove('fa-xmark');
                }
            });
        });
    }

    // 7. Navbar scroll blur
    const navbar = document.getElementById('navbar');
    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            navbar.style.padding = '12px 0';
            navbar.style.boxShadow = '0 10px 30px rgba(0, 0, 0, 0.7)';
        } else {
            navbar.style.padding = '16px 0';
            navbar.style.boxShadow = 'none';
        }
    });

    // 8. Render Interactive Financial Charts (Chart.js)
    if (typeof Chart !== 'undefined') {
        // Chart 1: Forex Cumulative Strategy
        const forexCtx = document.getElementById('forexChart');
        if (forexCtx) {
            new Chart(forexCtx, {
                type: 'line',
                data: {
                    labels: ['2000', '2004', '2008', '2012', '2016', '2020', '2024'],
                    datasets: [
                        {
                            label: 'SMA 50/200 Trend Strategy',
                            data: [1.0, 1.28, 1.62, 1.85, 2.15, 2.42, 2.78],
                            borderColor: '#10b981',
                            backgroundColor: 'rgba(16, 185, 129, 0.1)',
                            borderWidth: 2.2,
                            fill: true,
                            tension: 0.35,
                            pointRadius: 3,
                            pointBackgroundColor: '#10b981'
                        },
                        {
                            label: 'Buy & Hold Benchmark',
                            data: [1.0, 1.15, 1.25, 1.08, 1.12, 1.18, 1.14],
                            borderColor: '#64748b',
                            borderDash: [5, 5],
                            borderWidth: 1.5,
                            fill: false,
                            tension: 0.35,
                            pointRadius: 0
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { labels: { color: '#94a3b8', font: { family: 'Plus Jakarta Sans', size: 11 } } },
                        tooltip: { backgroundColor: '#0f172a', borderColor: '#d4af37', borderWidth: 1 }
                    },
                    scales: {
                        x: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#64748b' } },
                        y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#64748b' } }
                    }
                }
            });
        }

        // Chart 2: ROC-AUC Curve for Credit Bankruptcy
        const rocCtx = document.getElementById('rocChart');
        if (rocCtx) {
            new Chart(rocCtx, {
                type: 'line',
                data: {
                    labels: ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'],
                    datasets: [
                        {
                            label: 'Calibrated Random Forest (ROC-AUC = 0.940)',
                            data: [0.0, 0.65, 0.82, 0.89, 0.93, 0.96, 0.98, 0.99, 1.0, 1.0, 1.0],
                            borderColor: '#f59e0b',
                            backgroundColor: 'rgba(245, 158, 11, 0.12)',
                            borderWidth: 2.2,
                            fill: true,
                            tension: 0.3,
                            pointRadius: 3,
                            pointBackgroundColor: '#f59e0b'
                        },
                        {
                            label: 'Random Guessing Baseline',
                            data: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                            borderColor: '#475569',
                            borderDash: [4, 4],
                            borderWidth: 1.2,
                            fill: false,
                            pointRadius: 0
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { labels: { color: '#94a3b8', font: { family: 'Plus Jakarta Sans', size: 11 } } },
                        tooltip: { backgroundColor: '#0f172a', borderColor: '#d4af37', borderWidth: 1 }
                    },
                    scales: {
                        x: { title: { display: true, text: 'False Positive Rate', color: '#64748b' }, grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#64748b' } },
                        y: { title: { display: true, text: 'True Positive Rate', color: '#64748b' }, grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#64748b' } }
                    }
                }
            });
        }

        // Chart 3: Radar Chart for Workforce Risk Drivers
        const radarCtx = document.getElementById('radarChart');
        if (radarCtx) {
            new Chart(radarCtx, {
                type: 'radar',
                data: {
                    labels: ['Overtime Burden', 'Low Monthly Income', 'Tenure < 2 Yrs', 'Commute Distance', 'Role Dissatisfaction', 'Stock Option Level 0'],
                    datasets: [
                        {
                            label: 'Attrition Cohort',
                            data: [92, 78, 85, 70, 80, 75],
                            borderColor: '#f43f5e',
                            backgroundColor: 'rgba(244, 63, 94, 0.2)',
                            borderWidth: 2,
                            pointBackgroundColor: '#f43f5e'
                        },
                        {
                            label: 'Retained Cohort',
                            data: [28, 45, 32, 40, 30, 38],
                            borderColor: '#38bdf8',
                            backgroundColor: 'rgba(56, 189, 248, 0.15)',
                            borderWidth: 2,
                            pointBackgroundColor: '#38bdf8'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { labels: { color: '#94a3b8', font: { family: 'Plus Jakarta Sans', size: 11 } } }
                    },
                    scales: {
                        r: {
                            angleLines: { color: 'rgba(255, 255, 255, 0.08)' },
                            grid: { color: 'rgba(255, 255, 255, 0.08)' },
                            pointLabels: { color: '#cbd5e1', font: { family: 'Plus Jakarta Sans', size: 10 } },
                            ticks: { display: false }
                        }
                    }
                }
            });
        }
    }

    // 9. Scroll Reveal Fade-in
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });

    document.querySelectorAll('.glass, .timeline-item').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'opacity 0.6s cubic-bezier(0.16, 1, 0.3, 1), transform 0.6s cubic-bezier(0.16, 1, 0.3, 1)';
        observer.observe(el);
    });
});
