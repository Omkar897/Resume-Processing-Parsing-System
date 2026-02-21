(function () {
    const root = document.documentElement;
    const glow = document.getElementById("cursorGlow");
    const auroraA = document.querySelector(".aurora-a");
    const auroraB = document.querySelector(".aurora-b");
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    let mx = window.innerWidth * 0.5;
    let my = window.innerHeight * 0.5;
    let pmx = mx;
    let pmy = my;
    let mouseSpeed = 0;
    let gx = mx;
    let gy = my;

    if (glow && !window.matchMedia("(pointer: coarse)").matches) {
        glow.style.opacity = "1";
    }

    function updatePointer(e) {
        pmx = mx;
        pmy = my;
        mx = e.clientX;
        my = e.clientY;

        const vx = mx - pmx;
        const vy = my - pmy;
        mouseSpeed = Math.min(42, Math.sqrt(vx * vx + vy * vy));

        const px = ((mx / window.innerWidth) - 0.5) * 42;
        const py = ((my / window.innerHeight) - 0.5) * 42;

        if (auroraA) {
            auroraA.style.transform = `translate3d(${px * -0.34}px, ${py * -0.28}px, 0)`;
        }
        if (auroraB) {
            auroraB.style.transform = `translate3d(${px * 0.46}px, ${py * 0.32}px, 0)`;
        }

        root.style.setProperty("--mx", `${mx}px`);
        root.style.setProperty("--my", `${my}px`);
    }
    window.addEventListener("mousemove", updatePointer);

    if (glow) {
        function animateGlow() {
            gx += (mx - gx) * 0.14;
            gy += (my - gy) * 0.14;
            glow.style.left = `${gx}px`;
            glow.style.top = `${gy}px`;
            requestAnimationFrame(animateGlow);
        }
        requestAnimationFrame(animateGlow);
    }

    const revealItems = Array.from(document.querySelectorAll("[data-reveal]"));
    if ("IntersectionObserver" in window && revealItems.length > 0) {
        const observer = new IntersectionObserver(
            (entries) => {
                entries.forEach((entry) => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add("visible");
                        observer.unobserve(entry.target);
                    }
                });
            },
            { threshold: 0.12 }
        );
        revealItems.forEach((item, idx) => {
            item.style.transitionDelay = `${Math.min(idx * 0.05, 0.35)}s`;
            observer.observe(item);
        });
    } else {
        revealItems.forEach((item) => item.classList.add("visible"));
    }

    const surfaces = Array.from(document.querySelectorAll(
        ".feature-card, .upload-box, .job-card, .email-section, .resume-insights-section, .actions"
    ));
    surfaces.forEach((el) => {
        el.classList.add("interactive-surface");
        el.addEventListener("mousemove", (e) => {
            const rect = el.getBoundingClientRect();
            const x = ((e.clientX - rect.left) / rect.width) * 100;
            const y = ((e.clientY - rect.top) / rect.height) * 100;
            el.style.setProperty("--hx", `${x}%`);
            el.style.setProperty("--hy", `${y}%`);
        });
    });

    const canvas = document.getElementById("particleField");
    if (!canvas || reducedMotion) {
        return;
    }

    const ctx = canvas.getContext("2d", { alpha: true });
    if (!ctx) {
        return;
    }

    let dpr = Math.min(window.devicePixelRatio || 1, 2);
    let width = 0;
    let height = 0;
    let spacing = 76;
    let rows = 0;
    let cols = 0;
    let mesh = [];
    let t = 0;

    function clamp(v, min, max) {
        return Math.max(min, Math.min(max, v));
    }

    function setupMesh() {
        dpr = Math.min(window.devicePixelRatio || 1, 2);
        width = window.innerWidth;
        height = window.innerHeight;

        canvas.width = Math.floor(width * dpr);
        canvas.height = Math.floor(height * dpr);
        canvas.style.width = `${width}px`;
        canvas.style.height = `${height}px`;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        spacing = width < 700 ? 56 : 72;
        cols = Math.ceil(width / spacing) + 2;
        rows = Math.ceil(height / spacing) + 2;

        mesh = [];
        for (let r = 0; r < rows; r += 1) {
            const row = [];
            const offset = r % 2 === 0 ? 0 : spacing * 0.5;
            for (let c = 0; c < cols; c += 1) {
                const bx = c * spacing + offset - spacing * 0.5;
                const by = r * spacing - spacing * 0.5;
                row.push({
                    bx,
                    by,
                    x: bx,
                    y: by,
                    vx: 0,
                    vy: 0,
                    phase: Math.random() * Math.PI * 2,
                    jitter: 0.35 + Math.random() * 0.85,
                    size: 0.65 + Math.random() * 1.05,
                });
            }
            mesh.push(row);
        }
    }

    function drawLink(a, b, radius, breakPower) {
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const span = spacing * 1.45;
        if (dist > span) {
            return;
        }

        const mxLine = (a.x + b.x) * 0.5;
        const myLine = (a.y + b.y) * 0.5;
        const mdx = mxLine - mx;
        const mdy = myLine - my;
        const mDist = Math.sqrt(mdx * mdx + mdy * mdy);

        const nearMouse = mDist < radius;
        const crack = nearMouse ? clamp(1 - mDist / radius, 0, 1) : 0;

        const strainA = Math.hypot(a.x - a.bx, a.y - a.by);
        const strainB = Math.hypot(b.x - b.bx, b.y - b.by);
        const strain = (strainA + strainB) * 0.04;

        let alpha = 0.28 * (1 - dist / span);
        alpha *= (1 - crack * breakPower);
        alpha *= clamp(1 - strain * 0.24, 0.2, 1);

        if (alpha < 0.015) {
            return;
        }

        if (crack > 0.75 && Math.random() < crack * 0.35) {
            return;
        }

        const hue = 177 + Math.sin(t * 0.9 + a.phase) * 20;
        ctx.strokeStyle = `hsla(${hue}, 95%, 74%, ${alpha})`;
        ctx.lineWidth = 0.9;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
    }

    function animateMesh() {
        t += 0.016;
        mouseSpeed *= 0.9;

        ctx.clearRect(0, 0, width, height);

        const influenceRadius = 180 + mouseSpeed * 5.5;
        const breakPower = 0.92 + mouseSpeed * 0.012;

        for (let r = 0; r < rows; r += 1) {
            for (let c = 0; c < cols; c += 1) {
                const p = mesh[r][c];

                const breatheX = Math.cos(t * 1.35 + p.phase) * p.jitter;
                const breatheY = Math.sin(t * 1.22 + p.phase * 1.11) * p.jitter;

                p.vx += (p.bx - p.x) * 0.026 + breatheX * 0.01;
                p.vy += (p.by - p.y) * 0.026 + breatheY * 0.01;

                const dx = p.x - mx;
                const dy = p.y - my;
                const d2 = dx * dx + dy * dy;

                if (d2 < influenceRadius * influenceRadius) {
                    const dist = Math.sqrt(d2) || 1;
                    const f = (1 - dist / influenceRadius);
                    const burst = 0.62 + mouseSpeed * 0.06;
                    p.vx += (dx / dist) * f * burst;
                    p.vy += (dy / dist) * f * burst;

                    // Extra small jitter near cursor to create fracture-like vibration.
                    p.vx += (Math.random() - 0.5) * f * 0.18;
                    p.vy += (Math.random() - 0.5) * f * 0.18;
                }

                p.vx *= 0.88;
                p.vy *= 0.88;
                p.x += p.vx;
                p.y += p.vy;

                const hue = 181 + Math.sin(t + p.phase) * 14;
                ctx.fillStyle = `hsla(${hue}, 95%, 76%, 0.76)`;
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
                ctx.fill();
            }
        }

        // Draw full mesh links.
        for (let r = 0; r < rows; r += 1) {
            for (let c = 0; c < cols; c += 1) {
                const p = mesh[r][c];

                if (c + 1 < cols) {
                    drawLink(p, mesh[r][c + 1], influenceRadius, breakPower);
                }
                if (r + 1 < rows) {
                    drawLink(p, mesh[r + 1][c], influenceRadius, breakPower);

                    const diagCol = r % 2 === 0 ? c : c + 1;
                    if (diagCol >= 0 && diagCol < cols) {
                        drawLink(p, mesh[r + 1][diagCol], influenceRadius, breakPower);
                    }
                }
            }
        }

        requestAnimationFrame(animateMesh);
    }

    setupMesh();
    window.addEventListener("resize", setupMesh);
    requestAnimationFrame(animateMesh);
})();
