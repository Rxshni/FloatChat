(function initLandingParticles() {
  const canvas = document.getElementById("particle-canvas");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  const dots = [];
  const dotCount = 56;

  function between(min, max) {
    return Math.random() * (max - min) + min;
  }

  function resize() {
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;
  }

  function makeDot() {
    return {
      x: between(0, canvas.width),
      y: between(0, canvas.height),
      r: between(0.7, 2.2),
      vx: between(-0.04, 0.04),
      vy: between(-0.06, -0.015),
      alpha: between(0.16, 0.52)
    };
  }

  function seed() {
    dots.length = 0;
    for (let i = 0; i < dotCount; i += 1) dots.push(makeDot());
  }

  function step(dot) {
    dot.x += dot.vx;
    dot.y += dot.vy;

    if (dot.y < -8) {
      dot.y = canvas.height + 8;
      dot.x = between(0, canvas.width);
    }

    if (dot.x < -12) dot.x = canvas.width + 12;
    if (dot.x > canvas.width + 12) dot.x = -12;
  }

  function draw(dot) {
    ctx.beginPath();
    ctx.arc(dot.x, dot.y, dot.r, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(120, 226, 255, ${dot.alpha})`;
    ctx.shadowColor = "rgba(56, 189, 248, 0.55)";
    ctx.shadowBlur = 8;
    ctx.fill();
  }

  function render() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (let i = 0; i < dots.length; i += 1) {
      step(dots[i]);
      draw(dots[i]);
    }
    requestAnimationFrame(render);
  }

  resize();
  seed();
  render();

  window.addEventListener("resize", () => {
    resize();
    seed();
  });
})();

(function initChatPage() {
  const thread = document.getElementById("chat-thread");
  const input = document.getElementById("chat-input");
  const sendBtn = document.getElementById("send-btn");

  if (!thread || !input || !sendBtn) return;

  function addMessage(text, role) {
    const row = document.createElement("div");
    row.className = `message-row ${role === "user" ? "user-row" : "bot-row"}`;

    const bubble = document.createElement("div");
    bubble.className = `message-bubble ${role === "user" ? "user-bubble" : "bot-bubble"}`;
    bubble.textContent = text;

    row.appendChild(bubble);
    thread.appendChild(row);

    row.scrollIntoView({ behavior: "smooth", block: "end" });
  }

  function handleSend() {
    const text = input.value.trim();
    if (!text) return;

    addMessage(text, "user");
    input.value = "";
    input.focus();

    window.setTimeout(() => {
      addMessage("This is a placeholder response from FloatChat.", "bot");
    }, 350);
  }

  sendBtn.addEventListener("click", handleSend);

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      handleSend();
    }
  });
})();
