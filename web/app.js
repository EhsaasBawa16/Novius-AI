/* Novius app - vanilla JS, no dependencies */
(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const feed = $('feed'), chat = $('chat'), input = $('input'), sendBtn = $('send');
  const webbtn = $('webbtn'), msel = $('msel'), msbtn = $('msbtn'), msmenu = $('msmenu');
  const msname = $('msname'), curbadge = $('curbadge'), chatlist = $('chatlist');
  const search = $('search'), whoEl = $('who'), s2b = $('s2b');

  const MARK = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M6 19V5l12 14V5"/></svg>';
  const SEND = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19V5M5 12l7-7 7 7"/></svg>';
  const STOP = '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="5" y="5" width="14" height="14" rx="3"/></svg>';
  const CHEV = '<svg class="chev" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"><path d="M9 6l6 6-6 6"/></svg>';

  const CHIPS = [
    ['Define', 'What is love?'],
    ['Calculate', 'what is 23*7'],
    ['Science', 'what is a black hole?'],
    ['Explore', 'tell me about the sun'],
    ['Casual', 'wsp bro'],
    ['Code', 'write a python function that reverses a list'],
  ];

  let webMode = false, model = 'crusta', busy = false, ctrl = null, stick = true;
  let currentChat = null, pendingRender = false;
  const MODELS = {};

  /* ---------- tiny markdown renderer ---------- */
  const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');

  function inline(s) {
    s = esc(s);
    s = s.replace(/`([^`]+)`/g, '<code class="ic">$1</code>');
    s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    s = s.replace(/\*([^*\n]+)\*/g, '<em>$1</em>');
    s = s.replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener">$1</a>');
    return s;
  }

  function codeBlock(part) {
    let body = part, lang = '';
    const nl = body.indexOf('\n');
    if (nl >= 0) {
      const first = body.slice(0, nl).trim();
      if (/^[a-zA-Z0-9+#-]{1,12}$/.test(first)) { lang = first; body = body.slice(nl + 1); }
    }
    return `<div class="codeblock"><div class="cbhead"><span>${esc(lang || 'code')}</span>` +
      `<button class="cbcopy" type="button">Copy</button></div>` +
      `<pre><code>${esc(body.replace(/\n$/, ''))}</code></pre></div>`;
  }

  function text2html(text) {
    const lines = text.split('\n');
    let html = '', para = [], list = null;
    const flushPara = () => { if (para.length) { html += `<p>${inline(para.join(' '))}</p>`; para = []; } };
    const flushList = () => {
      if (list) { html += `<${list.tag}>${list.items.map((i) => `<li>${inline(i)}</li>`).join('')}</${list.tag}>`; list = null; }
    };
    for (const raw of lines) {
      const line = raw.trimEnd();
      if (!line.trim()) { flushPara(); flushList(); continue; }
      const h = line.match(/^(#{1,3})\s+(.*)/);
      if (h) { flushPara(); flushList(); const n = h[1].length + 2; html += `<h${n}>${inline(h[2])}</h${n}>`; continue; }
      const ul = line.match(/^\s*[-*•]\s+(.*)/);
      if (ul) { flushPara(); if (!list || list.tag !== 'ul') { flushList(); list = { tag: 'ul', items: [] }; } list.items.push(ul[1]); continue; }
      const ol = line.match(/^\s*\d+[.)]\s+(.*)/);
      if (ol) { flushPara(); if (!list || list.tag !== 'ol') { flushList(); list = { tag: 'ol', items: [] }; } list.items.push(ol[1]); continue; }
      const bq = line.match(/^>\s?(.*)/);
      if (bq) { flushPara(); flushList(); html += `<blockquote>${inline(bq[1])}</blockquote>`; continue; }
      flushList(); para.push(line.trim());
    }
    flushPara(); flushList();
    return html;
  }

  function md(src) {
    if (!src) return '';
    const parts = src.split('```');
    let out = '';
    for (let i = 0; i < parts.length; i++) {
      if (i % 2 === 1) out += codeBlock(parts[i]);
      else out += text2html(parts[i]);
    }
    return out;
  }

  /* ---------- helpers ---------- */
  function toast(msg) {
    const t = document.createElement('div');
    t.className = 'toast'; t.textContent = msg;
    $('toasts').appendChild(t);
    setTimeout(() => t.classList.add('out'), 1800);
    setTimeout(() => t.remove(), 2300);
  }

  function sync() {
    sendBtn.innerHTML = busy ? STOP : SEND;
    sendBtn.disabled = !busy && !input.value.trim();
    sendBtn.title = busy ? 'Stop' : 'Send';
  }

  function scroll(force) { if (force || stick) chat.scrollTop = chat.scrollHeight; }

  chat.addEventListener('scroll', () => {
    const dist = chat.scrollHeight - chat.scrollTop - chat.clientHeight;
    stick = dist < 140;
    s2b.classList.toggle('show', dist > 300);
  });
  s2b.onclick = () => scroll(true);

  webbtn.onclick = () => {
    webMode = !webMode;
    webbtn.classList.toggle('on', webMode);
    webbtn.setAttribute('aria-pressed', webMode);
    input.placeholder = webMode ? 'Ask anything - Novius will search the web...' : 'Message Novius...';
  };

  /* ---------- models dropdown ---------- */
  async function loadModels() {
    const models = await (await fetch('/api/models')).json();
    msmenu.innerHTML = '';
    for (const m of models) {
      MODELS[m.id] = m;
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'msitem' + (m.id === model ? ' sel' : '');
      b.innerHTML = `<div class="msrow"><span class="msname2">${m.label}</span>` +
        `<span class="badge${m.ready ? '' : ' off'}">${m.ready ? m.badge : 'not trained'}</span></div>` +
        `<div class="mstag">${m.tagline} &middot; ${m.params}</div>`;
      b.onclick = () => { model = m.id; updateModelUI(); loadModels(); toggleMenu(false); };
      msmenu.appendChild(b);
    }
    if (!MODELS[model]) model = Object.keys(MODELS)[0] || model;
    updateModelUI();
  }

  function updateModelUI() {
    const m = MODELS[model];
    msname.textContent = m ? m.label.replace(/ V\d\.\d$/, '') : 'Novius';
    curbadge.textContent = m ? (m.ready ? m.badge : 'not trained') : '';
    curbadge.classList.toggle('off', m ? !m.ready : true);
  }

  function toggleMenu(open) { msel.classList.toggle('open', open); }
  msbtn.onclick = (e) => { e.stopPropagation(); toggleMenu(!msel.classList.contains('open')); };
  document.addEventListener('click', (e) => { if (!e.target.closest('.mselect')) toggleMenu(false); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') toggleMenu(false); });

  /* ---------- messages ---------- */
  function welcome() {
    feed.innerHTML = `<div class="welcome"><div class="orb">${MARK.replace('16', '30').replace('16', '30')}</div>` +
      `<h1>Welcome to Novius</h1><div class="sub">V4.0 &middot; four models &middot; learns from your chats</div>` +
      `<p>Pick a model in the top bar, then say anything. Your chats are saved here.</p>` +
      `<div class="chips">${CHIPS.map((c) =>
        `<button class="chip" data-q="${c[1]}"><small>${c[0]}</small>${c[1]}</button>`).join('')}</div></div>`;
  }

  function addUser(text) {
    const w = document.querySelector('.welcome'); if (w) w.remove();
    const row = document.createElement('div'); row.className = 'row user';
    const b = document.createElement('div'); b.className = 'bubble'; b.textContent = text;
    row.appendChild(b); feed.appendChild(row); scroll(true);
  }

  function addAI() {
    const row = document.createElement('div'); row.className = 'row ai';
    const m = MODELS[model];
    const mlabel = m ? m.label : 'Novius';
    row.innerHTML = `<div class="avatar">${MARK}</div><div class="content">` +
      `<div class="who">Novius <span>&middot; ${mlabel}</span></div>` +
      `<div class="tcard" role="status"><video class="tvid" muted loop playsinline preload="auto" src="/assets/thinking.mp4"></video>` +
      `<div class="torb"></div><div class="tlabel"><span class="tt">Thinking</span>` +
      `<span class="tmeta"><span class="tsec">0.0</span>s</span></div></div>` +
      `<div class="status"></div>` +
      `<details class="think" style="display:none"><summary>${CHEV}<span class="sum">Thought process</span></summary>` +
      `<div class="tbody"></div></details>` +
      `<div class="reply md"></div><span class="cursor"></span></div>`;
    const v = row.querySelector('.tvid'); v.muted = true;
    v.onerror = () => row.querySelector('.tcard').classList.add('novideo');
    if (!matchMedia('(prefers-reduced-motion: reduce)').matches) v.play().catch(() => {});
    feed.appendChild(row); scroll(true);
    return row;
  }

  function renderStored(messages) {
    feed.innerHTML = '';
    if (!messages.length) { welcome(); return; }
    for (const m of messages) {
      if (m.role === 'user') { addUser(m.text); continue; }
      const row = document.createElement('div');
      row.className = 'row ai';
      row.innerHTML = `<div class="avatar">${MARK}</div><div class="content">` +
        `<div class="who">Novius</div><div class="reply md"></div></div>`;
      const el = row.querySelector('.reply');
      el.innerHTML = md(m.text);
      const cp = document.createElement('button');
      cp.className = 'copy'; cp.type = 'button';
      cp.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 012-2h9"/></svg><span>Copy</span>';
      cp.onclick = () => {
        navigator.clipboard.writeText(m.text);
        cp.lastChild.textContent = 'Copied!';
        setTimeout(() => cp.lastChild.textContent = 'Copy', 1200);
      };
      row.querySelector('.content').appendChild(cp);
      feed.appendChild(row);
    }
    scroll(true);
  }

  /* ---------- chats ---------- */
  async function loadChats() {
    const q = search.value.trim();
    const chats = await (await fetch('/api/chats' + (q ? '?q=' + encodeURIComponent(q) : ''))).json();
    chatlist.innerHTML = '';
    if (!chats.length) {
      chatlist.innerHTML = '<div class="noresults">' + (q ? 'No chats match your search' : 'No chats yet - say something!') + '</div>';
      return;
    }
    for (const c of chats) {
      const d = document.createElement('div');
      d.className = 'chatcard' + (c.id === currentChat ? ' sel' : '');
      d.innerHTML = `<div class="ct"></div><div class="cp"></div>` +
        `<div class="cactions"><button class="cbtn ren" title="Rename">` +
        `<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 3a2.8 2.8 0 114 4L7.5 20.5 2 22l1.5-5.5z"/></svg></button>` +
        `<button class="cbtn del" title="Delete">` +
        `<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><path d="M18 6L6 18M6 6l12 12"/></svg></button></div>`;
      d.querySelector('.ct').textContent = c.title;
      d.querySelector('.cp').textContent = c.preview;
      d.onclick = (e) => { if (e.target.closest('.cbtn')) return; openChat(c.id); };
      d.querySelector('.ren').onclick = () => renameChat(c);
      d.querySelector('.del').onclick = () => deleteChat(c);
      chatlist.appendChild(d);
    }
  }

  async function openChat(id) {
    currentChat = id;
    try {
      const c = await (await fetch('/api/chats/' + id)).json();
      renderStored(c.messages || []);
    } catch (e) { /* keep current view */ }
    loadChats();
    document.body.classList.remove('nav');
    input.focus();
  }

  function newChat() {
    currentChat = null;
    welcome();
    loadChats();
    document.body.classList.remove('nav');
    input.focus();
  }

  async function renameChat(c) {
    const title = prompt('Rename chat', c.title);
    if (!title || !title.trim()) return;
    await fetch('/api/chats/' + c.id + '/rename', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: title.trim() }),
    });
    toast('Chat renamed');
    loadChats();
  }

  async function deleteChat(c) {
    if (!confirm('Delete "' + c.title + '"?')) return;
    await fetch('/api/chats/' + c.id + '/delete', { method: 'POST' });
    toast('Chat deleted');
    if (currentChat === c.id) newChat(); else loadChats();
  }

  /* ---------- send / stream ---------- */
  async function send() {
    const text = input.value.trim();
    if (!text || busy) return;
    busy = true; input.value = ''; input.style.height = 'auto'; sync();
    addUser(text);
    const row = addAI(), q = (s) => row.querySelector(s);
    const card = q('.tcard'), label = q('.tt'), secEl = q('.tsec'), status = q('.status');
    const think = q('.think'), tbody = q('.tbody'), reply = q('.reply');
    const t0 = performance.now();
    let ended = false, got = false, raw = '';
    const timer = setInterval(() => { secEl.textContent = ((performance.now() - t0) / 1000).toFixed(1); }, 100);

    function endThink() {
      if (ended) return; ended = true; clearInterval(timer);
      const s = ((performance.now() - t0) / 1000).toFixed(1);
      if (performance.now() - t0 < 350) card.remove();
      else { card.classList.add('out'); setTimeout(() => card.remove(), 520); }
      if (tbody.textContent) {
        q('.sum').textContent = 'Thought for ' + s + 's';
        think.style.display = 'block';
      }
    }

    ctrl = new AbortController();
    try {
      const resp = await fetch('/api/chat', {
        method: 'POST', signal: ctrl.signal,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model, prompt: text, research: webMode, chat_id: currentChat }),
      });
      const reader = resp.body.getReader(), dec = new TextDecoder();
      let buf = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let i;
        while ((i = buf.indexOf('\n')) >= 0) {
          const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1);
          if (!line) continue;
          const evt = JSON.parse(line);
          if (evt.type === 'chat_id') currentChat = evt.id;
          else if (evt.type === 'status') {
            if (evt.text) label.textContent = evt.text.replace(/[ .]+$/, '');
            else endThink();
          } else if (evt.type === 'think') tbody.textContent += evt.text;
          else if (evt.type === 'token') {
            endThink(); got = true; raw += evt.text;
            if (!pendingRender) {
              pendingRender = true;
              requestAnimationFrame(() => {
                pendingRender = false;
                reply.innerHTML = md(raw);
                const cur = row.querySelector('.cursor');
                if (cur) reply.appendChild(cur);
                scroll();
              });
            }
          } else if (evt.type === 'source') {
            const a = document.createElement('a');
            a.className = 'srclink'; a.href = evt.url;
            a.target = '_blank'; a.rel = 'noopener';
            a.textContent = 'Wikipedia source ↗';
            q('.content').appendChild(a); scroll();
          } else if (evt.type === 'learned') {
            const n = document.createElement('span');
            n.className = 'learnnote';
            n.textContent = '✦ learned from this chat';
            q('.content').appendChild(n);
          } else if (evt.type === 'error') { endThink(); status.textContent = evt.text; }
        }
      }
    } catch (e) {
      endThink();
      status.textContent = e.name === 'AbortError' ? 'Stopped.' : 'Connection error: ' + e;
    }
    endThink();
    const cur = row.querySelector('.cursor'); if (cur) cur.remove();
    if (got) {
      reply.innerHTML = md(raw);
      const cp = document.createElement('button');
      cp.className = 'copy'; cp.type = 'button';
      cp.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 012-2h9"/></svg><span>Copy</span>';
      cp.onclick = () => {
        navigator.clipboard.writeText(raw);
        cp.lastChild.textContent = 'Copied!';
        setTimeout(() => cp.lastChild.textContent = 'Copy', 1200);
      };
      q('.content').appendChild(cp);
    }
    if (!got && !status.textContent && !tbody.textContent) status.textContent = '(no output)';
    busy = false; ctrl = null; sync(); input.focus();
    loadChats();
  }

  /* ---------- events ---------- */
  sendBtn.onclick = () => busy ? ctrl && ctrl.abort() : send();
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); if (!busy) send(); }
  });
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 150) + 'px';
    sync();
  });

  feed.addEventListener('click', (e) => {
    const chip = e.target.closest('.chip');
    if (chip) { input.value = chip.dataset.q; send(); return; }
    const cb = e.target.closest('.cbcopy');
    if (cb) {
      const code = cb.closest('.codeblock').querySelector('code').textContent;
      navigator.clipboard.writeText(code);
      cb.textContent = 'Copied!';
      setTimeout(() => cb.textContent = 'Copy', 1200);
    }
  });

  $('newchat').onclick = () => { if (ctrl) ctrl.abort(); newChat(); };
  $('burger').onclick = () => document.body.classList.toggle('nav');
  $('scrim').onclick = () => document.body.classList.remove('nav');
  $('signout').onclick = async () => {
    await fetch('/api/logout', { method: 'POST' });
    location.reload();
  };

  let searchTimer = null;
  search.addEventListener('input', () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(loadChats, 200);
  });
  search.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') { search.value = ''; loadChats(); search.blur(); }
  });
  $('searchbtn').onclick = () => { search.focus(); loadChats(); };

  document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
      e.preventDefault();
      search.focus();
    }
  });

  /* ---------- ambient parallax ---------- */
  if (!matchMedia('(prefers-reduced-motion: reduce)').matches) {
    window.addEventListener('pointermove', (e) => {
      const x = e.clientX / innerWidth - .5, y = e.clientY / innerHeight - .5;
      document.querySelectorAll('.aurora i').forEach((b, i) => {
        const f = (i + 1) * 12;
        b.style.translate = `${x * f}px ${y * f}px`;
      });
    });
  }

  /* ---------- init ---------- */
  (async () => {
    try {
      const me = await (await fetch('/api/auth/me')).json();
      whoEl.textContent = me.name || me.email;
      const av = document.querySelector('.meav');
      if (av) av.textContent = (me.name || me.email || '?').trim().charAt(0).toUpperCase();
    } catch (e) { location.reload(); }
  })();
  welcome(); sync(); loadModels(); loadChats(); input.focus();
})();
