(function () {
  'use strict';

  var root = document.documentElement;
  var state = { data: null, lang: 'zh', rank: null, pubBody: null, pubSection: null };

  var UI = {
    zh: {
      lang: { label: '语言' },
      theme: { label: '主题', light: '浅色', dark: '深色' },
      nav: {
        label: '章节',
        research: '研究兴趣',
        experience: '教育与工作经历',
        projects: '科研项目',
        publications: '代表性成果',
        students: '师生成果',
        service: '学术服务与荣誉'
      },
      profile: { label: '个人资料', metrics: '主要指标' },
      error: { load: '无法加载 data/site.js。请确认它和本页在同一目录。' },
      rank: { label: '分区体系', print: '分区体系：{name}' },
      pub: { count: '{label} {n} 篇', countPlural: '{label} {n} 篇', sep: '；' },
      print: { cv: 'CV' },
      footer: { updated: '最近更新：{date}' }
    },
    en: {
      lang: { label: 'Language' },
      theme: { label: 'Theme', light: 'Light', dark: 'Dark' },
      nav: {
        label: 'Sections',
        research: 'Research',
        experience: 'Experience',
        projects: 'Projects',
        publications: 'Publications',
        students: 'With Students',
        service: 'Service'
      },
      profile: { label: 'Profile', metrics: 'Metrics' },
      error: { load: 'Cannot load data/site.js. Make sure it sits next to this page.' },
      rank: { label: 'Ranking system', print: 'Ranking: {name}' },
      pub: { count: '{label} · {n} paper', countPlural: '{label} · {n} papers', sep: '; ' },
      print: { cv: 'CV' },
      footer: { updated: 'Last updated: {date}' }
    }
  };

  var ICONS = {
    scholar: '<path d="M12 3.5 1.8 8.6 12 13.7l10.2-5.1L12 3.5Z"/><path d="M5.6 11v4.4c0 1.8 2.9 3.2 6.4 3.2s6.4-1.4 6.4-3.2V11"/>',
    github: '<path d="M9 19.5c-3.5 1-3.5-1.8-5-2.2m10 4.2v-3.4c0-1 .1-1.4-.5-2 2.3-.3 4.5-1.2 4.5-5a3.9 3.9 0 0 0-1.1-2.7 3.6 3.6 0 0 0-.1-2.7s-1.1-.3-3.5 1.4a8.6 8.6 0 0 0-4.6 0C6.3 5.4 5.2 5.7 5.2 5.7a3.6 3.6 0 0 0-.1 2.7A3.9 3.9 0 0 0 4 11.1c0 3.8 2.2 4.7 4.5 5-.4.4-.5.9-.5 1.6V21"/>',
    orcid: '<circle cx="12" cy="12" r="9"/><path d="M9.2 8.6v7"/><circle cx="9.2" cy="6.9" r=".7" fill="currentColor" stroke="none"/><path d="M12.4 8.6h2.3a3.5 3.5 0 0 1 0 7h-2.3v-7Z"/>',
    researchgate: '<circle cx="12" cy="12" r="9"/><path d="M8.8 16.2V7.8h3.4a2.7 2.7 0 0 1 0 5.4H8.8"/><path d="M11.2 13.2 15.2 16.2"/>',
    cv: '<path d="M13.5 3H6.8A1.8 1.8 0 0 0 5 4.8v14.4A1.8 1.8 0 0 0 6.8 21h10.4a1.8 1.8 0 0 0 1.8-1.8V8.5L13.5 3Z"/><path d="M13.5 3v5.5H19"/>',
    clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7.5V12l3 2"/>',
    mail: '<rect x="3.5" y="5.5" width="17" height="13" rx="1.5"/><path d="m4.5 7 7.5 5.5L19.5 7"/>'
  };

  function icon(name) {
    return '<svg viewBox="0 0 24 24" aria-hidden="true">' + (ICONS[name] || '') + '</svg>';
  }

  function el(tag, attrs) {
    var node = document.createElement(tag);
    if (!attrs) return node;
    Object.keys(attrs).forEach(function (key) {
      var value = attrs[key];
      if (key === 'text') node.textContent = value;
      else if (key === 'html') node.innerHTML = value;
      else if (key === 'class') node.className = value;
      else node.setAttribute(key, value);
    });
    return node;
  }

  layui.use(['i18n'], function () {
    var $ = layui.$;
    var i18n = layui.i18n;

    var messages = i18n.config.messages || (i18n.config.messages = {});
    (messages['zh-CN'] = messages['zh-CN'] || {}).ui = UI.zh;
    (messages.en = messages.en || {}).ui = UI.en;

    function $t(key, data) {
      return i18n.$t('ui.' + key, data);
    }

    function T(value) {
      if (value === null || value === undefined) return '';
      if (typeof value === 'string') return value;
      return value[state.lang] || value.en || value.zh || '';
    }

    function read(key) {
      try { return localStorage.getItem(key); } catch (e) { return null; }
    }

    function store(key, value) {
      try { localStorage.setItem(key, value); } catch (e) {}
    }

    var navEl = document.getElementById('nav');
    var sidebarEl = document.getElementById('sidebar');
    var contentEl = document.getElementById('content-body');
    var copyrightEl = document.getElementById('copyright');
    var updatedEl = document.getElementById('updated');
    var errorEl = document.getElementById('load-error');
    var langSelect = document.getElementById('lang-toggle');
    var themeMeta = document.querySelector('meta[name="theme-color"]');
    var themeButtons = document.querySelectorAll('.theme-option');

    function setTheme(theme) {
      var next = theme === 'dark' ? 'dark' : 'light';
      root.dataset.theme = next;
      store('theme', next);

      Array.prototype.forEach.call(themeButtons, function (button) {
        button.setAttribute('aria-pressed', String(button.dataset.themeValue === next));
      });
      if (themeMeta) themeMeta.setAttribute('content', next === 'dark' ? '#0f1113' : '#ffffff');
    }

    Array.prototype.forEach.call(themeButtons, function (button) {
      button.addEventListener('click', function () {
        setTheme(button.dataset.themeValue);
      });
    });

    setTheme(root.dataset.theme);

    function fillStaticText() {
      $('[data-i18n]').each(function () {
        $(this).text($t(this.getAttribute('data-i18n')));
      });
      $('[data-i18n-label]').each(function () {
        this.setAttribute('aria-label', $t(this.getAttribute('data-i18n-label')));
      });
    }

    function setLang(lang, persist) {
      state.lang = lang === 'en' ? 'en' : 'zh';
      root.dataset.lang = state.lang;
      root.lang = state.lang === 'zh' ? 'zh-CN' : 'en';
      i18n.set({ locale: state.lang === 'zh' ? 'zh-CN' : 'en' });
      if (persist !== false) store('lang', state.lang);
      if (langSelect) langSelect.value = state.lang;

      fillStaticText();
      paintQuip();
      render();
    }

    if (langSelect) {
      langSelect.addEventListener('change', function () {
        setLang(langSelect.value);
      });
    }

    function roleLine(profile) {
      var parts = [T(profile.org), T(profile.title)].filter(Boolean);
      var node = el('p', { class: 'role' });
      parts.forEach(function (part) {
        node.appendChild(el('span', { class: 'role-line', text: part }));
      });
      return node;
    }

    function renderSidebar() {
      var p = state.data.profile;
      sidebarEl.innerHTML = '';

      sidebarEl.appendChild(el('img', {
        class: 'portrait',
        src: p.photo,
        alt: T(p.name),
        width: '600',
        height: '750'
      }));

      var main = el('div', { class: 'side-main' });
      var identity = el('div', { class: 'identity' });
      identity.appendChild(el('h1', { text: T(p.name) }));
      identity.appendChild(roleLine(p));
      main.appendChild(identity);

      var metrics = el('dl', { class: 'metrics', 'aria-label': $t('profile.metrics') });
      (p.metrics || []).forEach(function (metric) {
        var row = el('div');
        row.appendChild(el('dt', { text: T(metric.label) }));
        row.appendChild(el('dd', { text: metric.value }));
        metrics.appendChild(row);
      });
      main.appendChild(metrics);

      var contacts = el('ul', { class: 'side-contact' });
      (p.contacts || []).forEach(function (contact) {
        var node = el('a', { href: contact.href });
        if (contact.external) {
          node.setAttribute('target', '_blank');
          node.setAttribute('rel', 'noopener');
        }
        node.innerHTML = icon(contact.icon) + '<span></span>';
        node.querySelector('span').textContent = T(contact.label);

        var item = el('li');
        item.appendChild(node);
        contacts.appendChild(item);
      });

      var printItem = el('li');
      var printBtn = el('button', { class: 'cv-print', type: 'button', 'aria-label': 'CV (PDF)' });
      printBtn.addEventListener('click', function () { window.print(); });
      printBtn.innerHTML = icon('cv') + '<span>CV (PDF)</span>';
      printItem.appendChild(printBtn);
      contacts.appendChild(printItem);
      main.appendChild(contacts);

      if (p.collaboration) {
        var card = el('section', { class: 'side-block side-card' });
        card.appendChild(el('p', { class: 'card-title', text: T(p.collaboration.title) }));
        card.appendChild(el('p', { class: 'card-text', text: T(p.collaboration.text) }));
        var mail = el('a', { class: 'card-link', href: 'mailto:' + p.collaboration.email });
        mail.innerHTML = icon('mail') + '<span></span>';
        mail.querySelector('span').textContent = p.collaboration.email;
        card.appendChild(mail);
        main.appendChild(card);
      }

      sidebarEl.appendChild(main);

      if (p.meta) {
        var meta = el('div', { class: 'side-meta' });
        var item = el('p', { class: 'meta-item' });
        item.innerHTML = icon('clock') + '<span class="local-time"></span>';
        meta.appendChild(item);
        sidebarEl.appendChild(meta);
        tickClock();
      }
    }

    function bodyWrap() {
      return el('div', { class: 'section-body' });
    }

    function renderInterests(section) {
      var body = bodyWrap();
      var list = el('dl', { class: 'interests' });

      (section.items || []).forEach(function (item) {
        var row = el('div', { class: 'interest' });
        row.appendChild(el('dt', { text: T(item.term) }));
        row.appendChild(el('dd', { text: T(item.desc) }));
        list.appendChild(row);
      });

      body.appendChild(list);
      return body;
    }

    function timeValue(text, part) {
      var raw = String(text || '').trim();
      if (!raw) return null;
      var piece = raw.split(/\s*[–—-]\s*/)[part || 0] || '';
      var match = piece.match(/(\d{4})\D*(\d{1,2})?/);
      if (!match) return null;
      return Number(match[1]) * 100 + (match[2] ? Number(match[2]) : 0);
    }

    function byTime(a, b) {
      var startA = timeValue(a.date, 0);
      var startB = timeValue(b.date, 0);
      if (startA == null && startB == null) return 0;
      if (startA == null) return 1;
      if (startB == null) return -1;
      if (startA !== startB) return startB - startA;
      var endA = timeValue(a.date, 1);
      var endB = timeValue(b.date, 1);
      if (endA == null && endB == null) return 0;
      if (endA == null) return 1;
      if (endB == null) return -1;
      return endB - endA;
    }

    function renderTimeline(section) {
      var body = bodyWrap();
      var list = el('ul', { class: 'rows' });

      (section.items || []).slice().sort(byTime).forEach(function (item) {
        var row = el('li', { class: 'row' });
        row.appendChild(el('div', { class: 'when', text: item.date }));

        var what = el('div', { class: 'what' });
        var primary = el('p', { class: 'primary' });
        primary.appendChild(document.createTextNode(T(item.org)));
        primary.appendChild(el('span', { class: 'duty', text: T(item.duty) }));
        what.appendChild(primary);

        var advisorName = item.advisor ? T(item.advisor.name).trim() : '';
        if (advisorName) {
          var advisor = el('p', { class: 'advisor' });
          var advisorLabel = T(item.advisor.label).trim();
          if (advisorLabel) advisor.appendChild(document.createTextNode(advisorLabel + ' '));
          var advisorUrl = String(item.advisor.url || '').trim();
          if (advisorUrl) {
            advisor.appendChild(el('a', {
              href: advisorUrl,
              target: '_blank',
              rel: 'noopener',
              text: advisorName
            }));
          } else {
            advisor.appendChild(document.createTextNode(advisorName));
          }
          what.appendChild(advisor);
        }

        row.appendChild(what);
        list.appendChild(row);
      });

      body.appendChild(list);
      return body;
    }

    function projectRoleLabel(role) {
      if (role === 'join') return state.lang === 'zh' ? '参与' : 'Participant';
      if (role === 'lead') return state.lang === 'zh' ? '主持' : 'PI';
      return '';
    }

    function renderProjects(section) {
      var body = bodyWrap();
      var list = el('ul', { class: 'rows' });

      (section.items || []).slice().sort(byTime).forEach(function (item) {
        var row = el('li', { class: 'row' });
        row.appendChild(el('div', { class: 'when', text: item.date }));

        var what = el('div', { class: 'what' });
        what.appendChild(el('p', { class: 'primary', text: T(item.title) }));

        var meta = el('p', { class: 'project-meta' });
        meta.appendChild(el('span', { class: 'funder', text: T(item.funder) }));
        var amount = T(item.amount).trim();
        var role = projectRoleLabel(item.role);
        var tail = [amount, role].filter(Boolean).join(' · ');
        if (tail) meta.appendChild(el('span', { class: 'amount', text: tail }));
        what.appendChild(meta);

        row.appendChild(what);
        list.appendChild(row);
      });

      body.appendChild(list);
      return body;
    }

    function renderStudents(section) {
      var body = bodyWrap();
      var kinds = [
        { key: 'contest', zh: '学生竞赛', en: 'Student Competitions' },
        { key: 'paper', zh: '师生论文', en: 'Papers with Students' }
      ];
      var groups = { contest: [], paper: [] };

      (section.items || []).forEach(function (item) {
        var key = item.kind === 'paper' ? 'paper' : 'contest';
        groups[key].push(item);
      });

      kinds.forEach(function (kind) {
        if (!groups[kind.key].length) return;
        body.appendChild(el('h3', { class: 'group', text: T(kind) }));
        var list = el('ol', { class: 'pubs' });
        groups[kind.key].forEach(function (item) {
          var entry = el('li');
          entry.appendChild(el('p', { class: 'pub-title', text: T(item.title) }));
          var names = T(item.students).trim();
          if (names) entry.appendChild(el('p', { class: 'pub-authors', text: names }));
          list.appendChild(entry);
        });
        body.appendChild(list);
      });

      return body;
    }

    var titleBeforePrint = '';

    function printTitle() {
      var profile = state.data && state.data.profile;
      var name = profile ? T(profile.name) : '';
      var rankName = state.pubSection ? rankSystemLabel(state.pubSection) : '';
      var title = name ? name + ' - ' + $t('print.cv') : $t('print.cv');
      if (rankName) title += ' - ' + rankName;
      return title;
    }

    window.addEventListener('beforeprint', function () {
      titleBeforePrint = document.title;
      document.title = printTitle();
    });
    window.addEventListener('afterprint', function () {
      if (titleBeforePrint) document.title = titleBeforePrint;
    });

    function rankSystemLabel(section) {
      var found = null;
      ((section.rank || {}).systems || []).forEach(function (system) {
        if (system.key === state.rank) found = system;
      });
      return found ? T(found.label) : '';
    }

    var ZONE_LABELS = {
      1: { zh: '一区', en: 'Q1' },
      2: { zh: '二区', en: 'Q2' },
      3: { zh: '三区', en: 'Q3' },
      4: { zh: '四区', en: 'Q4' }
    };

    function zoneRank(zone) {
      var value = Number(zone);
      return value === 1 || value === 2 || value === 3 || value === 4 ? value : 9;
    }

    function currentZone(item) {
      var rank = item ? item.rank : null;
      if (rank === null || rank === undefined) return { state: 'none' };
      if (typeof rank === 'number') {
        var numeric = zoneRank(rank);
        return numeric < 9 ? { state: 'zone', value: numeric } : { state: 'drop' };
      }
      if (!Object.prototype.hasOwnProperty.call(rank, state.rank) || rank[state.rank] === '' || rank[state.rank] == null) {
        return { state: 'missing' };
      }
      var value = zoneRank(rank[state.rank]);
      return value < 9 ? { state: 'zone', value: value } : { state: 'drop' };
    }

    function itemBucket(item) {
      if (state.rank === 'ei') return item && item.ei ? 'ei' : 0;
      var zone = currentZone(item);
      return zone.state === 'zone' ? zone.value : 0;
    }

    function ccfLabel(item) {
      if (item && item.ccf === 1) return 'CCF-A';
      if (item && item.ccf === 2) return 'CCF-B';
      return '';
    }

    function bucketLabel(section, bucket) {
      if (bucket === 'ei') return 'EI';
      var zones = section && section.rank && section.rank.zones;
      if (zones) {
        for (var i = 0; i < zones.length; i++) {
          if (zones[i].id === bucket && zones[i].label) return T(zones[i].label);
        }
      }
      return T(ZONE_LABELS[bucket] || '');
    }

    function bucketKeys(buckets) {
      return Object.keys(buckets).map(function (key) {
        return key === 'ei' ? key : Number(key);
      }).sort(function (a, b) {
        var left = a === 'ei' ? 99 : a;
        var right = b === 'ei' ? 99 : b;
        return left - right;
      });
    }

    function noteCounts(section, buckets) {
      var parts = [];
      bucketKeys(buckets).forEach(function (bucket) {
        var count = buckets[bucket].length;
        if (!count) return;
        parts.push($t(count === 1 ? 'pub.count' : 'pub.countPlural', {
          label: bucketLabel(section, bucket),
          n: count
        }));
      });
      return parts.join($t('pub.sep'));
    }

    function selfNames(section) {
      var names = (section && section.self) || [];
      return Array.isArray(names) ? names : [names];
    }

    function escapeHtml(value) {
      return String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    function authorTokens(item) {
      return String(item.authors || '').split(',').map(function (name) {
        return name.trim();
      }).filter(Boolean);
    }

    function authorName(token) {
      return token.replace(/\*/g, '').replace(/\.$/, '').trim();
    }

    function selfIndex(section, item) {
      var names = selfNames(section);
      var tokens = authorTokens(item);
      for (var i = 0; i < tokens.length; i++) {
        if (names.indexOf(authorName(tokens[i])) !== -1) return i;
      }
      return -1;
    }

    function renderAuthors(section, item) {
      var index = selfIndex(section, item);

      return authorTokens(item).map(function (token, i) {
        var corresponding = token.indexOf('*') !== -1;
        var name = authorName(token);
        var period = /\.$/.test(token.trim());
        var html = escapeHtml(name);

        if (i === index) {
          html = '<span class="' + (i === 0 ? 'me' : 'me-plain') + '">' + html + '</span>';
        }
        if (corresponding) html += '<sup>*</sup>';
        if (period) html += '.';
        return html;
      }).join(', ');
    }

    function itemOrder(section, item) {
      if (/major\s+revision/i.test(item.venue || '')) return 3;
      var index = selfIndex(section, item);
      if (index === 0) return 0;
      if (index > 0) return 1;
      return 2;
    }

    function pubYear(item) {
      var match = String(item.venue || '').match(/\b(?:19|20)\d{2}\b/);
      return match ? Number(match[0]) : null;
    }

    function sortItems(section, items) {
      return (items || []).slice().sort(function (a, b) {
        var order = itemOrder(section, a) - itemOrder(section, b);
        if (order) return order;
        var yearA = pubYear(a);
        var yearB = pubYear(b);
        if (yearA == null && yearB == null) return 0;
        if (yearA == null) return -1;
        if (yearB == null) return 1;
        return yearB - yearA;
      });
    }

    function buildSeg(className, label, options) {
      var wrap = el('div', { class: 'seg ' + className, role: 'group', 'aria-label': label });
      wrap.style.setProperty('--seg-count', String(options.length));
      wrap.appendChild(el('span', { class: 'seg-thumb', 'aria-hidden': 'true' }));

      var index = 0;
      options.forEach(function (option, i) {
        if (option.active) index = i;
        var button = el('button', {
          class: 'seg-option',
          type: 'button',
          'aria-pressed': String(option.active),
          text: option.text
        });
        Object.keys(option.attrs || {}).forEach(function (key) {
          button.setAttribute(key, option.attrs[key]);
        });
        button.addEventListener('click', option.onClick);
        wrap.appendChild(button);
      });

      wrap.style.setProperty('--seg-index', String(index));
      return wrap;
    }

    function buildRankSwitch(section) {
      var cfg = section.rank || {};
      var systems = cfg.systems || [];
      if (!systems.length) return null;

      return buildSeg('seg-rank', T(cfg.label) || $t('rank.label'), systems.map(function (system) {
        return {
          active: system.key === state.rank,
          text: T(system.label),
          attrs: { 'data-rank': system.key },
          onClick: function () { setRank(system.key); }
        };
      }));
    }

    function refillPublications(selector) {
      if (!(state.pubBody && state.pubSection)) return;
      fillPublications(state.pubBody, state.pubSection);
      if (!selector) return;
      var again = state.pubBody.querySelector(selector);
      if (again) again.focus({ preventScroll: true });
    }

    function setRank(key) {
      if (state.rank === key) return;
      state.rank = key;
      store('rank', key);

      var focused = document.activeElement && document.activeElement.dataset
        ? document.activeElement.dataset.rank
        : null;
      refillPublications(focused ? '.seg-option[data-rank="' + focused + '"]' : null);
    }

    function pubList(section, items) {
      var list = el('ol', { class: 'pubs' });

      sortItems(section, items).forEach(function (item) {
        var entry = el('li');
        entry.appendChild(el('p', { class: 'pub-title', text: item.title }));
        entry.appendChild(el('p', { class: 'pub-authors', html: renderAuthors(section, item) }));
        var venue = item.venue || '';
        var mark = ccfLabel(item);
        if (mark) venue = venue ? venue.replace(/\s+$/, '') + ' (' + mark + ')' : '(' + mark + ')';
        entry.appendChild(el('p', { class: 'pub-venue', text: venue }));
        list.appendChild(entry);
      });

      return list;
    }

    function renderPublications(section) {
      var body = bodyWrap();
      state.pubBody = body;
      state.pubSection = section;
      fillPublications(body, section);
      return body;
    }

    function resolveRank(section) {
      var cfg = section.rank || {};
      var systems = cfg.systems || [];
      if (!systems.length) return;

      var keys = systems.map(function (system) { return system.key; });
      if (keys.indexOf(state.rank) !== -1) return;

      state.rank = keys.indexOf(cfg.default) === -1 ? keys[0] : cfg.default;
    }

    function fillPublications(body, section) {
      body.innerHTML = '';

      resolveRank(section);

      var buckets = state.rank === 'ei'
        ? { ei: [] }
        : { 1: [], 2: [], 3: [], 4: [] };
      (section.items || []).forEach(function (item) {
        var bucket = itemBucket(item);
        if (bucket) buckets[bucket].push(item);
      });

      var bar = el('div', { class: 'pub-bar' });
      var note = el('p', { class: 'note', html: T(section.note) });

      var counts = noteCounts(section, buckets);
      if (counts) note.appendChild(el('span', { class: 'note-count', text: counts }));

      var rankName = rankSystemLabel(section);
      if (rankName) {
        note.appendChild(el('span', {
          class: 'note-rank print-only',
          text: $t('rank.print', { name: rankName })
        }));
      }

      bar.appendChild(note);

      var switches = el('div', { class: 'pub-switches' });
      var switcher = buildRankSwitch(section);
      if (switcher) switches.appendChild(switcher);
      bar.appendChild(switches);

      body.appendChild(bar);

      bucketKeys(buckets).forEach(function (bucket) {
        if (!buckets[bucket].length) return;

        var heading = el('h3', { class: 'group' });
        heading.appendChild(document.createTextNode(bucketLabel(section, bucket)));
        heading.appendChild(el('span', {
          class: 'print-only',
          text: ' · ' + buckets[bucket].length + (state.lang === 'zh' ? ' 篇' : (buckets[bucket].length === 1 ? ' paper' : ' papers'))
        }));
        body.appendChild(heading);
        body.appendChild(pubList(section, buckets[bucket]));
      });

      return body;
    }

    function renderService(section) {
      var body = bodyWrap();
      var cols = el('div', { class: 'service-cols' });

      (section.columns || []).forEach(function (column) {
        var wrap = el('div');
        wrap.appendChild(el('h3', { text: T(column.title) }));

        var list = el('ul');
        (column.items || []).forEach(function (item) {
          list.appendChild(el('li', { text: T(item) }));
        });

        wrap.appendChild(list);
        cols.appendChild(wrap);
      });

      body.appendChild(cols);
      return body;
    }

    var RENDERERS = {
      interests: renderInterests,
      timeline: renderTimeline,
      projects: renderProjects,
      publications: renderPublications,
      students: renderStudents,
      service: renderService
    };

    function renderContent() {
      contentEl.innerHTML = '';
      contentEl.appendChild(el('p', { class: 'lede', text: T(state.data.intro) }));

      (state.data.sections || []).forEach(function (section) {
        var node = el('section', { class: 'section', id: section.id });
        node.appendChild(el('h2', { text: T(section.title) }));

        var renderer = RENDERERS[section.type];
        if (renderer) node.appendChild(renderer(section));

        contentEl.appendChild(node);
      });
    }

    function updatedLine() {
      var now = new Date();
      var year = now.getFullYear();
      var month = now.getMonth();
      var day = now.getDate();
      var date = state.lang === 'zh'
        ? year + ' 年 ' + (month + 1) + ' 月 ' + day + ' 日'
        : ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'][month] + ' ' + day + ', ' + year;
      return $t('footer.updated', { date: date });
    }

    function renderFooter() {
      var year = String(new Date().getFullYear());
      if (copyrightEl) copyrightEl.textContent = '© ' + year + ' HomePage · CV';
      if (updatedEl) updatedEl.textContent = updatedLine();
    }

    function applyHead() {
      var profile = state.data.profile || {};
      var name = profile.name || {};
      var en = typeof name === 'string' ? name : (name.en || '');
      var zh = typeof name === 'string' ? '' : (name.zh || '');
      var configured = state.data.site && state.data.site.title;
      var title = typeof configured === 'string' ? configured.trim() : '';
      if (title) document.title = title;

      var description = document.querySelector('meta[name="description"]');
      if (description && state.data.intro) {
        description.setAttribute('content', T(state.data.intro));
      }

      var author = document.querySelector('meta[name="author"]');
      if (author && (en || zh)) author.setAttribute('content', en || zh);
    }

    function render() {
      if (!state.data) return;
      applyHead();
      renderSidebar();
      renderContent();
      renderFooter();
      updateActive();
    }

    function tickClock() {
      var nodes = document.querySelectorAll('.local-time');
      if (!nodes.length) return;

      var meta = state.data && state.data.profile.meta;
      if (!meta) return;

      var hhmm;
      try {
        hhmm = new Intl.DateTimeFormat('en-GB', {
          hour: '2-digit',
          minute: '2-digit',
          hour12: false,
          timeZone: meta.timezone
        }).format(new Date());
      } catch (e) {
        hhmm = new Date().toTimeString().slice(0, 5);
      }

      Array.prototype.forEach.call(nodes, function (node) {
        node.textContent = meta.location + ' ' + hhmm;
      });
    }

    window.setInterval(tickClock, 30000);

    function updateActive() {
      var links = Array.prototype.slice.call(navEl.querySelectorAll('a'));
      var sections = links
        .map(function (link) { return document.getElementById(link.getAttribute('href').slice(1)); })
        .filter(Boolean);

      if (!sections.length) return;

      if (window.scrollY < 90) {
        links.forEach(function (link) { link.classList.remove('active'); });
        return;
      }

      var current = sections[0];
      sections.forEach(function (section) {
        if (section.getBoundingClientRect().top <= 140) current = section;
      });

      if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 8) {
        current = sections[sections.length - 1];
      }

      links.forEach(function (link) {
        link.classList.toggle('active', link.getAttribute('href') === '#' + current.id);
      });
    }

    var ticking = false;
    window.addEventListener('scroll', function () {
      if (ticking) return;
      ticking = true;
      window.requestAnimationFrame(function () {
        updateActive();
        ticking = false;
      });
    }, { passive: true });

    var quipEl = document.getElementById('quip');
    var quipText = '';
    var DEFAULT_QUIP = 'https://v1.hitokoto.cn/?c=d&c=i&c=k&min_length=12&max_length=16&encode=json';

    function paintQuip() {
      if (!quipEl) return;
      var show = !!quipText && state.lang === 'zh';
      quipEl.hidden = !show;
      quipEl.textContent = show ? quipText : '';
    }

    function loadQuip() {
      var quip = state.data && state.data.site && state.data.site.quip;
      if (quip === false || quip === '' || !quipEl || !window.fetch) return;

      var url = typeof quip === 'string' ? quip : DEFAULT_QUIP;

      fetch(url, { cache: 'no-store' })
        .then(function (response) {
          return response.ok ? response.json() : null;
        })
        .then(function (data) {
          if (!data || !data.hitokoto) return;
          quipText = '「' + data.hitokoto + '」';
          paintQuip();
        })
        .catch(function () {});
    }

    function loadJSON(path) {
      var files = window.HOMEPAGE && window.HOMEPAGE.files;
      if (!files || !Object.prototype.hasOwnProperty.call(files, path)) {
        return Promise.reject(new Error(path + ' 未加载'));
      }
      return Promise.resolve(files[path]);
    }

    function loadSite() {
      return loadJSON('data/site.js').then(function (site) {
        var sectionLoads = (site.sections || []).map(function (section) {
          return loadJSON(section.file).then(function (content) {
            return Object.assign({}, section, content);
          });
        });

        return Promise.all([loadJSON(site.profile), Promise.all(sectionLoads)])
          .then(function (results) {
            site.profile = results[0];
            site.sections = results[1];
            return site;
          });
      });
    }

    var savedRank = read('rank');
    if (savedRank) state.rank = savedRank;

    var savedLang = read('lang');
    var initialLang = savedLang === 'en' || savedLang === 'zh' ? savedLang : 'zh';
    setLang(initialLang, false);

    loadSite()
      .then(function (data) {
        state.data = data;
        var fallback = data.site && data.site.defaultLang;
        if (!savedLang && (fallback === 'zh' || fallback === 'en') && fallback !== state.lang) {
          setLang(fallback);
        } else {
          render();
        }
        loadQuip();
      })
      .catch(function (error) {
        if (errorEl) errorEl.hidden = false;
        if (window.console) window.console.error('Failed to load site data', error);
      });
  });
})();
