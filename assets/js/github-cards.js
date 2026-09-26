// Renders GitHub user/repo cards from the public REST API (replaces github-readme-stats).
(function () {
  const API = "https://api.github.com";
  const TTL = 6 * 60 * 60 * 1000;

  async function getJSON(path) {
    const key = "gh-cache:" + path;
    try {
      const hit = JSON.parse(localStorage.getItem(key));
      if (hit && Date.now() - hit.t < TTL) return hit.d;
    } catch (e) {}
    const res = await fetch(API + path, { headers: { Accept: "application/vnd.github+json" } });
    if (!res.ok) throw new Error(res.status);
    const data = await res.json();
    try {
      localStorage.setItem(key, JSON.stringify({ t: Date.now(), d: data }));
    } catch (e) {}
    return data;
  }

  const fmt = (n) => (n >= 1000 ? (n / 1000).toFixed(1).replace(/\.0$/, "") + "k" : String(n));

  function stat(icon, value, label) {
    const s = document.createElement("span");
    s.title = label;
    s.innerHTML = `<i class="${icon}"></i> `;
    s.append(value);
    return s;
  }

  function fill(card, desc, stats) {
    card.querySelector(".gh-card-desc").textContent = desc || "";
    card.querySelector(".gh-card-meta").replaceChildren(...stats);
  }

  document.querySelectorAll("[data-gh-repo]").forEach(async (card) => {
    try {
      const r = await getJSON("/repos/" + card.dataset.ghRepo);
      const stats = [];
      if (r.language) {
        const lang = document.createElement("span");
        lang.innerHTML = '<i class="fa-solid fa-code"></i> ';
        lang.append(r.language);
        stats.push(lang);
      }
      stats.push(stat("fa-solid fa-star", fmt(r.stargazers_count), "Stars"));
      stats.push(stat("fa-solid fa-code-fork", fmt(r.forks_count), "Forks"));
      fill(card, r.description, stats);
    } catch (e) {
      fill(card, "View on GitHub", []);
    }
  });

  document.querySelectorAll("[data-gh-user]").forEach(async (card) => {
    const user = card.dataset.ghUser;
    try {
      const [u, repos] = await Promise.all([getJSON("/users/" + user), getJSON("/users/" + user + "/repos?per_page=100&type=owner")]);
      const stars = repos.reduce((sum, r) => sum + r.stargazers_count, 0);
      fill(card, u.bio || u.name, [
        stat("fa-solid fa-star", fmt(stars), "Total stars"),
        stat("fa-solid fa-book", fmt(u.public_repos), "Public repositories"),
        stat("fa-solid fa-users", fmt(u.followers), "Followers"),
      ]);
    } catch (e) {
      fill(card, "View on GitHub", []);
    }
  });
})();
